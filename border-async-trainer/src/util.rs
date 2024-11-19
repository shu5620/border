//! Utility function.
use crate::{
    actor_stats_fmt, ActorManager, ActorManagerConfig, AsyncTrainer, AsyncTrainerConfig, SyncModel,
};
use border_core::{record::TensorboardRecorder, Agent, Env, ReplayBufferBase, StepProcessorBase};
use crossbeam_channel::{bounded, unbounded};
use log::info;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    path::Path,
    sync::{Arc, Mutex},
};

/// Runs asynchronous training.
///
/// This function runs [`ActorManager`] and [`AsyncTrainer`] on threads.
/// These communicate using [`crossbeam_channel`]. Training logs are recorded for
/// tensorboard.
///
/// * `model_dir` - Directory where trained models and tensor board log will be saved.
/// * `agent_config` - Configuration of the agent to be trained.
/// * `agent_configs` - Configurations of agents for asynchronous sampling.
///   It must share the same structure of the model ([`SyncModel::ModelInfo`]),
///   while exploration parameters can be different.
/// * `env_config_train` - Configuration of the environment with which transitions are
///   sampled.
/// * `env_config_eval` - Configuration of the environment on which the agent being trained
///   is evaluated.
/// * `replay_buffer_config` - Configuration of the replay buffer.
/// * `actor_man_config` - Configuration of [`ActorManager`].
/// * `async_trainer_config` - Configuration of [`AsyncTrainer`].
pub fn train_async<A, E, R, S, P>(
    model_dir: &P,
    agent_config: &A::Config,
    agent_configs: &Vec<A::Config>,
    env_config_train: &E::Config,
    env_config_eval: &E::Config,
    step_proc_config: &S::Config,
    replay_buffer_config: &R::Config,
    actor_man_config: &ActorManagerConfig,
    async_trainer_config: &AsyncTrainerConfig,
) where
    A: Agent<E, R> + SyncModel,
    E: Env,
    R: ReplayBufferBase<PushedItem = S::Output> + Send + 'static,
    S: StepProcessorBase<E>,
    A::Config: Send + 'static,
    E::Config: Send + 'static,
    S::Config: Send + 'static,
    R::PushedItem: Send + 'static,
    A::ModelInfo: Send + 'static,
    P: AsRef<Path>,
{
    let mut recorder = TensorboardRecorder::new(model_dir);

    // Shared flag to stop actor threads
    let stop = Arc::new(Mutex::new(false));

    // Creates channels
    // let (item_s, item_r) = unbounded(); // items pushed to replay buffer
    let (item_s, item_r) = bounded(async_trainer_config.channel_capacity); // items pushed to replay buffer
    let (model_s, model_r) = unbounded(); // model_info

    // guard for initialization of envs in multiple threads
    let guard_init_env = Arc::new(Mutex::new(true));

    // Actor manager and async trainer
    let mut actors = ActorManager::<A, E, R, S>::build(
        actor_man_config,
        agent_configs,
        env_config_train,
        step_proc_config,
        item_s,
        model_r,
        stop.clone(),
    );
    let mut trainer = AsyncTrainer::<A, E, R>::build(
        async_trainer_config,
        agent_config,
        env_config_eval,
        replay_buffer_config,
        item_r,
        model_s,
        stop.clone(),
    );

    // Starts sampling and training
    actors.run(guard_init_env.clone());
    let stats = trainer.train(&mut recorder, guard_init_env);
    info!("Stats of async trainer");
    info!("{}", stats.fmt());

    let stats = actors.stop_and_join();
    info!("Stats of generated samples in actors");
    info!("{}", actor_stats_fmt(&stats));
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EarlyStoppingMonitorConfig {
    /// 改善が見られない状態が何回続いたら停止するか
    pub patience: usize,
    /// 移動中央値を計算する際のウィンドウサイズ
    pub window_size: usize,
    /// Early Stoppingを開始する前の最小ステップ数
    pub min_steps: usize,
    /// 報酬による早期終了用の閾値
    pub reward_threshold: f32,
}

/// 移動中央値を使用したEarly Stoppingモニター
/// 損失値の監視に特化した実装
pub struct EarlyStoppingMonitor {
    /// 改善が見られない状態が何回続いたら停止するか
    patience: usize,
    /// 移動中央値を計算する際のウィンドウサイズ
    window_size: usize,
    /// Early Stoppingを開始する前の最小ステップ数
    min_steps: usize,
    /// 値を保持するスライディングウィンドウ
    values: VecDeque<f32>,
    /// これまでの最良値（最小の損失値）
    best_value: Option<f32>,
    /// 改善が見られていない連続ステップ数
    counter: usize,
    /// 合計ステップ数
    steps_counter: usize,
}

impl EarlyStoppingMonitor {
    /// 新しいEarlyStoppingMonitorを作成
    ///
    /// # 引数
    ///
    /// * `patience` - 改善が見られない状態が何回続いたら停止するか
    /// * `window_size` - 移動中央値を計算する際のウィンドウサイズ
    /// * `min_steps` - Early Stoppingを開始する前の最小ステップ数
    pub fn new(early_stopping_config: EarlyStoppingMonitorConfig) -> Self {
        Self {
            patience: early_stopping_config.patience,
            window_size: early_stopping_config.window_size,
            min_steps: early_stopping_config.min_steps,
            values: VecDeque::with_capacity(early_stopping_config.window_size),
            best_value: None,
            counter: 0,
            steps_counter: 0,
        }
    }

    /// 値の配列から中央値を計算
    fn calculate_median(values: &[f32]) -> f32 {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted_values.len() / 2;
        if sorted_values.len() % 2 == 0 {
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[mid]
        }
    }

    /// 新しい損失値を追加し、学習を停止すべきかどうかを判断
    ///
    /// # 引数
    ///
    /// * `value` - 監視する新しい損失値
    ///
    /// # 戻り値
    ///
    /// * `bool` - 学習を停止すべき場合はtrue
    pub fn add_value(&mut self, value: f32) -> bool {
        self.steps_counter += 1;

        // スライディングウィンドウに新しい値を追加
        self.values.push_back(value);
        if self.values.len() > self.window_size {
            self.values.pop_front();
        }

        // 最小ステップ数に達していない場合は継続
        if self.steps_counter < self.min_steps {
            return false;
        }

        // ウィンドウサイズ分のデータが集まっていない場合は継続
        if self.values.len() < self.window_size {
            return false;
        }

        // 現在の中央値を計算
        let current_median = Self::calculate_median(&self.values.make_contiguous());

        // 最良値が未設定の場合、現在の中央値を最良値として設定
        if self.best_value.is_none() {
            self.best_value = Some(current_median);
            return false;
        }

        // 現在の中央値が改善している（より小さい）かチェック
        if current_median < self.best_value.unwrap() {
            self.best_value = Some(current_median);
            self.counter = 0;
        } else {
            self.counter += 1;
        }

        // 改善が見られない状態が続いている場合、停止を推奨
        self.counter >= self.patience
    }

    /// これまでの最良値（最小の損失値）を取得
    pub fn best_value(&self) -> Option<f32> {
        self.best_value
    }

    /// モニターの状態をリセット
    pub fn reset(&mut self) {
        self.values.clear();
        self.best_value = None;
        self.counter = 0;
        self.steps_counter = 0;
    }
}
