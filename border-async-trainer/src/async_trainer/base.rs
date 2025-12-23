use crate::{
    async_trainer::dynamic::{RichEvalEvaluator, RichEvalSnapshot},
    util::EarlyStoppingMonitor,
    util::EarlyStoppingMonitorConfig,
    AsyncTrainStat, AsyncTrainerConfig, ExitReason, PushedItemMessage, SyncModel,
};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue::Scalar, Recorder},
    Agent, Env, Obs, ReplayBufferBase,
};
use crossbeam_channel::{Receiver, Sender};
use log::{info, warn};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::SystemTime,
};

#[cfg_attr(doc, aquamarine::aquamarine)]
/// Manages asynchronous training loop in a single machine.
///
/// It interacts with [`ActorManager`] as shown below:
///
/// ```mermaid
/// flowchart LR
///   subgraph ActorManager
///     E[Actor]-->|ReplayBufferBase::PushedItem|H[ReplayBufferProxy]
///     F[Actor]-->H
///     G[Actor]-->H
///   end
///   K-->|SyncModel::ModelInfo|E
///   K-->|SyncModel::ModelInfo|F
///   K-->|SyncModel::ModelInfo|G
///
///   subgraph I[AsyncTrainer]
///     H-->|PushedItemMessage|J[ReplayBuffer]
///     J-->|ReplayBufferBase::Batch|K[Agent]
///   end
/// ```
///
/// * In [`ActorManager`] (right), [`Actor`]s sample transitions, which have type
///   [`ReplayBufferBase::PushedItem`], in parallel and push the transitions into
///   [`ReplayBufferProxy`]. It should be noted that [`ReplayBufferProxy`] has a
///   type parameter of [`ReplayBufferBase`] and the proxy accepts
///   [`ReplayBufferBase::PushedItem`].
/// * The proxy sends the transitions into the replay buffer, implementing
///   [`ReplayBufferBase`], in the [`AsyncTrainer`].
/// * The [`Agent`] in [`AsyncTrainer`] trains its model parameters by using batches
///   of type [`ReplayBufferBase::Batch`], which are taken from the replay buffer.
/// * The model parameters of the [`Agent`] in [`AsyncTrainer`] are wrapped in
///   [`SyncModel::ModelInfo`] and periodically sent to the [`Agent`]s in [`Actor`]s.
///   [`Agent`] must implement [`SyncModel`] to synchronize its model.
///
/// [`ActorManager`]: crate::ActorManager
/// [`Actor`]: crate::Actor
/// [`ReplayBufferBase::PushedItem`]: border_core::ReplayBufferBase::PushedItem
/// [`ReplayBufferProxy`]: crate::ReplayBufferProxy
/// [`ReplayBufferBase`]: border_core::ReplayBufferBase
/// [`SyncModel::ModelInfo`]: crate::SyncModel::ModelInfo
pub struct AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    /// Where to save the trained model.
    model_dir: Option<String>,

    model_name: String,

    /// Whether to save the best model
    /// If true, the model is saved when the reward in eval updates the maximum.
    /// If false, the model is saved at regular intervals according to save_interval.
    save_best_model: bool,

    /// Interval of recording in training steps.
    record_interval: usize,

    /// Interval of evaluation in training steps.
    eval_interval: usize,

    /// The maximal number of training steps.
    max_train_steps: usize,

    /// Interval of saving the model in optimization steps.
    save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    sync_interval: usize,

    /// The number of episodes for evaluation
    eval_episodes: usize,

    /// Configuration of early stopping.
    early_stopping_config: EarlyStoppingMonitorConfig,

    /// Timeout in minutes.
    timeout_minutes: Option<u64>,

    /// Receiver of pushed items.
    r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,

    /// If `false`, stops the actor threads.
    stop: Arc<Mutex<bool>>,

    /// Configuration of [Agent].
    agent_config: A::Config,

    /// Configuration of [Env]. Note that it is used only for evaluation, not for training.
    env_config: E::Config,

    /// Sender of model info.
    model_info_sender: Sender<(usize, A::ModelInfo)>,

    /// Configuration of replay buffer.
    replay_buffer_config: R::Config,

    /// Optional evaluator for rich-metric-based early stopping.
    rich_eval_evaluator: Option<Box<dyn RichEvalEvaluator<A, E>>>,

    /// Baseline reward captured at the first rich evaluation.
    rich_eval_baseline: Option<f32>,

    phantom: PhantomData<(A, E, R)>,
}

impl<A, E, R> AsyncTrainer<A, E, R>
where
    A: Agent<E, R> + SyncModel,
    E: Env,
    // R: ReplayBufferBase + Sync + Send + 'static,
    R: ReplayBufferBase,
    R::PushedItem: Send + 'static,
{
    /// Creates [`AsyncTrainer`].
    pub fn build(
        config: &AsyncTrainerConfig,
        agent_config: &A::Config,
        env_config: &E::Config,
        replay_buffer_config: &R::Config,
        r_bulk_pushed_item: Receiver<PushedItemMessage<R::PushedItem>>,
        model_info_sender: Sender<(usize, A::ModelInfo)>,
        stop: Arc<Mutex<bool>>,
        rich_eval_evaluator: Option<Box<dyn RichEvalEvaluator<A, E>>>,
    ) -> Self {
        Self {
            model_dir: config.model_dir.clone(),
            model_name: config.model_name.clone(),
            save_best_model: config.save_best_model,
            record_interval: config.record_interval,
            eval_interval: config.eval_interval,
            max_train_steps: config.max_train_steps,
            save_interval: config.save_interval,
            sync_interval: config.sync_interval,
            eval_episodes: config.eval_episodes,
            early_stopping_config: config.early_stopping_config.clone(),
            timeout_minutes: config.timeout_minutes,
            agent_config: agent_config.clone(),
            env_config: env_config.clone(),
            replay_buffer_config: replay_buffer_config.clone(),
            r_bulk_pushed_item,
            model_info_sender,
            stop,
            rich_eval_evaluator,
            rich_eval_baseline: None,
            phantom: PhantomData,
        }
    }

    fn save_model(&self, agent: &A) {
        let path = std::path::Path::new(&self.model_dir.as_ref().unwrap()).join(&self.model_name);
        match agent.save(&path) {
            Ok(()) => info!("Saved the model in {:?}", &path),
            Err(_) => info!("Failed to save model."),
        }
    }

    fn run_single_eval_episode(
        &mut self,
        agent: &mut A,
        env: &mut E,
        record: &mut Record,
        ix: usize,
    ) -> Result<f32> {
        let mut prev_obs = env.reset_with_index(ix)?;
        assert_eq!(prev_obs.len(), 1); // env must be non-vectorized

        let mut episode_reward = 0f32;

        loop {
            let act = agent.sample(&prev_obs);
            let (step, record_) = env.step(&act);
            record.extend(record_);
            episode_reward += step.reward[0];
            if step.is_done[0] == 1 {
                break;
            }
            prev_obs = step.obs;
        }

        Ok(episode_reward)
    }

    fn run_eval_episodes(
        &mut self,
        agent: &mut A,
        env: &mut E,
        record: &mut Record,
        evaluator: Option<&dyn RichEvalEvaluator<A, E>>,
    ) -> Result<(Vec<f32>, Vec<RichEvalSnapshot>)> {
        agent.eval();
        env.set_eval_mode();

        let mut episode_rewards = Vec::with_capacity(self.eval_episodes);
        let mut snapshots = Vec::new();

        for ix in 0..self.eval_episodes {
            let episode_reward = self.run_single_eval_episode(agent, env, record, ix)?;
            episode_rewards.push(episode_reward);
            record.insert(format!("eval_reward_episode_{ix}"), Scalar(episode_reward));

            if let Some(evaluator) = evaluator {
                match evaluator.evaluate(agent, env) {
                    Ok(snapshot) => snapshots.push(snapshot),
                    Err(err) => warn!("Failed to evaluate rich metrics on episode {ix}: {}", err),
                }
            }
        }

        agent.train();
        env.set_train_mode();

        Ok((episode_rewards, snapshots))
    }

    fn evaluate(&mut self, agent: &mut A, env: &mut E, record: &mut Record) -> Result<f32> {
        self.run_eval_episodes(agent, env, record, None)
            .map(|(rewards, _)| *rewards.last().unwrap_or(&0.0))
    }

    /// Do evaluation.
    #[inline(always)]
    fn eval(
        &mut self,
        agent: &mut A,
        env: &mut E,
        record: &mut Record,
        max_eval_reward: &mut f32,
    ) -> f32 {
        let eval_reward = self.evaluate(agent, env, record).unwrap();
        self.handle_eval_reward(eval_reward, agent, record, max_eval_reward);
        eval_reward
    }

    /// Inserts evaluation reward into the record and updates the best model if necessary.
    #[inline]
    fn handle_eval_reward(
        &mut self,
        reward: f32,
        agent: &mut A,
        record: &mut Record,
        max_eval_reward: &mut f32,
    ) {
        record.insert("eval_reward", Scalar(reward));

        // Save the best model up to the current iteration
        if self.save_best_model && reward > *max_eval_reward {
            *max_eval_reward = reward;
            self.save_model(agent);
            info!("Saved the best model");
        }
    }

    #[inline]
    fn eval_with_rich_metrics(
        &mut self,
        evaluator: &dyn RichEvalEvaluator<A, E>,
        agent: &mut A,
        env: &mut E,
        record: &mut Record,
        max_eval_reward: &mut f32,
    ) -> Option<(f32, bool)> {
        let (episode_rewards, snapshots) =
            match self.run_eval_episodes(agent, env, record, Some(evaluator)) {
                Ok(result) => result,
                Err(err) => {
                    warn!("Failed to evaluate episodes for rich metrics: {}", err);
                    return Some((0.0, false));
                }
            };

        // Use the last episode's reward for early-stopping decisions
        let reward = *episode_rewards.last().unwrap_or(&0.0);
        self.handle_eval_reward(reward, agent, record, max_eval_reward);

        if snapshots.is_empty() {
            warn!("Rich evaluation snapshots were empty; skipping metric aggregation");
            return Some((reward, false));
        }

        // Use only the last episode's snapshot for recording/early-stopping
        let last_snapshot = snapshots.last().unwrap();

        for (ix, snapshot) in snapshots.iter().enumerate() {
            for (metric_id, value) in snapshot.metrics.iter().copied() {
                record.insert(
                    format!("rich_eval_metric_{metric_id}_episode_{ix}"),
                    Scalar(value),
                );
            }
        }

        for (metric_id, value) in last_snapshot.metrics.iter().copied() {
            record.insert(format!("rich_eval_metric_{metric_id}"), Scalar(value));
        }

        let snapshot_reward_total = last_snapshot.reward;

        if reward.is_finite() {
            if self.rich_eval_baseline.is_none() {
                self.rich_eval_baseline = Some(reward);
                record.insert("eval_reward_baseline", Scalar(reward));
                info!("Captured baseline rich evaluation reward: {:.4}", reward);
            } else if let Some(baseline) = self.rich_eval_baseline {
                record.insert("eval_reward_baseline", Scalar(baseline));
                let reward_ok = reward >= baseline;
                let metrics_map: HashMap<u32, f32> =
                    last_snapshot.metrics.iter().copied().collect();
                let target_metrics = &self.early_stopping_config.target_evaluation_index;
                let metrics_ok = !target_metrics.is_empty()
                    && target_metrics.iter().all(|(id, threshold)| {
                        metrics_map
                            .get(id)
                            .map(|value| *value >= *threshold)
                            .unwrap_or(false)
                    });

                if reward_ok && metrics_ok {
                    info!("Early stopping condition satisfied by rich evaluation");
                    return Some((reward, true));
                }
            }
        } else {
            warn!("Rich evaluation returned non-finite reward; skipping early stop");
        }

        if snapshot_reward_total.is_finite()
            && (snapshot_reward_total - reward).abs() > f32::EPSILON
        {
            warn!(
                "Rich evaluator reported total reward {:.4} differing from recorded eval reward {:.4}",
                snapshot_reward_total, reward
            );
        }

        Some((reward, false))
    }

    /// Record.
    #[inline]
    fn record(
        &mut self,
        record: &mut Record,
        opt_steps_: &mut usize,
        samples: &mut usize,
        time: &mut SystemTime,
        samples_total: usize,
    ) {
        let duration = time.elapsed().unwrap().as_secs_f32();
        let ops = (*opt_steps_ as f32) / duration;
        let sps = (*samples as f32) / duration;
        let spo = (*samples as f32) / (*opt_steps_ as f32);
        record.insert("samples_total", Scalar(samples_total as _));
        record.insert("opt_steps_per_sec", Scalar(ops));
        record.insert("samples_per_sec", Scalar(sps));
        record.insert("samples_per_opt_steps", Scalar(spo));
        // info!("Collected samples per optimization step = {}", spo);

        // Reset counter
        *opt_steps_ = 0;
        *samples = 0;
        *time = SystemTime::now();
    }

    /// Flush record.
    #[inline]
    fn flush(&mut self, opt_steps: usize, mut record: Record, recorder: &mut impl Recorder) {
        record.insert("opt_steps", Scalar(opt_steps as _));
        recorder.write(record);
    }

    /// Save model.
    #[inline]
    fn save(&self, agent: &A) {
        if self.save_best_model {
            return;
        }

        self.save_model(agent);
    }

    /// Sync model.
    #[inline]
    fn sync(&mut self, agent: &A) {
        let model_info = agent.model_info();
        // TODO: error handling
        self.model_info_sender.send(model_info).unwrap();
    }

    // /// Run a thread for replay buffer.
    // fn run_replay_buffer_thread(&self, buffer: Arc<Mutex<R>>) {
    //     let r = self.r_bulk_pushed_item.clone();
    //     let stop = self.stop.clone();

    //     std::thread::spawn(move || loop {
    //         let msg = r.recv().unwrap();
    //         {
    //             let mut buffer = buffer.lock().unwrap();
    //             buffer.push(msg.pushed_item);
    //         }
    //         if *stop.lock().unwrap() {
    //             break;
    //         }
    //         std::thread::sleep(std::time::Duration::from_millis(100));
    //     });
    // }

    /// Runs training loop.
    ///
    /// In the training loop, the following values will be pushed into the given recorder:
    ///
    /// * `samples_total` - Total number of samples pushed into the replay buffer.
    ///   Here, a "sample" is an item in [`ExperienceBufferBase::PushedItem`].
    /// * `opt_steps_per_sec` - The number of optimization steps per second.
    /// * `samples_per_sec` - The number of samples per second.
    /// * `samples_per_opt_steps` - The number of samples per optimization step.
    ///
    /// These values will typically be monitored with tensorboard.
    ///
    /// [`ExperienceBufferBase::PushedItem`]: border_core::ExperienceBufferBase::PushedItem
    pub fn train(
        &mut self,
        recorder: &mut impl Recorder,
        guard_init_env: Arc<Mutex<bool>>,
    ) -> AsyncTrainStat {
        let timeout_duration = if let Some(timeout_minutes) = self.timeout_minutes {
            Some(std::time::Duration::from_secs(timeout_minutes * 60))
        } else {
            None
        };

        let start_time_for_timeout = std::time::Instant::now();

        // TODO: error handling
        let mut env = {
            let mut tmp = guard_init_env.lock().unwrap();
            *tmp = true;
            let mut env = E::build(&self.env_config, 0).unwrap();
            env.set_train_mode();
            env
        };
        let mut agent = A::build(self.agent_config.clone());
        let mut buffer = R::build(&self.replay_buffer_config);
        // let buffer = Arc::new(Mutex::new(R::build(&self.replay_buffer_config)));
        agent.train();

        // self.run_replay_buffer_thread(buffer.clone());

        // Early Stoppingモニターの初期化
        let mut early_stopping = EarlyStoppingMonitor::new(self.early_stopping_config.clone());

        let mut max_eval_reward = f32::MIN;
        let mut opt_steps = 0;
        let mut opt_steps_ = 0;
        let mut samples = 0;
        let time_total = SystemTime::now();
        let mut samples_total = 0;
        let mut time = SystemTime::now();

        info!("Send model info first in AsyncTrainer");
        self.sync(&mut agent);

        info!("Starts training loop");
        let exit_reason: ExitReason = loop {
            if let Some(timeout_duration) = &timeout_duration {
                if start_time_for_timeout.elapsed() >= *timeout_duration {
                    // モデルを保存して終了
                    self.save(&agent);
                    // チャンネルをフラッシュして終了
                    *self.stop.lock().unwrap() = true;
                    let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                    self.sync(&agent);
                    break ExitReason::Timeout;
                }
            }

            // Update replay buffer
            let msgs: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
            msgs.into_iter().for_each(|msg| {
                samples += msg.pushed_items.len();
                samples_total += msg.pushed_items.len();
                msg.pushed_items
                    .into_iter()
                    .for_each(|pushed_item| buffer.push(pushed_item).unwrap())
            });

            let (record, loss): (Option<Record>, Option<f32>) = agent.opt(&mut buffer);

            if let Some(mut record) = record {
                opt_steps += 1;
                opt_steps_ += 1;

                let do_eval = opt_steps % self.eval_interval == 0;
                let do_record = opt_steps % self.record_interval == 0;
                let do_flush = do_eval || do_record;
                let do_save = opt_steps % self.save_interval == 0;
                let do_sync = opt_steps % self.sync_interval == 0;

                if do_eval {
                    info!("Starts evaluation of the trained model");
                    let mut stop_reason: Option<&str> = None;
                    let eval_reward = if self.rich_eval_evaluator.is_some() {
                        let evaluator_opt = self.rich_eval_evaluator.take();
                        let eval_output = evaluator_opt.as_ref().and_then(|evaluator| {
                            self.eval_with_rich_metrics(
                                evaluator.as_ref(),
                                &mut agent,
                                &mut env,
                                &mut record,
                                &mut max_eval_reward,
                            )
                        });
                        self.rich_eval_evaluator = evaluator_opt;

                        match eval_output {
                            Some((reward, should_stop)) => {
                                if should_stop {
                                    stop_reason = Some("rich_eval");
                                }
                                reward
                            }
                            None => {
                                self.eval(&mut agent, &mut env, &mut record, &mut max_eval_reward)
                            }
                        }
                    } else {
                        self.eval(&mut agent, &mut env, &mut record, &mut max_eval_reward)
                    };

                    // リッチ評価で終了しなかった場合は、従来の報酬閾値による判定を実行
                    if stop_reason.is_none()
                        && (self.rich_eval_evaluator.is_none()
                            || self
                                .early_stopping_config
                                .target_evaluation_index
                                .is_empty())
                        && eval_reward >= self.early_stopping_config.reward_threshold
                    {
                        stop_reason = Some("reward_threshold");
                        info!("Early stopping condition satisfied by reward threshold");
                    }

                    if let Some(_reason) = stop_reason {
                        // ログを保存して終了
                        info!("Records training logs");
                        self.record(
                            &mut record,
                            &mut opt_steps_,
                            &mut samples,
                            &mut time,
                            samples_total,
                        );
                        // モデルを保存して終了
                        self.save(&agent);
                        // チャンネルをフラッシュして終了
                        *self.stop.lock().unwrap() = true;
                        let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                        self.sync(&agent);
                        break ExitReason::GoalAchieved;
                    }
                }
                if do_record {
                    // lossがNaNの場合は無視
                    if loss.is_none() {
                        continue;
                    }

                    // Early Stopping判定
                    if early_stopping.add_value(loss.unwrap()) {
                        info!(
                            "Early stopping triggered. Best loss: {}",
                            early_stopping.best_value().unwrap()
                        );

                        // ログを保存して終了
                        info!("Records training logs");
                        self.record(
                            &mut record,
                            &mut opt_steps_,
                            &mut samples,
                            &mut time,
                            samples_total,
                        );

                        // モデルを保存して終了
                        info!("Saves the trained model");
                        self.save(&agent);

                        // チャンネルをフラッシュして終了
                        *self.stop.lock().unwrap() = true;
                        let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                        self.sync(&agent);
                        break ExitReason::Converged;
                    } else {
                        info!("Records training logs");
                        self.record(
                            &mut record,
                            &mut opt_steps_,
                            &mut samples,
                            &mut time,
                            samples_total,
                        );
                    }
                }
                if do_flush {
                    info!("Flushes records");
                    self.flush(opt_steps, record, recorder);
                }
                if do_save {
                    info!("Saves the trained model");
                    self.save(&agent);
                }
                if opt_steps == self.max_train_steps {
                    // Flush channels
                    *self.stop.lock().unwrap() = true;
                    let _: Vec<_> = self.r_bulk_pushed_item.try_iter().collect();
                    self.sync(&agent);
                    break ExitReason::MaxStepsReached;
                }
                if do_sync {
                    info!("Sends the trained model info to ActorManager");
                    self.sync(&agent);
                }
            }
        };
        info!("Stopped training loop");

        let duration = time_total.elapsed().unwrap();
        let time_total = duration.as_secs_f32();
        let samples_per_sec = samples_total as f32 / time_total;
        let opt_per_sec = self.max_train_steps as f32 / time_total;
        AsyncTrainStat {
            samples_per_sec,
            duration,
            opt_per_sec,
            exit_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use border_core::{
        record::{Record, RecordValue::Scalar},
        Act, ExperienceBufferBase, Info, Obs, Policy, Step,
    };
    use crossbeam_channel::{bounded, unbounded};

    #[derive(Clone, Debug)]
    struct DummyObs(i32);

    impl Obs for DummyObs {
        fn dummy(_n: usize) -> Self {
            DummyObs(0)
        }

        fn merge(self, _obs_reset: Self, _is_done: &[i8]) -> Self {
            self
        }

        fn len(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    struct DummyAct;

    impl Act for DummyAct {
        fn len(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    struct DummyInfo;

    impl Info for DummyInfo {}

    #[derive(Clone)]
    struct DummyEnvConfig {
        episodes: Vec<Vec<(f32, f32)>>,
    }

    struct DummyEnv {
        config: DummyEnvConfig,
        episode_ix: usize,
        step_ix: usize,
        episode_reward: f32,
        episode_metric: f32,
    }

    impl DummyEnv {
        fn reset_state(&mut self, ix: usize) -> DummyObs {
            self.episode_ix = ix;
            self.step_ix = 0;
            self.episode_reward = 0.0;
            self.episode_metric = 0.0;
            DummyObs(0)
        }
    }

    impl Env for DummyEnv {
        type Config = DummyEnvConfig;
        type Obs = DummyObs;
        type Act = DummyAct;
        type Info = DummyInfo;

        fn build(config: &Self::Config, _seed: i64) -> Result<Self> {
            Ok(Self {
                config: config.clone(),
                episode_ix: 0,
                step_ix: 0,
                episode_reward: 0.0,
                episode_metric: 0.0,
            })
        }

        fn step(&mut self, _a: &Self::Act) -> (Step<Self>, Record) {
            let episode = &self.config.episodes[self.episode_ix];
            let (reward, metric) = episode[self.step_ix];
            self.step_ix += 1;
            self.episode_reward += reward;
            self.episode_metric += metric;

            let is_done = if self.step_ix >= episode.len() { 1 } else { 0 };
            let obs = DummyObs(self.step_ix as i32);

            (
                Step::new(
                    obs.clone(),
                    DummyAct,
                    vec![reward],
                    vec![is_done],
                    DummyInfo,
                    DummyObs(0),
                ),
                Record::empty(),
            )
        }

        fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
            if let Some(flags) = is_done {
                if flags.get(0).copied() == Some(1) {
                    return Ok(self.reset_state(self.episode_ix));
                }
            }
            Ok(self.reset_state(self.episode_ix))
        }

        fn step_with_reset(&mut self, a: &Self::Act) -> (Step<Self>, Record) {
            let (step, record) = self.step(a);
            if step.is_done[0] == 1 {
                let _ = self.reset(Some(&step.is_done));
            }
            (step, record)
        }

        fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs> {
            Ok(self.reset_state(ix))
        }

        fn set_train_mode(&mut self) {}

        fn set_eval_mode(&mut self) {}
    }

    #[derive(Clone)]
    struct DummyAgentConfig;

    #[derive(Clone)]
    struct DummyAgent {
        is_train: bool,
    }

    impl Policy<DummyEnv> for DummyAgent {
        type Config = DummyAgentConfig;

        fn build(_config: Self::Config) -> Self {
            Self { is_train: true }
        }

        fn sample(&mut self, _obs: &DummyObs) -> DummyAct {
            DummyAct
        }
    }

    impl Agent<DummyEnv, DummyReplayBuffer> for DummyAgent {
        fn train(&mut self) {
            self.is_train = true;
        }

        fn eval(&mut self) {
            self.is_train = false;
        }

        fn is_train(&self) -> bool {
            self.is_train
        }

        fn opt(&mut self, _buffer: &mut DummyReplayBuffer) -> (Option<Record>, Option<f32>) {
            (None, None)
        }

        fn save<T: AsRef<std::path::Path>>(&self, _path: T) -> Result<()> {
            Ok(())
        }

        fn load<T: AsRef<std::path::Path>>(&mut self, _path: T) -> Result<()> {
            Ok(())
        }
    }

    impl SyncModel for DummyAgent {
        type ModelInfo = ();

        fn model_info(&self) -> (usize, Self::ModelInfo) {
            (0, ())
        }

        fn sync_model(&mut self, _model_info: &Self::ModelInfo) {}
    }

    #[derive(Clone)]
    struct DummyReplayBufferConfig;

    struct DummyReplayBuffer;

    impl ExperienceBufferBase for DummyReplayBuffer {
        type PushedItem = ();

        fn push(&mut self, _tr: Self::PushedItem) -> Result<()> {
            Ok(())
        }

        fn len(&self) -> usize {
            0
        }
    }

    impl ReplayBufferBase for DummyReplayBuffer {
        type Config = DummyReplayBufferConfig;
        type Batch = ();

        fn build(_config: &Self::Config) -> Self {
            DummyReplayBuffer
        }

        fn batch(&mut self, _size: usize) -> Result<Self::Batch> {
            Ok(())
        }

        fn update_priority(&mut self, _ixs: &Option<Vec<usize>>, _td_err: &Option<Vec<f32>>) {}
    }

    struct DummyEvaluator;

    impl RichEvalEvaluator<DummyAgent, DummyEnv> for DummyEvaluator {
        fn evaluate(
            &self,
            _agent: &mut DummyAgent,
            env: &mut DummyEnv,
        ) -> Result<RichEvalSnapshot> {
            Ok(RichEvalSnapshot {
                reward: env.episode_reward,
                metrics: vec![(1, env.episode_metric)],
            })
        }
    }

    #[test]
    fn rich_eval_aggregates_episode_totals() {
        let trainer_config = AsyncTrainerConfig {
            model_dir: None,
            model_name: "dummy".into(),
            save_best_model: false,
            record_interval: 1,
            eval_interval: 1,
            max_train_steps: 1,
            save_interval: 1,
            sync_interval: 1,
            eval_episodes: 2,
            channel_capacity: 1,
            early_stopping_config: EarlyStoppingMonitorConfig {
                patience: 1,
                window_size: 1,
                min_steps: 0,
                reward_threshold: f32::MIN,
                target_evaluation_index: vec![(1, 0.0)],
            },
            timeout_minutes: None,
        };

        let env_config = DummyEnvConfig {
            episodes: vec![vec![(1.0, 10.0), (2.0, 20.0)], vec![(0.5, 7.0)]],
        };
        let agent_config = DummyAgentConfig;
        let replay_buffer_config = DummyReplayBufferConfig;

        let (_item_s, item_r) = bounded(trainer_config.channel_capacity);
        let (model_s, _model_r) = unbounded();
        let stop = Arc::new(Mutex::new(false));
        let evaluator: Box<dyn RichEvalEvaluator<DummyAgent, DummyEnv>> = Box::new(DummyEvaluator);

        let mut trainer = AsyncTrainer::<DummyAgent, DummyEnv, DummyReplayBuffer>::build(
            &trainer_config,
            &agent_config,
            &env_config,
            &replay_buffer_config,
            item_r,
            model_s,
            stop,
            Some(evaluator),
        );

        let mut agent = DummyAgent::build(agent_config.clone());
        let mut env = DummyEnv::build(&env_config, 0).unwrap();
        let mut record = Record::empty();
        let mut max_eval_reward = f32::MIN;

        let evaluator_ref = trainer.rich_eval_evaluator.take().unwrap();
        let result = trainer
            .eval_with_rich_metrics(
                evaluator_ref.as_ref(),
                &mut agent,
                &mut env,
                &mut record,
                &mut max_eval_reward,
            )
            .unwrap();
        trainer.rich_eval_evaluator = Some(evaluator_ref);

        // Reward uses the last episode only: second episode total is 0.5
        assert!((result.0 - 0.5).abs() < 1e-6);

        // Metric also uses the last episode only: second episode total is 7.0
        let metric_value = match record.get("rich_eval_metric_1").unwrap() {
            Scalar(v) => *v,
            _ => panic!("Unexpected record value"),
        };
        assert!((metric_value - 7.0).abs() < 1e-6);
    }
}
