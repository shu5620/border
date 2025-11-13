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

    fn evaluate(&mut self, agent: &mut A, env: &mut E, record: &mut Record) -> Result<f32> {
        agent.eval();
        env.set_eval_mode();

        let mut r_total = 0f32;

        for ix in 0..self.eval_episodes {
            let mut prev_obs = env.reset_with_index(ix)?;
            assert_eq!(prev_obs.len(), 1); // env must be non-vectorized

            loop {
                let act = agent.sample(&prev_obs);
                let (step, record_) = env.step(&act);
                record.extend(record_);
                r_total += step.reward[0];
                if step.is_done[0] == 1 {
                    break;
                }
                prev_obs = step.obs;
            }
        }

        agent.train();
        env.set_train_mode();

        Ok(r_total / self.eval_episodes as f32)
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
        let reward = self.evaluate(agent, env, record).unwrap();
        self.handle_eval_reward(reward, agent, record, max_eval_reward);

        match evaluator.evaluate(agent, env) {
            Ok(snapshot) => {
                let RichEvalSnapshot {
                    reward: snapshot_reward,
                    metrics,
                } = snapshot;

                for (metric_id, value) in metrics.iter() {
                    record.insert(format!("rich_eval_metric_{metric_id}"), Scalar(*value));
                }

                if reward.is_finite() {
                    if self.rich_eval_baseline.is_none() {
                        self.rich_eval_baseline = Some(reward);
                        record.insert("eval_reward_baseline", Scalar(reward));
                        info!("Captured baseline rich evaluation reward: {:.4}", reward);
                    } else if let Some(baseline) = self.rich_eval_baseline {
                        record.insert("eval_reward_baseline", Scalar(baseline));
                        let reward_ok = reward >= baseline;
                        let metrics_map: HashMap<u32, f32> = metrics.iter().copied().collect();
                        let metrics_ok = self
                            .early_stopping_config
                            .target_evaluation_index
                            .iter()
                            .all(|(id, threshold)| {
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

                if snapshot_reward.is_finite() && (snapshot_reward - reward).abs() > f32::EPSILON {
                    warn!(
                        "Rich evaluator reported reward {:.4} differing from recorded eval reward {:.4}",
                        snapshot_reward, reward
                    );
                }

                Some((reward, false))
            }
            Err(err) => {
                warn!("Failed to evaluate rich early stopping metrics: {}", err);
                Some((reward, false))
            }
        }
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
