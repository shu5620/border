use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ExitReason {
    GoalAchieved,
    Converged,
    MaxStepsReached,
    Timeout,
}

/// Stats of [AsyncTrainer](crate::AsyncTrainer)`::train()`.
pub struct AsyncTrainStat {
    /// The number of samples pushed to the replay buffer per second.
    pub samples_per_sec: f32,

    /// Duration of training.
    pub duration: Duration,

    /// The number of optimization steps per second.
    pub opt_per_sec: f32,

    pub exit_reason: ExitReason,
}

impl AsyncTrainStat {
    /// Returns a formatted string.
    pub fn fmt(&self) -> String {
        let mut s = "samples/sec, opt_steps/sec, duration, exit_reason\n".to_string();
        s += format!(
            "{}, {}, {}, {:?}\n",
            self.samples_per_sec,
            self.opt_per_sec,
            self.duration.as_secs_f32(),
            self.exit_reason
        )
        .as_str();
        s
    }
}
