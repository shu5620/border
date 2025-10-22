use anyhow::Result;
use std::collections::HashMap;

/// Snapshot of custom evaluation metrics used for dynamic early stopping.
#[derive(Debug, Clone)]
pub struct DynamicRewardSnapshot {
    /// Custom reward value computed from evaluation.
    pub reward: f32,
    /// Pairs of evaluation index id and the corresponding value.
    pub metrics: Vec<(u32, f32)>,
}

impl DynamicRewardSnapshot {
    /// Returns the metrics as a map for convenient lookup.
    pub fn metrics_map(&self) -> HashMap<u32, f32> {
        self.metrics.iter().copied().collect()
    }
}

/// Evaluates custom reward / metrics for dynamic early stopping decisions.
pub trait DynamicRewardEvaluator<A, E>: Send {
    /// Produces a snapshot of the latest evaluation.
    ///
    /// Implementations may leverage the agent, environment, or both to compute the
    /// custom reward signal that should be used for early stopping.
    fn evaluate(&mut self, agent: &mut A, env: &mut E) -> Result<DynamicRewardSnapshot>;
}
