use anyhow::Result;
use std::collections::HashMap;

/// Snapshot of rich evaluation metrics used for advanced early-stopping decisions.
#[derive(Debug, Clone)]
pub struct RichEvalSnapshot {
    /// Aggregated reward computed from evaluation.
    pub reward: f32,
    /// Pairs of evaluation index id and the corresponding value.
    pub metrics: Vec<(u32, f32)>,
}

impl RichEvalSnapshot {
    /// Returns the metrics as a map for convenient lookup.
    pub fn metrics_map(&self) -> HashMap<u32, f32> {
        self.metrics.iter().copied().collect()
    }
}

/// Evaluates a rich set of metrics for early stopping decisions.
pub trait RichEvalEvaluator<A, E>: Send {
    /// Produces a snapshot of the latest evaluation.
    ///
    /// Implementations may leverage the agent, environment, or both to compute the
    /// custom reward signal that should be used for early stopping.
    fn evaluate(&self, agent: &mut A, env: &mut E) -> Result<RichEvalSnapshot>;
}
