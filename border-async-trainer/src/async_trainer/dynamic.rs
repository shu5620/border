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

    /// Creates a new empty snapshot.
    pub fn empty() -> Self {
        Self {
            reward: 0.0,
            metrics: Vec::new(),
        }
    }

    /// Computes the difference between this snapshot and a previous one.
    /// Returns a new snapshot with reward and metrics representing the delta.
    pub fn delta_from(&self, prev: &RichEvalSnapshot) -> Self {
        let reward_delta = self.reward - prev.reward;
        let prev_map = prev.metrics_map();

        let metrics_delta: Vec<(u32, f32)> = self
            .metrics
            .iter()
            .map(|(id, value)| {
                let prev_value = prev_map.get(id).copied().unwrap_or(0.0);
                (*id, value - prev_value)
            })
            .collect();

        Self {
            reward: reward_delta,
            metrics: metrics_delta,
        }
    }

    /// Accumulates another snapshot's values into this one.
    pub fn accumulate(&mut self, other: &RichEvalSnapshot) {
        self.reward += other.reward;

        let mut self_map: HashMap<u32, f32> = self.metrics_map();

        for (id, value) in &other.metrics {
            self_map
                .entry(*id)
                .and_modify(|v| *v += value)
                .or_insert(*value);
        }

        self.metrics = self_map.into_iter().collect();
        // Sort by metric id for consistent ordering
        self.metrics.sort_by_key(|(id, _)| *id);
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
