//! Scheduling the exponent of importance weight for PER.
use serde::{Deserialize, Serialize};

/// Scheduler of the exponent of importance weight for PER.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct IwScheduler {
    /// Initial value of $\beta$.
    pub beta_0: f64,

    /// Final value of $\beta$.
    pub beta_final: f64,

    /// Optimization steps when beta reaches its final value.
    pub n_opts_final: usize,

    /// Current optimizatioin steps.
    pub n_opts: usize,
}

impl IwScheduler {
    /// Creates a scheduler.
    pub fn new(beta_0: f64, beta_final: f64, n_opts_final: usize) -> Self {
        Self { beta_0, beta_final, n_opts_final, n_opts: 0 }
    }

    /// Gets the exponents of importance sampling weight.
    pub fn beta(&self) -> f64 {
        let n_opts = self.n_opts;
        if n_opts >= self.n_opts_final {
            self.beta_final
        } else {
            let d = self.beta_final - self.beta_0;
            self.beta_0 + d * (n_opts as f64 / self.n_opts_final as f64)
        }
    }

    /// Add optimization steps for scheduling beta through training.
    pub fn add_n_opts(&mut self) {
        self.n_opts += 1;
    }
}
