use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

use crate::util::EarlyStoppingMonitorConfig;

/// Configuration of [AsyncTrainer](crate::AsyncTrainer)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsyncTrainerConfig {
    /// Where to save the trained model.
    pub model_dir: Option<String>,

    pub model_name: String,

    /// Whether to save the best model
    /// If true, the model is saved when the reward in eval updates the maximum.
    /// If false, the model is saved at regular intervals according to save_interval.
    pub save_best_model: bool,

    /// Interval of recording in training steps.
    pub record_interval: usize,

    /// Interval of evaluation in training steps.
    pub eval_interval: usize,

    /// The maximal number of training steps.
    pub max_train_steps: usize,

    /// Interval of saving the model in optimization steps.
    pub save_interval: usize,

    /// Interval of synchronizing model parameters in training steps.
    pub sync_interval: usize,

    /// The number of episodes for evaluation
    pub eval_episodes: usize,

    /// capacity of channel between each actor-manager and async-trainer
    pub channel_capacity: usize,

    /// Configuration of early stopping.
    pub early_stopping_config: EarlyStoppingMonitorConfig,

    /// Timeout in minutes.
    pub timeout_minutes: Option<u64>,
}

impl AsyncTrainerConfig {
    /// Constructs [AsyncTrainerConfig] from YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let rdr = BufReader::new(file);
        let b = serde_yaml::from_reader(rdr)?;
        Ok(b)
    }

    /// Saves [AsyncTrainerConfig].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(serde_yaml::to_string(&self)?.as_bytes())?;
        Ok(())
    }
}
