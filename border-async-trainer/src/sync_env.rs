/// Synchronizes env
pub trait SyncEnv {
    /// Information of env.
    type EnvInfo: Clone;

    /// Get `EnvlInfo`.
    fn env_info(&self) -> Option<Self::EnvInfo> { None }

    /// Synchronizes env.
    #[allow(unused_variables)]
    fn sync_env(&mut self, env_info: &Self::EnvInfo) {}
}
