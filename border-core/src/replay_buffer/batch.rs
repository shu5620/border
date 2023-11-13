//! A generic implementation of [`StdBatchBase`](crate::StdBatchBase).
use super::SubBatch;
use crate::StdBatchBase;

/// A generic implementation of [`StdBatchBase`](`crate::StdBatchBase`).
pub struct StdBatch<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    /// Observations.
    pub obs: O,

    /// Actions.
    pub act: A,

    /// Next observations.
    pub next_obs: O,

    /// Rewards.
    pub reward: R,

    /// Done flags.
    pub is_done: Vec<i8>,

    /// Priority weights.
    pub weight: Option<Vec<f32>>,

    /// Sample indices.
    pub ix_sample: Option<Vec<usize>>,
}

impl<O, A, R> StdBatchBase for StdBatch<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    type ObsBatch = O;
    type ActBatch = A;
    type RewardBatch = R;

    fn unpack(
        self,
    ) -> (
        Self::ObsBatch,
        Self::ActBatch,
        Self::ObsBatch,
        Self::RewardBatch,
        Vec<i8>,
        Option<Vec<usize>>,
        Option<Vec<f32>>,
    ) {
        (
            self.obs,
            self.act,
            self.next_obs,
            self.reward,
            self.is_done,
            self.ix_sample,
            self.weight,
        )
    }

    fn len(&self) -> usize {
        self.is_done.len()
    }

    fn obs(&self) -> &Self::ObsBatch {
        &self.obs
    }

    fn act(&self) -> &Self::ActBatch {
        &self.act
    }

    fn next_obs(&self) -> &Self::ObsBatch {
        &self.next_obs
    }

    fn reward(&self) -> &Self::RewardBatch {
        &self.reward
    }

    fn is_done(&self) -> &Vec<i8> {
        &self.is_done
    }

    fn weight(&self) -> &Option<Vec<f32>> {
        &self.weight
    }

    fn ix_sample(&self) -> &Option<Vec<usize>> {
        &self.ix_sample
    }

    fn empty() -> Self {
        Self {
            obs: O::new(0),
            act: A::new(0),
            next_obs: O::new(0),
            reward: R::new(0),
            is_done: vec![],
            ix_sample: None,
            weight: None,
        }
    }
}

impl<O, A, R> StdBatch<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    /// Creates new batch with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: R::new(capacity),
            is_done: vec![0; capacity],
            ix_sample: None,
            weight: None,
        }
    }
}