//! Simple generic replay buffer.
mod iw_scheduler;
mod sum_tree;
use super::{config::PerConfig, StdBatch, SimpleReplayBufferConfig, SubBatch};
use crate::{StdBatchBase, ExperienceBufferBase, ReplayBufferBase};
use anyhow::Result;
pub use iw_scheduler::IwScheduler;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sum_tree::SumTree;
pub use sum_tree::WeightNormalizer;

struct PerState {
    sum_tree: SumTree,
    iw_scheduler: IwScheduler,
}

impl PerState {
    fn new(capacity: usize, per_config: &PerConfig) -> Self {
        Self {
            sum_tree: SumTree::new(capacity, per_config.alpha, per_config.normalize),
            iw_scheduler: IwScheduler::new(
                per_config.beta_0,
                per_config.beta_final,
                per_config.n_opts_final,
            ),
        }
    }
}

/// A simple generic replay buffer.
pub struct SimpleReplayBuffer<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    capacity: usize,
    i: usize,
    size: usize,
    obs: O,
    act: A,
    next_obs: O,
    reward: R,
    is_done: Vec<i8>,
    rng: StdRng,
    per_state: Option<PerState>,
    latest_indexes: Vec<usize>
}

impl<O, A, R> SimpleReplayBuffer<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    #[inline]
    fn push_is_done(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_done[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    fn sample_is_done(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_done[*ix]).collect()
    }

    /// Sets priorities for the added samples.
    fn set_priority(&mut self, batch_size: usize) {
        let sum_tree = &mut self.per_state.as_mut().unwrap().sum_tree;
        let max_p = sum_tree.max();

        for j in 0..batch_size {
            let i = (self.i + j) % self.capacity;
            sum_tree.add(i, max_p);
        }
    }
}

impl<O, A, R> ExperienceBufferBase for SimpleReplayBuffer<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    type PushedItem = StdBatch<O, A, R>;

    fn len(&self) -> usize {
        self.size
    }

    fn push(&mut self, tr: Self::PushedItem) -> Result<()> {
        let len = tr.len(); // batch size
        let (obs, act, next_obs, reward, is_done, _, _) = tr.unpack();
        self.obs.push(self.i, obs);
        self.act.push(self.i, act);
        self.next_obs.push(self.i, next_obs);
        self.reward.push(self.i, reward);
        self.push_is_done(self.i, &is_done);

        if self.per_state.is_some() {
            self.set_priority(len)
        };

        for i in 0..len {
            let ix = (self.i + i) % self.capacity;
            self.latest_indexes.push(ix);
        }

        self.i = (self.i + len) % self.capacity;
        self.size += len;
        if self.size >= self.capacity {
            self.size = self.capacity;
        }

        Ok(())
    }
}


impl<O, A, R> ReplayBufferBase for SimpleReplayBuffer<O, A, R>
where
    O: SubBatch,
    A: SubBatch,
    R: SubBatch,
{
    type Config = SimpleReplayBufferConfig;
    type Batch = StdBatch<O, A, R>;

    fn build(config: &Self::Config) -> Self {
        let capacity = config.capacity;
        let per_state = match &config.per_config {
            Some(per_config) => Some(PerState::new(capacity, per_config)),
            None => None,
        };

        Self {
            capacity,
            i: 0,
            size: 0,
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: R::new(capacity),
            is_done: vec![0; capacity],
            // rng: Rng::with_seed(config.seed),
            rng: StdRng::seed_from_u64(config.seed as _),
            per_state,
            latest_indexes: vec![],
        }
    }

    fn batch(&mut self, size: usize) -> anyhow::Result<Self::Batch> {
        let (ixs, weight) = if let Some(per_state) = &mut self.per_state {
            let sum_tree = &mut per_state.sum_tree;
            let beta = per_state.iw_scheduler.beta();
            let (ixs, weight) = sum_tree.sample(size, beta);
            let ixs = ixs.iter().map(|&ix| ix as usize).collect();
            (ixs, Some(weight))
        } else {
            let ixs = (0..size)
                // .map(|_| self.rng.usize(..self.size))
                .map(|_| (self.rng.next_u32() as usize) % self.size)
                .collect::<Vec<_>>();
            let weight = None;
            (ixs, weight)
        };

        Ok(Self::Batch {
            obs: self.obs.sample(&ixs),
            act: self.act.sample(&ixs),
            next_obs: self.next_obs.sample(&ixs),
            reward: self.reward.sample(&ixs),
            is_done: self.sample_is_done(&ixs),
            ix_sample: Some(ixs),
            weight,
        })
    }

    fn batch_latest(&mut self) -> anyhow::Result<Self::Batch> {

        let ixs = self.latest_indexes.clone();
        let batch = Ok(Self::Batch {
            obs: self.obs.sample(&ixs),
            act: self.act.sample(&ixs),
            next_obs: self.next_obs.sample(&ixs),
            reward: self.reward.sample(&ixs),
            is_done: self.sample_is_done(&ixs),
            ix_sample: Some(ixs),
            weight: None,
        });
        self.latest_indexes = Vec::new();
        batch
    }

    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_errs: &Option<Vec<f32>>) {
        if let Some(per_state) = &mut self.per_state {
            let ixs = ixs
                .as_ref()
                .expect("ixs should be Some(_) in update_priority().");
            let td_errs = td_errs
                .as_ref()
                .expect("td_errs should be Some(_) in update_priority().");
            for (&ix, &td_err) in ixs.iter().zip(td_errs.iter()) {
                per_state.sum_tree.update(ix, td_err);
            }
            per_state.iw_scheduler.add_n_opts();
        }
    }
}
