use anyhow::Result;
use border_core::{record::BufferedRecorder, shape, util, Env as _, Policy};
use border_derive::{Act, Obs, SubBatch};
use border_py_gym_env::{
    FrameStackFilter, PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct,
    PyGymEnvDiscreteActRawFilter, PyGymEnvObs,
};
use border_tch_agent::TensorSubBatch;
// use clap::{App, Arg};
use std::convert::TryFrom;
// use tch::Tensor;

const N_STACK: i64 = 4;

type PyObsDtype = u8;

shape!(ObsShape, [N_STACK as usize, 1, 84, 84]);
shape!(ActShape, [1]);

#[derive(Clone, Debug, Obs)]
struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, f64>);

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, u8>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, f64>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

type ObsFilter = FrameStackFilter<ObsShape, PyObsDtype, f64, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;

#[derive(Clone)]
struct RandomPolicyConfig {
    pub n_acts: usize,
}

struct RandomPolicy {
    n_acts: usize,
}

impl Policy<Env> for RandomPolicy {
    type Config = RandomPolicyConfig;

    fn build(config: Self::Config) -> Self {
        Self {
            n_acts: config.n_acts,
        }
    }

    fn sample(&mut self, _: &Obs) -> Act {
        let v = fastrand::u32(..self.n_acts as u32);
        Act(PyGymEnvDiscreteAct::new(vec![v as i32]))
    }
}

fn env_config(name: &str) -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name(name.to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
        .atari_wrapper(Some(border_py_gym_env::AtariWrapper::Eval))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = env_config("PongNoFrameskip-v4");
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();
    let n_acts = env.get_num_actions_atari();
    env.set_render(true);
    let policy_config = RandomPolicyConfig {
        n_acts: n_acts as _,
    };
    let mut policy = RandomPolicy::build(policy_config);

    let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

    Ok(())
}
