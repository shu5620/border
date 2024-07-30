use border_core::shape;
use border_py_gym_env::{newtype_act_d, newtype_obs};

shape!(ObsShape, [1, 2, 3]);
newtype_obs!(Obs, ObsFilter, ObsShape, f64, f64);
newtype_act_d!(ActD, ActDFilter);
newtype_act_d!(ActC, ActCFilter);

fn main() {}
