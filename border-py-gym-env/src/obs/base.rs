use border_core::{Obs, Shape};
use log::trace;
use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::cast::AsPrimitive;
use numpy::{Element, PyArrayDyn};
use pyo3::PyObject;
use std::fmt::Debug;
use std::marker::PhantomData;
#[cfg(feature = "tch")]
use {tch::Tensor, std::convert::TryFrom};

fn any(is_done: &[i8]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

/// Convert PyObject to ArrayD.
///
/// If the shape of the PyArray has the number of axes equal to the shape of
/// observation, i.e., `S.shape().len()`, it is considered an observation from a
/// non-vectorized environment, an axis will be appended before the leading dimension.
/// in order for the array to meet the shape of the array in [`PyGymEnvObs`].
pub fn pyobj_to_arrayd<S, T1, T2>(obs: PyObject) -> ArrayD<T2>
where
    S: Shape,
    T1: Element + AsPrimitive<T2>,
    T2: 'static + Copy,
{
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<T1> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        // let obs = obs.mapv(|elem| elem as f64);
        let obs = obs.mapv(|elem| elem.as_());
        let obs = {
            if obs.shape().len() == S::shape().len() + 1 {
                // In this case obs has an axis for len
                obs
            } else if obs.shape().len() == S::shape().len() {
                // add axis for the number of samples in obs
                obs.insert_axis(Axis(0))
            } else {
                panic!();
            }
        };
        obs
    })
}

/// Observation represented by an [ndarray::ArrayD].
///
/// `S` is the shape of an observation, except for batch and process dimensions.
/// `T` is the dtype of ndarray in the Python gym environment.
/// For some reason, the dtype of observations in Python gym environments seems to
/// vary, f64 or f64. To get observations in Rust side, the dtype is specified as a
/// type parameter, instead of checking the dtype of Python array at runtime.
#[derive(Clone, Debug)]
pub struct PyGymEnvObs<S, T1, T2>
where
    S: Shape,
    T1: Element + Debug,
    T2: 'static + Copy,
{
    pub obs: ArrayD<T2>,
    pub(crate) phantom: PhantomData<(S, T1)>,
}

impl<S, T1, T2> From<ArrayD<T2>> for PyGymEnvObs<S, T1, T2>
where
    S: Shape,
    T1: Element + Debug,
    T2: 'static + Copy,
{
    fn from(obs: ArrayD<T2>) -> Self {
        Self {
            obs,
            phantom: PhantomData,
        }
    }
}

// impl<S, T1, T2> Obs for PyGymEnvObs<S, T1, T2> where
//     S: Shape,
//     T1: Element + Debug + num_traits::identities::Zero,
// {
impl<S, T1, T2> Obs for PyGymEnvObs<S, T1, T2>
where
    S: Shape,
    T1: Debug + Element,
    T2: 'static + Copy + Debug + num_traits::Zero,
{
    fn dummy(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData,
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[i8]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0 {
                    self.obs
                        .index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }

    fn len(&self) -> usize {
        self.obs.shape()[0]
    }
}

// #[cfg(feature = "tch")]
// impl<S, T1, T2> From<PyGymEnvObs<S, T1, T2>> for Tensor
// where
//     S: Shape,
//     T1: Element + Debug,
//     T2: 'static + Copy,
// {
//     fn from(obs: PyGymEnvObs<S, T1, T2>) -> Tensor {
//         let tmp = &obs.obs;
//         Tensor::try_from(tmp).unwrap()
//         // Tensor::try_from(&obs.obs).unwrap()
//     }
// }

#[cfg(feature = "tch")]
impl<S, T1> From<PyGymEnvObs<S, T1, f64>> for Tensor
where
    S: Shape,
    T1: Element + Debug,
{
    fn from(obs: PyGymEnvObs<S, T1, f64>) -> Tensor {
        let tmp = &obs.obs;
        Tensor::try_from(tmp).unwrap()
        // Tensor::try_from(&obs.obs).unwrap()
    }
}

#[cfg(feature = "tch")]
impl<S, T1> From<PyGymEnvObs<S, T1, u8>> for Tensor
where
    S: Shape,
    T1: Element + Debug,
{
    fn from(obs: PyGymEnvObs<S, T1, u8>) -> Tensor {
        let tmp = &obs.obs;
        Tensor::try_from(tmp).unwrap()
        // Tensor::try_from(&obs.obs).unwrap()
    }
}
