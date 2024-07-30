use super::{get_field_type, get_type_str};
use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    let DeriveInput { ident, data, .. } = input;
    let field_type = get_field_type(data);
    let field_type_str = get_type_str(
        field_type.clone(),
        "The item for deriving Act must be a new type like Act(PyGymEnvContinuousAct<ActShape, f64>)",
    );

    let output = if field_type_str == "PyGymEnvContinuousAct" {
        py_gym_env_cont_act(ident, field_type)
    } else if field_type_str == "PyGymEnvDiscreteAct" {
        py_gym_env_disc_act(ident, field_type)
    } else if field_type_str == "BorderAtariAct" {
        atari_env_act(ident, field_type)
    } else {
        panic!(
            "Deriving Act support PyGymEnvDiscreteAct, PyGymEnvContinuousAct, or BorderAtariAct, given {:?}",
            field_type_str
        );
    };

    output.into()
}

fn py_gym_env_cont_act(
    ident: proc_macro2::Ident,
    field_type: syn::Type,
) -> proc_macro2::TokenStream {
    #[allow(unused_mut)]
    let mut output = common(ident.clone(), field_type.clone());

    #[cfg(feature = "tch")]
    output.extend(quote! {
        impl From<#ident> for tch::Tensor {
            fn from(act: #ident) -> tch::Tensor {
                let v = act.0.act.iter().map(|e| *e as f64).collect::<Vec<_>>();
                let t: tch::Tensor = std::convert::TryFrom::<Vec<f64>>::try_from(v).unwrap();

                // The first dimension of the action tensor is the number of processes,
                // which is 1 for the non-vectorized environment.
                t.unsqueeze(0)
            }
        }

        impl From<tch::Tensor> for #ident {
            /// `t` must be a 1-dimentional tensor of `f64`.
            fn from(t: tch::Tensor) -> Self {
                // In non-vectorized environment, the batch dimension is not required, thus dropped.
                let shape = t.size()[1..]
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<_>>();
                let act: Vec<f64> = t.into();

                let act = ndarray::Array1::<f64>::from(act).into_shape(ndarray::IxDyn(&shape)).unwrap();

                #ident(PyGymEnvContinuousAct::new(act))
            }
        }
    }.into_iter());

    output
}

fn py_gym_env_disc_act(
    ident: proc_macro2::Ident,
    field_type: syn::Type,
) -> proc_macro2::TokenStream {
    #[allow(unused_mut)]
    let mut output = common(ident.clone(), field_type.clone());

    #[cfg(feature = "tch")]
    output.extend(quote! {
        impl From<#ident> for tch::Tensor {
            fn from(act: #ident) -> tch::Tensor {
                let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
                let t: tch::Tensor = std::convert::TryFrom::<Vec<i64>>::try_from(v).unwrap();

                // The first dimension is for the batch
                t.unsqueeze(0)
            }
        }

        impl From<tch::Tensor> for #ident {
            fn from(t: tch::Tensor) -> Self {
                let data: Vec<i64> = t.into();
                let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
                #ident(PyGymEnvDiscreteAct::new(data))
            }
        }
    }.into_iter());

    output
}

fn atari_env_act(
    ident: proc_macro2::Ident,
    field_type: syn::Type,
) -> proc_macro2::TokenStream {
    #[allow(unused_mut)]
    let mut output = common(ident.clone(), field_type.clone());

    #[cfg(feature = "tch")]
    output.extend(quote! {
        impl From<#ident> for tch::Tensor {
            fn from(act: #ident) -> tch::Tensor {
                // let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
                let v = vec![act.0.act as i64];
                let t: tch::Tensor = std::convert::TryFrom::<Vec<i64>>::try_from(v).unwrap();

                // The first dimension is for the batch
                t.unsqueeze(0)
            }
        }

        impl From<tch::Tensor> for #ident {
            fn from(t: tch::Tensor) -> Self {
                let data: Vec<i64> = t.into();
                // Non-vectorized environment
                #ident(BorderAtariAct::new(data[0] as u8))
            }
        }
    }.into_iter());

    output
}

fn common(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
    quote! {
        impl border_core::Act for #ident {
            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl Into<#field_type> for #ident {
            fn into(self) -> #field_type {
                self.0
            }
        }
    }
}