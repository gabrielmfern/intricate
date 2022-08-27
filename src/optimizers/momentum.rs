//! A module that contains the momentum based optimizer that tries to dampen the training process
//! as to make it improve faster.

use std::collections::HashMap;

use opencl3::{
    device::cl_float,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY},
};

use super::{OptimizationError, Optimizer};
use crate::utils::{BufferOperations, OpenCLState};

#[derive(Debug)]
/// The momentum based optimizer is one that tries to simulate momentum using a `gamma` constant
/// that defines how much of the last update vector should be added together with the current
/// update vector as to further improve the training process.
pub struct MomentumOptimizer<'a> {
    learning_rate: f32,
    momentum_gamma: f32,

    last_update_vectors: HashMap<usize, HashMap<String, Buffer<cl_float>>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> MomentumOptimizer<'a> {
    /// Creates a new instance of a Optimizer based on Momentum, that tries to speed up the
    /// training process in the right direction.
    ///
    /// The **momentum_gamma** parameter here is how much of the last update vector should be
    /// considered in the current one as to simulate momentum. This value is usually just `0.9`.
    pub fn new(learning_rate: f32, momentum_gamma: f32) -> Self {
        MomentumOptimizer {
            learning_rate,
            momentum_gamma,

            last_update_vectors: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for MomentumOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), opencl3::error_codes::ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn optimize_parameters(
        &self,
        _parameters: &mut Buffer<cl_float>,
        _parameter_id: String,
        _layer_index: usize,
    ) -> Result<(), OptimizationError> {
        Ok(())
    }

    fn compute_update_vectors(
        &mut self,
        gradients: &Buffer<cl_float>,
        parameter_id: String,
        layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        let normal_update_vector = gradients.scale(self.learning_rate, CL_MEM_READ_ONLY, state)?;

        if self.last_update_vectors.get(&layer_index).is_none() {
            self.last_update_vectors
                .insert(layer_index, HashMap::default());
        }

        let layer_update_vectors = self.last_update_vectors.get_mut(&layer_index).unwrap();

        let last_update_vector_option = layer_update_vectors.get(&parameter_id);

        let update_vector;

        if let Some(last_update_vector) = last_update_vector_option {
            if last_update_vector.size()? != normal_update_vector.size()? {
                update_vector = last_update_vector
                    .scale(self.momentum_gamma, CL_MEM_READ_ONLY, state)?
                    .add(&normal_update_vector, CL_MEM_READ_ONLY, state)?;
            } else {
                update_vector = normal_update_vector;
            }
        } else {
            update_vector = normal_update_vector;
        }

        layer_update_vectors
            .insert(parameter_id, update_vector.clone(CL_MEM_READ_ONLY, state)?);

        Ok(update_vector)
    }
}
