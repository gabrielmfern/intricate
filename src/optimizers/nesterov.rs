//! A module that contains the momentum based optimizer that tries to dampen the training process
//! as to make it improve faster.

use std::collections::HashMap;

use opencl3::{
    device::cl_float,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
};

use super::{OptimizationError, Optimizer};
use crate::utils::{opencl::InplaceBufferOperations, BufferOperations, OpenCLState};

#[derive(Debug)]
/// The momentum based optimizer is one that tries to simulate momentum using a `gamma` constant
/// that defines how much of the last update vector should be added together with the current
/// update vector as to further improve the training process.
pub struct NesterovMomentumAcceleratedOptimizer<'a> {
    learning_rate: f32,
    momentum_gamma: f32,

    last_update_vectors: HashMap<usize, HashMap<String, Buffer<cl_float>>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> NesterovMomentumAcceleratedOptimizer<'a> {
    /// Creates a new instance of a Optimizer based on Momentum, that tries to speed up the
    /// training process in the right direction.
    ///
    /// The **momentum_gamma** parameter here is how much of the last update vector should be
    /// considered in the current one as to simulate momentum. This value is usually just `0.9`.
    pub fn new(learning_rate: f32, momentum_gamma: f32) -> Self {
        NesterovMomentumAcceleratedOptimizer {
            learning_rate,
            momentum_gamma,

            last_update_vectors: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for NesterovMomentumAcceleratedOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), opencl3::error_codes::ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn optimize_parameters(
        &self,
        parameters: &mut Buffer<cl_float>,
        parameter_id: String,
        layer_index: usize,
    ) -> Result<(), OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        if let Some(layer_update_vectors) = self.last_update_vectors.get(&layer_index) {
            if let Some(parameter_last_update_vector) = layer_update_vectors.get(&parameter_id) {
                parameters.subtract_inplc(
                    &parameter_last_update_vector.scale(
                        self.momentum_gamma,
                        CL_MEM_READ_ONLY,
                        state,
                    )?,
                    state,
                )?;
            }
        }

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
                let mut scalled_last_update_vec = last_update_vector
                    .scale(self.momentum_gamma, CL_MEM_READ_WRITE, state)?;
                scalled_last_update_vec.add_inplc(&normal_update_vector, state)?;

                update_vector = scalled_last_update_vec;
            } else {
                update_vector = normal_update_vector;
            }
        } else {
            update_vector = normal_update_vector;
        }

        layer_update_vectors.insert(parameter_id, update_vector.clone(CL_MEM_READ_ONLY, state)?);

        Ok(update_vector)
    }
}
