//! A module that contains the momentum based optimizer that tries to dampen the training process
//! as to make it improve faster.

use std::collections::HashMap;

use opencl3::{
    device::cl_float,
    memory::Buffer,
};

use super::{OptimizationError, Optimizer};
use crate::utils::{opencl::InplaceBufferOperations, BufferOperations, OpenCLState};

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
        _timestep: usize, 
        _layer_index: usize,
    ) -> Result<(), OptimizationError> {
        Ok(())
    }

    fn compute_update_vectors(
        &mut self,
        gradients: &Buffer<cl_float>,
        parameter_id: String,
        _timestep: usize, 
        layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        let normal_update_vector = gradients.scale(self.learning_rate, state)?;

        if !self.last_update_vectors.contains_key(&layer_index) {
            self.last_update_vectors
                .insert(layer_index, HashMap::default());
        }

        let layer_update_vectors = self.last_update_vectors.get_mut(&layer_index).unwrap();

        let last_update_vector_option = layer_update_vectors.get(&parameter_id);

        let update_vector;

        if let Some(last_update_vector) = last_update_vector_option {
            let mut scalled_last_update_vec =
                last_update_vector.scale(self.momentum_gamma, state)?;
            scalled_last_update_vec.add_inplc(&normal_update_vector, state)?;

            update_vector = scalled_last_update_vec;
        } else {
            update_vector = normal_update_vector;
        }

        layer_update_vectors.insert(parameter_id, update_vector.clone(state)?);

        Ok(update_vector)
    }
}


#[cfg(test)]
mod momentum_tests {
    use rand::prelude::*;

    use crate::{
        optimizers::Optimizer,
        utils::{
            opencl::{BufferLike, DeviceType},
            setup_opencl,
        },
    };

    use super::MomentumOptimizer;

    #[test]
    fn should_compute_update_vectors_correctly() {
        // theta = theta - v_t
        // v_t = gamma * v_(t-1) + learning_rate * gradients of theta with respect to the loss
        let mut rng = thread_rng();

        let gradients = vec![rng.gen_range(0f32..1f32)];

        let gamma = 0.9;
        let learning_rate = 0.01;

        let expected_inital_update_vector = vec![learning_rate * gradients[0]];
        let expected_second_update_vector =
            vec![gamma * expected_inital_update_vector[0] + learning_rate * gradients[0]];

        let state = setup_opencl(DeviceType::GPU).unwrap();

        let gradients_buf = gradients
            .to_buffer(false, &state)
            .unwrap();

        let mut optimizer = MomentumOptimizer::new(learning_rate, gamma);
        optimizer.init(&state).unwrap();

        let initial_update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
            .unwrap();
        let secondary_update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
            .unwrap();

        let initial_update_vector =
            Vec::<f32>::from_buffer(&initial_update_buf, false, &state).unwrap();
        let secondary_update_vector =
            Vec::<f32>::from_buffer(&secondary_update_buf, false, &state).unwrap();

        assert!(
            (dbg!(initial_update_vector[0]) - dbg!(expected_inital_update_vector[0])).abs()
                / expected_inital_update_vector[0]
                <= 0.001
        );
        assert!(
            (dbg!(secondary_update_vector[0]) - dbg!(expected_second_update_vector[0])).abs()
                / expected_second_update_vector[0]
                <= 0.001
        );
    }
}