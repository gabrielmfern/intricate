//! A module that contains the basic optimizer.

use opencl3::{
    device::cl_float,
    memory::Buffer,
};

use super::{OptimizationError, Optimizer};
use crate::utils::{BufferOperations, OpenCLState};

#[derive(Debug)]
/// A very basic and archaic optimizer that does not alter the parameters and just scaled the
/// gradients by a fixed learning rate to compute the update vectors.
pub struct BasicOptimizer<'a> {
    learning_rate: f32,
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> BasicOptimizer<'a> {
    /// Creates a new instance of the Basic Optimizer with a certain learning rate.
    pub fn new(learning_rate: f32) -> Self {
        BasicOptimizer {
            learning_rate,
            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for BasicOptimizer<'a> {
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
        _parameter_id: String,
        _timestep: usize, 
        _layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        Ok(gradients.scale(self.learning_rate, state)?)
    }
}


#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::{
        optimizers::Optimizer,
        utils::{
            opencl::{BufferLike, DeviceType},
            setup_opencl,
        },
    };

    use super::BasicOptimizer;

    #[test]
    fn should_compute_update_vectors_correctly() {
        // theta = theta - v_t
        // v_t = learning_rate * gradients of theta with respect to the loss
        let mut rng = thread_rng();

        let gradients = vec![rng.gen_range(0f32..1f32)];

        let learning_rate = 0.01;

        let expected_update_vector = vec![learning_rate * gradients[0]];

        let state = setup_opencl(DeviceType::GPU).unwrap();

        let gradients_buf = gradients
            .to_buffer(false, &state)
            .unwrap();

        let mut optimizer = BasicOptimizer::new(learning_rate);
        optimizer.init(&state).unwrap();

        let update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0)
            .unwrap();

        let update_vector =
            Vec::<f32>::from_buffer(&update_buf, false, &state).unwrap();

        assert!(
            (dbg!(update_vector[0]) - dbg!(expected_update_vector[0])).abs()
                / expected_update_vector[0]
                <= 0.001
        );
    }
}