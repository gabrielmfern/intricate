//! The module that contains the Adagrad optimizer.

use std::collections::HashMap;

use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

use crate::utils::{opencl::InplaceBufferOperations, BufferOperations, OpenCLState};

use super::{OptimizationError, Optimizer};

#[derive(Debug)]
/// The Adagrad Optimizer does a gradient-based optimization that adapts the learning rates for
/// parameters that are much more necessary than others for the given purpose of the Model.
pub struct AdagradOptimizer<'a> {
    learning_rate: f32,
    epsilon: f32,

    gradients_history_summation_per_parameter: HashMap<usize, HashMap<String, Buffer<cl_float>>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> AdagradOptimizer<'a> {
    /// Creates a new uninitialized instance of the Adagrad optimizer.
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        AdagradOptimizer {
            learning_rate,
            epsilon,
            gradients_history_summation_per_parameter: HashMap::default(),
            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for AdagradOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError> {
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

        if !self
            .gradients_history_summation_per_parameter
            .contains_key(&layer_index)
        {
            self.gradients_history_summation_per_parameter
                .insert(layer_index, HashMap::default());
        }

        let layer_gradients_history_summation = self
            .gradients_history_summation_per_parameter
            .get_mut(&layer_index)
            .unwrap();

        let mut update_vector;

        if let Some(gradients_history_summation) =
            layer_gradients_history_summation.get_mut(&parameter_id)
        {
            update_vector = gradients_history_summation.shift(self.epsilon, state)?;

            update_vector.inverse_sqrt_inplc(state)?;
            update_vector.scale_inplc(self.learning_rate, state)?;
            update_vector.multiply_inplc(gradients, state)?;

            let squared_gradients = gradients.multiply(gradients, state)?;
            gradients_history_summation.add_inplc(&squared_gradients, state)?;
        } else {
            update_vector = gradients.scale(self.learning_rate, state)?;

            let squared_gradients = gradients.multiply(gradients, state)?;
            layer_gradients_history_summation.insert(parameter_id.to_string(), squared_gradients);
        }

        Ok(update_vector)
    }
}

#[cfg(test)]
mod tests {
    use super::AdagradOptimizer;
    use crate::{optimizers::Optimizer, utils::opencl::*};
    use opencl3::memory::CL_MEM_READ_ONLY;
    use rand::prelude::*;

    #[test]
    fn should_compute_update_vectors_correctly() {
        let mut rng = thread_rng();

        let gradients = vec![rng.gen_range(0f32..1f32)];

        let episilon = 0.000_000_01;
        let learning_rate = 0.01;

        let expected_first_update_vector = vec![learning_rate * gradients[0]];
        let expected_second_update_vector =
            vec![learning_rate / (gradients[0] + episilon) * gradients[0]];
        let expected_third_update_vector = vec![
            learning_rate / ((gradients[0].powf(2.0) + gradients[0].powf(2.0)).sqrt() + episilon)
                * gradients[0],
        ];

        let state = setup_opencl(DeviceType::GPU).unwrap();

        let gradients_buf = gradients
            .to_buffer(CL_MEM_READ_ONLY, false, &state)
            .unwrap();

        let mut optimizer = AdagradOptimizer::new(learning_rate, episilon);
        optimizer.init(&state).unwrap();

        let first_update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0)
            .unwrap();
        let second_update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0)
            .unwrap();
        let third_update_buf = optimizer
            .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0)
            .unwrap();

        let first_update_vector =
            Vec::<f32>::from_buffer(&first_update_buf, false, &state).unwrap();
        let second_update_vector =
            Vec::<f32>::from_buffer(&second_update_buf, false, &state).unwrap();
        let third_update_vector =
            Vec::<f32>::from_buffer(&third_update_buf, false, &state).unwrap();

        assert!(
            (dbg!(first_update_vector[0]) - dbg!(expected_first_update_vector[0])).abs()
                / expected_first_update_vector[0]
                <= 0.001
        );
        assert!(
            (dbg!(second_update_vector[0]) - dbg!(expected_second_update_vector[0])).abs()
                / expected_second_update_vector[0]
                <= 0.001
        );
        assert!(
            (dbg!(third_update_vector[0]) - dbg!(expected_third_update_vector[0])).abs()
                / expected_third_update_vector[0]
                <= 0.001
        );
    }
}
