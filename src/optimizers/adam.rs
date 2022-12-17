//! The modulem that contains the Adam optimizer.

use std::collections::HashMap;

use opencl3::{memory::Buffer, types::cl_float};

use crate::utils::{OpenCLState, BufferOperations, opencl::InplaceBufferOperations};

use super::{Optimizer, OptimizationError};


#[derive(Debug)]
/// An optimizer that works well with pretty much everything. It has four hyper parameters that can
/// be tuned for your Model. This Optimizer tries to combine features from other optimizers but
/// without needing as much memory requirements such as Adagrad or sometimes blowing up to infinity
/// with Nesterov's optimizer or just the basic Momentum optimizer.
pub struct AdamOptimizer<'a> {
    learning_rate_alpha: f32,
    decay_rate_beta_1: f32,
    decay_rate_beta_2: f32,

    safety_epsilon: f32,

    last_moment_1_per_parameter: HashMap<(usize, String), Buffer<cl_float>>,
    last_moment_2_per_parameter: HashMap<(usize, String), Buffer<cl_float>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> AdamOptimizer<'a> {
    /// Creates a new instance of the Adam optimizer.
    ///
    /// The hyper parameters here are usually just 0.001, 0.9, 0.99 and 0.00000001 respectively.
    pub fn new(
        learning_rate: f32, 
        decay_rate_beta_1: f32, 
        decay_rate_beta_2: f32, 
        safety_epsilon: f32,
    ) -> Self {
        AdamOptimizer {
            learning_rate_alpha: learning_rate,
            decay_rate_beta_1,
            decay_rate_beta_2,

            safety_epsilon,

            last_moment_1_per_parameter: HashMap::default(),
            last_moment_2_per_parameter: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for AdamOptimizer<'a> {
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
        timestep: usize, 
        layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        let mut current_moment_first_estimate = gradients.scale(1.0 - self.decay_rate_beta_1, state)?;

        if let Some(last_moment_1_estimate) = self.last_moment_1_per_parameter.get(&(layer_index, parameter_id.to_string())) {
            current_moment_first_estimate.add_inplc(
                &last_moment_1_estimate.scale(self.decay_rate_beta_1, state)?, 
                state
            )?;
        }

        self.last_moment_1_per_parameter.insert(
            (layer_index, parameter_id.to_string()), 
            current_moment_first_estimate.clone(state)?
        );

        let mut current_moment_second_esteimate = gradients.multiply(gradients, state)?
            .scale(1.0 - self.decay_rate_beta_2, state)?;

        if let Some(last_moment_second_estimate) = self.last_moment_2_per_parameter.get(&(layer_index, parameter_id.to_string())) {
            current_moment_second_esteimate.add_inplc(
                &last_moment_second_estimate.scale(self.decay_rate_beta_2, state)?, 
                state
            )?;
        }

        self.last_moment_2_per_parameter.insert(
            (layer_index, parameter_id.to_string()), 
            current_moment_second_esteimate.clone(state)?
        );

        // bias-correct the estimates inplace
        current_moment_first_estimate.scale_inplc(
            1.0 / (1.0 - self.decay_rate_beta_1.powf(timestep as f32)), 
            state
        )?;
        current_moment_second_esteimate.scale_inplc(
            1.0 / (1.0 - self.decay_rate_beta_2.powf(timestep as f32)), 
            state
        )?;

        Ok(
            current_moment_first_estimate.divide(
                &current_moment_second_esteimate.sqrt(state)?
                    .shift(self.safety_epsilon, state)?,
                state
            )?
            .scale(self.learning_rate_alpha, state)?
        )
    }
}