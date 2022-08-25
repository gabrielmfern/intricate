//! A module that contains the basic optimizer.

use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY}, device::cl_float};

use super::{Optimizer, OptimizationError};
use crate::{utils::{BufferOperations, OpenCLState}, types::ModelOptimizer};


#[derive(Debug)]
/// A very basic and archaic optimizer that does not alter the parameters and just scaled the
/// gradients by a fixed learning rate to compute the update vectors.
pub struct Basic<'a> {
    learning_rate: f32,
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Basic<'a> {
    /// Creates a new instance of the Basic optimizer but as an instance of the ModelOptimizer enum
    pub fn new(learning_rate: f32) -> ModelOptimizer<'a> {
        Self::new_raw(learning_rate).into()
    }

    /// Creates a raw instance of the Basic optimizer.
    pub fn new_raw(learning_rate: f32) -> Self {
        Basic { learning_rate, opencl_state: None }
    }
}

impl<'a> Optimizer<'a> for Basic<'a> {
    fn init(
        &mut self,
        opencl_state: &'a OpenCLState,
    ) -> Result<(), opencl3::error_codes::ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn optimize_parameters(
        &self,
        parameters: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        Ok(parameters.clone(CL_MEM_READ_ONLY, state)?)
    } 

    fn compute_update_vectors(
        &self,
        gradients: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        Ok(gradients.scale(self.learning_rate, CL_MEM_READ_ONLY, state)?)
    }
}