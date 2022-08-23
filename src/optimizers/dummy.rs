use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY}, device::cl_float};

use super::{Optimizer, OptimizationError};
use crate::{utils::{BufferOperations, OpenCLState}, types::PossibleOptimizer};


#[derive(Debug)]
pub struct Dummy<'a> {
    learning_rate: f32,
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Dummy<'a> {
    pub fn new(learning_rate: f32) -> PossibleOptimizer {
        Self::new_raw(learning_rate).into()
    }

    pub fn new_raw(learning_rate: f32) -> Self {
        Dummy { learning_rate, opencl_state: None }
    }
}

impl<'a> Optimizer<'a> for Dummy<'a> {
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