use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY}, device::cl_float};

use super::{Optimizer, OptimizationError};
use crate::utils::{BufferOperations, OpenCLState};


#[derive(Debug)]
pub struct Dummy<'a> {
    learning_rate: f32,
    opencl_state: &'a OpenCLState,
}

impl<'a> Optimizer<'a> for Dummy<'a> {
    fn optimize_parameters(
        &self,
        parameters: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        Ok(parameters.clone(CL_MEM_READ_ONLY, self.opencl_state)?)
    } 

    fn compute_update_vectors(
        &self,
        gradients: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        Ok(gradients.scale(self.learning_rate, CL_MEM_READ_ONLY, self.opencl_state)?)
    }
}
