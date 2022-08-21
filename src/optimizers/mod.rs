//! The module that contains all of the implemented optimizers in Intricate

use intricate_macros::ErrorsEnum;
use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

#[derive(Debug, ErrorsEnum)]
pub enum OptimizationError {
    OpenCL(ClError),
    NoCommandQueueFound,
    UninitializedState,
}

pub trait Optimizer<'a> {
    fn optimize_parameters(
        &self,
        parameters: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;

    fn compute_update_vectors(
        &self,
        gradients: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;
}
