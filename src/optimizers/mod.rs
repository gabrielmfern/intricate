//! The module that contains all of the implemented optimizers in Intricate

pub mod dummy;

pub use dummy::Dummy;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

use crate::utils::{opencl::BufferOperationError, OpenCLState};

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum OptimizationError {
    OpenCL(ClError),
    BufferOperation(BufferOperationError),
    NoCommandQueueFound,
    UninitializedState,
}

pub trait Optimizer<'a> {
    fn init(
        &mut self,
        opencl_state: &'a OpenCLState,
    ) -> Result<(), ClError>;

    fn optimize_parameters(
        &self,
        parameters: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;

    fn compute_update_vectors(
        &self,
        gradients: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;
}