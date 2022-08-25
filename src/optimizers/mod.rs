//! The module that contains all of the implemented optimizers in Intricate

pub mod basic;

pub use basic::BasicOptimizer;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

use crate::utils::{opencl::BufferOperationError, OpenCLState};

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum that contains all of the possible errors that can happen whe trying to optimize
/// something using an Optimizer. 
pub enum OptimizationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when something goes wrong on a buffer operation.
    BufferOperation(BufferOperationError),
    /// Happens if no command queue was found on the OpenCLState.
    NoCommandQueueFound,
    /// Happens if the state is not initialized.
    UninitializedState,
}

/// An Optimizer is something that tries to improve the learning process based on some kind of
/// implementation that adapts to the loss function's curvature.
pub trait Optimizer<'a> {
    /// Initializes the Optimizer by saving the OpenCLState's reference to the struct and perhaps
    /// may initialize some buffers.
    fn init(
        &mut self,
        opencl_state: &'a OpenCLState,
    ) -> Result<(), ClError>;

    /// Optimizes the parameters of a Layer, in the case of the Dense, the weights a biases.
    ///
    /// Mostly this is used in an Optimizer like Nesterov's that tries to predict where the
    /// paremeters are going to be.
    fn optimize_parameters(
        &self,
        parameters: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;

    /// Computes the update vectors of some certain gradients.
    ///
    /// This is basically used for example, on the Basic optimizer, for scaling the gradients by
    /// the learning and doing some other type of transformation.
    fn compute_update_vectors(
        &self,
        gradients: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, OptimizationError>;
}