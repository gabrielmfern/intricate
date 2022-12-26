//! The module that contains all of the implemented optimizers in Intricate

pub mod adagrad;
#[cfg(test)]
mod adagrad_test;
pub mod adam;
pub mod basic;
#[cfg(test)]
mod basic_test;
pub mod momentum;
#[cfg(test)]
mod momentum_test;
pub mod nesterov;

pub use adagrad::AdagradOptimizer as Adagrad;
pub use adam::AdamOptimizer as Adam;
pub use basic::BasicOptimizer as Basic;
pub use momentum::MomentumOptimizer as Momentum;
pub use nesterov::NesterovOptimizer as Nesterov;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

use crate::{
    types::{KernelNotFoundError, ProgramNotFoundError},
    utils::{
        opencl::{opencl_state::EnsureKernelsAndProgramError, BufferOperationError},
        OpenCLState,
    },
};

use self::{adam::compile_adam, adagrad::compile_adagrad, momentum::compile_momentum};

pub(crate) fn compile_optimizers(
    state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    compile_adam(state)?;
    compile_adagrad(state)?;
    compile_momentum(state)?;

    Ok(())
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum that contains all of the possible errors that can happen whe trying to optimize
/// something using an Optimizer.
pub enum OptimizationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when something goes wrong on a buffer operation.
    BufferOperation(BufferOperationError),

    /// Happens when a specific program cannot be found compilled inside the OpenClState
    ProgramNotFound(ProgramNotFoundError),

    /// Happens when a kernel cannot be found inside a compilled program
    KernelNotFound(KernelNotFoundError),

    /// Happens if no command queue was found on the OpenCLState.
    NoCommandQueueFound,

    /// Happens if the state is not initialized.
    UninitializedState,
}

/// An Optimizer is something that tries to improve the learning process based on some kind of
/// implementation that adapts to the loss function's curvature.
pub trait Optimizer<'a>
where
    Self: std::fmt::Debug,
{
    /// Initializes the Optimizer by saving the OpenCLState's reference to the struct and perhaps
    /// may initialize some buffers.
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError>;

    /// Optimizes the parameters of a Layer, in the case of the Dense, the weights a biases.
    ///
    /// Mostly this is used in an Optimizer like Nesterov's that tries to predict where the
    /// paremeters are going to be.
    ///
    /// The **parameter_id** is basically used to keep track and store update vectors for more than
    /// one parameter if needed.
    fn optimize_parameters(
        &self,
        parameters: &mut Buffer<cl_float>,
        parameter_id: String,
        timestep: usize,
        layer_index: usize,
    ) -> Result<(), OptimizationError>;

    /// Computes the update vectors of some certain gradients.
    ///
    /// This is basically used for example, on the Basic optimizer, for scaling the gradients by
    /// the learning and doing some other type of transformation.
    ///
    /// The **parameter_id** is basically used to keep track and store update vectors for more than
    /// one parameter if needed.
    fn compute_update_vectors(
        &mut self,
        gradients: &Buffer<cl_float>,
        parameter_id: String,
        timestep: usize,
        layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError>;
}