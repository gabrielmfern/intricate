//! A module containing all of the available Loss Functions
//!
//! Also defines a simple trait implemented by Intricate on the loss functions

use std::fmt::Debug;

pub mod categorical_cross_entropy;
pub mod mean_squared;
pub mod mean_absolute;
pub mod mean_bias;

pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use mean_squared::MeanSquared;
pub use mean_absolute::MeanAbsolute;
pub use mean_bias::MeanBias;

use crate::{utils::{OpenCLState, opencl::{EnsureKernelsAndProgramError, BufferOperationError}}, types::{KernelNotFoundError, ProgramNotFoundError}};

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{device::cl_float, error_codes::ClError, memory::Buffer};

use self::{
    categorical_cross_entropy::compile_categorical_cross_entropy,
    mean_squared::compile_mean_squared,
    mean_absolute::compile_mean_absolute, mean_bias::compile_mean_bias,
};

pub(crate) fn compile_losses(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    compile_mean_squared(opencl_state)?;
    compile_categorical_cross_entropy(opencl_state)?;
    compile_mean_absolute(opencl_state)?;
    compile_mean_bias(opencl_state)?;

    Ok(())
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen when trying to compute the
/// overall loss of a Model from expected outputs with respect to actual outputs.
pub enum LossComputationError {
    /// Happens when the LossFunction trait object was not initialized.
    NotInitialized,
    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueue,

    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when the **expected outputs** and the **actual outputs** do not match in size.
    OutputsAndExpectedOutputsDoNotMatch,
    /// Happens when the given training data does not have the amount of samples specified inside
    /// of it.
    TrainingDataDoesNotHaveExpectedSamplesAmount,

    /// Happens when a required kernel was not found
    KernelNotFound(KernelNotFoundError),
    /// Happens when a required program was not found
    ProgramNotFound(ProgramNotFoundError),

    /// Happens when a buffer operation goes wrong
    BufferOperation(BufferOperationError),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen when trying to compute the
/// derivatives of the loss of a Model with respect to its outputs to do gradient descent on it.
pub enum LossToModelOutputsDerivativesComputationError {
    /// Happens when the LossFunction trait object was not initialized.
    NotInitialized,
    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueue,

    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when the **expected outputs** and the **actual outputs** do not match in size.
    OutputsAndExpectedOutputsDoNotMatch,
    /// Happens when the given training data does not have the amount of samples specified inside
    /// of it.
    TrainingDataDoesNotHaveExpectedSamplesAmount,

    /// Happens when a required kernel was not found
    KernelNotFound(KernelNotFoundError),
    /// Happens when a required program was not found
    ProgramNotFound(ProgramNotFoundError),

    /// Happens when a buffer operation goes wrong
    BufferOperation(BufferOperationError),
}

/// A simple trait implemented by Intricate that will define the base functions
/// for every Loss Function
pub trait LossFunction<'a>
where
    Self: Debug,
{
    /// Computes the `f32´ loss of between the **output samples**
    /// and the **expected output samples**.
    ///
    /// # Errors
    ///
    /// This function will return an Err if some error happened perhaps running
    /// OpenCL kernels.
    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<f32, LossComputationError>;

    /// Sets the "almost" static reference to the OpenCL context and Command Queue.
    ///
    /// # Errors
    ///
    /// This function will return an error if some error happens while compiling OpenCL
    /// programs, or any other type of OpenCL error.
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError>;

    /// Computes the derivative of the loss with respect to each one of the outputs
    /// given for some certain expected outputs.
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong when executing the kernel.
    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, LossToModelOutputsDerivativesComputationError>;
}