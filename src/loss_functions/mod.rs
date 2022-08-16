//! A module containing all of the available Loss Functions
//!
//! Also defines a simple trait implemented by Intricate on the loss functions

use std::fmt::Debug;

pub mod categorical_cross_entropy;
pub mod mean_squared;

pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use mean_squared::MeanSquared;

use crate::types::CompilationOrOpenCLError;

use opencl3::{
    command_queue::CommandQueue, context::Context, device::cl_float, error_codes::ClError,
    memory::Buffer,
};

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
    ) -> Result<f32, ClError>;

    /// Sets the "almost" static reference to the OpenCL context and Command Queue.
    ///
    /// # Errors
    ///
    /// This function will return an error if some error happens while compiling OpenCL
    /// programs, or any other type of OpenCL error.
    fn init(&mut self, context: &'a Context, queue: &'a CommandQueue) -> Result<(), CompilationOrOpenCLError>;

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
    ) -> Result<Buffer<cl_float>, ClError>;
}