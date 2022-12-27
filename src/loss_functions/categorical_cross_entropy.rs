//! The module that implements the Categorical Cross Entropy loss function.

use std::mem;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    command_queue::CommandQueue,
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};

use crate::utils::opencl::ensure_program;
use crate::utils::opencl::opencl_state::EnsureKernelsAndProgramError;
use crate::utils::opencl::{empty_buffer, opencl_state::IntricateProgram};
use crate::utils::BufferOperations;
use crate::utils::OpenCLState;
use crate::{
    loss_functions::LossFunction,
    utils::opencl::{BufferConversionError, BufferOperationError},
};

use super::LossToModelOutputsDerivativesComputationError;
use super::{LossComputationError, LossFn};

const PROGRAM_NAME: &str = "CATEGORICAL_CROSS_ENTROPY";
const PROGRAM_SOURCE: &str = include_str!("kernels/categorical_cross_entropy.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const NORMALIZE_OUTPUTS_KERNEL: &str = "normalize_outputs";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_OPTIMIZED_FOR_SOFTMAX_KERNEL: &str =
    "compute_loss_to_output_derivatives_optimized_for_softmax";

pub(crate) fn compile_categorical_cross_entropy(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        COMPUTE_LOSS_KERNEL,
        NORMALIZE_OUTPUTS_KERNEL,
        COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL,
        COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_OPTIMIZED_FOR_SOFTMAX_KERNEL,
    ];

    ensure_program(opencl_state, PROGRAM_NAME, PROGRAM_SOURCE, "", kernels)?;

    Ok(())
}

#[derive(Debug)]
/// The **Categorical Cross Entropy** loss function made for, as the name may suggest, classfying
/// the loss of a categorical Model.
/// May yield some NaN values when being used because of the nature of this loss function
///
/// This loss function is very good for categorical problems because it penalizes when some values
/// are high when they should be closer to 0, and if the values are a bit far from what they are
/// expected to be they are much more penalized that in other loss functions.
///
/// Disclaimer: This loss function will end up deactivating the last SoftMax layer if the Model
/// ends with it for the purpose of optimizing the calculations and avoiding NaN's.
pub struct CategoricalCrossEntropy<'a> {
    opencl_state: Option<&'a OpenCLState>,
    /// This keeps track of weather or not the last layer in the Model using the loss_fn is a
    /// Softmax
    is_optimized_for_softmax: bool,
}

impl<'a> CategoricalCrossEntropy<'a> {
    /// Crates a new raw instance of the Categorical Cross Entropy but as a raw version of the struct.
    ///
    /// Be aware that after creation this needs to be called the `init` method before computing the
    /// loss or anything like that.
    pub fn new_raw() -> CategoricalCrossEntropy<'a> {
        CategoricalCrossEntropy {
            opencl_state: None,
            is_optimized_for_softmax: false,
        }
    }

    /// Crates a new instance of the Categorical Cross Entropy but as a raw version of the struct.
    ///
    /// Be aware that after creation this needs to be called the `init` method before computing the
    /// loss or anything like that.
    pub fn new() -> LossFn<'a> {
        Self::new_raw().into()
    }

    pub(crate) fn set_optimized_for_softmax(&mut self, optimized: bool) -> () {
        self.is_optimized_for_softmax = optimized;
    }
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen when reducing the output samples
/// into the summation of the outputs per sample
pub enum ReduceOutputsPerSampleError {
    /// Happens when something goes wrong with OpenCL directly
    OpenCL(ClError),
    /// Happens when something goes wrong with a Buffer Operation
    BufferOperation(BufferOperationError),
    /// Happens when something goes wrong with a Buffer Conversion
    BufferConversion(BufferConversionError),
}

// pub(crate) fn sum_outputs_per_sample(
//     state: &OpenCLState,
//     outputs: &Buffer<cl_float>,
//     outputs_amount: usize,
//     samples_amount: usize,
// ) -> Result<Buffer<cl_float>, ReduceOutputsPerSampleError> {
//     let mut resulting_vec = Vec::with_capacity(samples_amount);

//     for sample_index in 0..samples_amount {
//         let offset = sample_index * outputs_amount;
//         let count = outputs_amount;
//         let outputs_for_sample = outputs.create_sub_buffer(CL_MEM_READ_ONLY, offset, count)?;

//         resulting_vec.push(
//             outputs_for_sample.sum(state)?
//         );
//     }

//     Ok(resulting_vec.to_buffer(false, state)?)
// }

fn normalize_output_samples(
    output_samples: &Buffer<cl_float>,
    state: &OpenCLState,
    queue: &CommandQueue,
    program: &IntricateProgram,
    samples_amount: usize,
    outputs_amount: usize,
    outputs_total_count: usize,
) -> Result<Buffer<cl_float>, BufferOperationError> {
    let outputs_sum_per_row = output_samples.sum_2d_per_row(state, outputs_amount)?;
    let mut normalized_outputs = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

    let normalize_outputs_kernel = program.get_krnl(NORMALIZE_OUTPUTS_KERNEL)?;

    ExecuteKernel::new(normalize_outputs_kernel)
        .set_arg(output_samples)
        .set_arg(&outputs_sum_per_row)
        .set_arg(&mut normalized_outputs)
        .set_arg(&(samples_amount as cl_int))
        .set_arg(&(outputs_amount as cl_int))
        .set_global_work_sizes(&[samples_amount, outputs_amount])
        .enqueue_nd_range(queue)?
        .wait()?;

    Ok(normalized_outputs)
}

impl<'a> LossFunction<'a> for CategoricalCrossEntropy<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<f32, LossComputationError> {
        if self.opencl_state.is_none() {
            return Err(LossComputationError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.len() == 0 {
            return Err(LossComputationError::NoCommandQueue);
        }

        let queue = state.queues.first().unwrap();

        let outputs_size = output_samples.size()?;

        if output_samples.size()? != expected_outputs.size()? {
            return Err(LossComputationError::OutputsAndExpectedOutputsDoNotMatch);
        }

        let outputs_total_count = outputs_size / mem::size_of::<cl_float>();

        if outputs_total_count % samples_amount != 0 {
            return Err(LossComputationError::TrainingDataDoesNotHaveExpectedSamplesAmount);
        }

        let outputs_amount = outputs_total_count / samples_amount;

        let program = state.get_prgm(PROGRAM_NAME)?;
        let sample_losses_buffer = empty_buffer(samples_amount, CL_MEM_READ_WRITE, state)?;

        let compute_loss_kernel = program.get_krnl(COMPUTE_LOSS_KERNEL)?;

        if self.is_optimized_for_softmax {
            ExecuteKernel::new(compute_loss_kernel)
                .set_arg(output_samples)
                .set_arg(expected_outputs)
                .set_arg(&sample_losses_buffer)
                .set_arg(&(outputs_amount as cl_int))
                .set_arg(&(samples_amount as cl_int))
                .set_global_work_size(samples_amount)
                .enqueue_nd_range(queue)?;
        } else {
            let normalized_outputs = normalize_output_samples(
                output_samples,
                state,
                queue,
                program,
                samples_amount,
                outputs_amount,
                outputs_total_count,
            )?;

            ExecuteKernel::new(compute_loss_kernel)
                .set_arg(&normalized_outputs)
                .set_arg(expected_outputs)
                .set_arg(&sample_losses_buffer)
                .set_arg(&(outputs_amount as cl_int))
                .set_arg(&(samples_amount as cl_int))
                .set_global_work_size(samples_amount)
                .enqueue_nd_range(queue)?;
        }

        queue.finish()?;

        Ok(sample_losses_buffer.sum(state)? / samples_amount as f32)
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, LossToModelOutputsDerivativesComputationError> {
        if self.opencl_state.is_none() {
            return Err(LossToModelOutputsDerivativesComputationError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.len() == 0 {
            return Err(LossToModelOutputsDerivativesComputationError::NoCommandQueue);
        }

        let queue = state.queues.first().unwrap();

        let outputs_size = output_samples.size()?;

        if output_samples.size()? != expected_outputs.size()? {
            return Err(
                LossToModelOutputsDerivativesComputationError::OutputsAndExpectedOutputsDoNotMatch,
            );
        }

        let outputs_total_count = outputs_size / mem::size_of::<cl_float>();

        if outputs_total_count % samples_amount != 0 {
            return Err(LossToModelOutputsDerivativesComputationError::TrainingDataDoesNotHaveExpectedSamplesAmount);
        }

        let outputs_amount = outputs_total_count / samples_amount;

        let program = state.get_prgm(PROGRAM_NAME)?;

        let derivatives_buffer = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

        if self.is_optimized_for_softmax {
            let optimized_loss_to_output_deriv_kernel = program
                .get_krnl(COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_OPTIMIZED_FOR_SOFTMAX_KERNEL)?;
            ExecuteKernel::new(optimized_loss_to_output_deriv_kernel)
                .set_arg(output_samples)
                .set_arg(expected_outputs)
                .set_arg(&derivatives_buffer)
                .set_arg(&(samples_amount as cl_int))
                .set_arg(&(outputs_amount as cl_int))
                .set_global_work_sizes(&[samples_amount, outputs_amount])
                .enqueue_nd_range(queue)?;
        } else {
            let normalized_outputs = normalize_output_samples(
                output_samples,
                state,
                queue,
                program,
                samples_amount,
                outputs_amount,
                outputs_total_count,
            )?;

            let loss_to_output_deriv_kernel =
                program.get_krnl(COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)?;
            ExecuteKernel::new(loss_to_output_deriv_kernel)
                .set_arg(&normalized_outputs)
                .set_arg(expected_outputs)
                .set_arg(&derivatives_buffer)
                .set_arg(&(samples_amount as cl_int))
                .set_arg(&(outputs_amount as cl_int))
                .set_global_work_sizes(&[samples_amount, outputs_amount])
                .enqueue_nd_range(queue)?
                .wait()?;
        }

        queue.finish()?;

        Ok(derivatives_buffer)
    }
}
