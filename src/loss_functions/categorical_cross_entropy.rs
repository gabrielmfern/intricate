//! The module that implements the Categorical Cross Entropy loss function.

use std::mem;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY},
};

use crate::{loss_functions::LossFunction, utils::opencl::{BufferOperationError, BufferConversionError, BufferLike}};
use crate::utils::opencl::empty_buffer;
use crate::utils::opencl::ensure_program;
use crate::utils::opencl::EnsureKernelsAndProgramError;
use crate::utils::BufferOperations;
use crate::utils::OpenCLState;

use super::LossComputationError;
use super::LossToModelOutputsDerivativesComputationError;

const PROGRAM_NAME: &str = "CATEGORICAL_CROSS_ENTROPY";
const PROGRAM_SOURCE: &str = include_str!("kernels/categorical_cross_entropy.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const NORMALIZE_OUTPUTS_KERNEL: &str = "normalize_outputs";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";

pub(crate) fn compile_categorical_cross_entropy(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        COMPUTE_LOSS_KERNEL.to_string(),
        NORMALIZE_OUTPUTS_KERNEL.to_string(),
        COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL.to_string(),
    ];

    ensure_program(
        opencl_state,
        PROGRAM_NAME.to_string(),
        PROGRAM_SOURCE.to_string(),
        "".to_string(),
        kernels,
    )?;

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
pub struct CategoricalCrossEntropy<'a> {
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> CategoricalCrossEntropy<'a> {
    /// Crates a new instance of the Categorical Cross Entropy but as a raw version of the struct.
    ///
    /// Be aware that after creation this needs to be called the `init` method before computing the
    /// loss or anything like that.`
    pub fn new() -> CategoricalCrossEntropy<'a> {
        CategoricalCrossEntropy { opencl_state: None }
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

pub(crate) fn sum_outputs_per_sample(
    state: &OpenCLState,
    outputs: &Buffer<cl_float>,
    outputs_amount: usize,
    samples_amount: usize,
) -> Result<Buffer<cl_float>, ReduceOutputsPerSampleError> {
    let mut resulting_vec = Vec::with_capacity(samples_amount);

    for sample_index in 0..samples_amount {
        let offset = sample_index * outputs_amount;
        let count = outputs_amount;
        let outputs_for_sample = outputs.create_sub_buffer(CL_MEM_READ_ONLY, offset, count)?;

        resulting_vec.push(
            outputs_for_sample.sum(state)?
        );
    }

    Ok(resulting_vec.to_buffer(false, state)?)
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

        // let outputs_summation_per_sample = sum_outputs_per_sample(
        //     state, 
        //     output_samples, 
        //     outputs_amount, 
        //     samples_amount
        // )?;
        // let normalized_outputs = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

        let program = state.get_prgm(PROGRAM_NAME)?;

        // let normalize_outputs_kernel = program.get_krnl(NORMALIZE_OUTPUTS_KERNEL)?;

        // ExecuteKernel::new(normalize_outputs_kernel)
        //     .set_arg(output_samples)
        //     .set_arg(&outputs_summation_per_sample)
        //     .set_arg(&normalized_outputs)
        //     .set_arg(&(samples_amount as cl_int))
        //     .set_arg(&(outputs_amount as cl_int))
        //     .set_global_work_sizes(&[samples_amount, outputs_amount])
        //     .enqueue_nd_range(queue)?
        //     .wait()?;

        let sample_losses_buffer = empty_buffer(samples_amount, CL_MEM_READ_WRITE, state)?;

        let compute_loss_kernel = program.get_krnl(COMPUTE_LOSS_KERNEL)?;

        ExecuteKernel::new(compute_loss_kernel)
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&sample_losses_buffer)
            .set_arg(&(outputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(queue)?;

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

        // let outputs_summation_per_sample = sum_outputs_per_sample(
        //     state, 
        //     output_samples, 
        //     outputs_amount, 
        //     samples_amount
        // )?;
        // let normalized_outputs = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

        let program = state.get_prgm(PROGRAM_NAME)?;

        // let normalize_outputs_kernel = program.get_krnl(NORMALIZE_OUTPUTS_KERNEL)?;

        // ExecuteKernel::new(normalize_outputs_kernel)
        //     .set_arg(output_samples)
        //     .set_arg(&outputs_summation_per_sample)
        //     .set_arg(&normalized_outputs)
        //     .set_arg(&(samples_amount as cl_int))
        //     .set_arg(&(outputs_amount as cl_int))
        //     .set_global_work_sizes(&[samples_amount, outputs_amount])
        //     .enqueue_nd_range(queue)?
        //     .wait()?;

        // queue.finish()?;

        let derivatives_buffer = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

        let loss_to_output_deriv_kernel =
            program.get_krnl(COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)?;

        ExecuteKernel::new(loss_to_output_deriv_kernel)
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&derivatives_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(outputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, outputs_amount])
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        Ok(derivatives_buffer)
    }
}

#[cfg(test)]
mod categorical_cross_entropy_tests {
    use std::ptr;

    use opencl3::{
        memory::{Buffer, CL_MEM_READ_ONLY},
        types::{cl_float, CL_NON_BLOCKING},
    };
    use rand::{thread_rng, Rng};

    use super::CategoricalCrossEntropy;
    use crate::utils::{approx_eq::assert_approx_equal_distance, setup_opencl, OpenCLState};
    use crate::{loss_functions::LossFunction, utils::opencl::DeviceType};

    #[test]
    fn should_compute_derivatives_up_to_a_certain_precision() {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU).unwrap();

        let context = &opencl_state.context;

        let mut gpu_loss = CategoricalCrossEntropy::new();
        gpu_loss.init(&opencl_state).unwrap();

        let outputs_amount: usize = 61;
        let samples_amount: usize = 113;
        let mut rng = rand::thread_rng();

        let output_samples: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();
        let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();

        let expected_derivatives: Vec<f32> = expected_outputs
            .iter()
            .zip(&output_samples)
            .map(|(expected_output, output)| {
                (expected_output, output.max(0.0000001).min(0.9999999))
            })
            .map(|(expected_output, output)| {
                -(expected_output / output
                  - (1.0 - expected_output) / (1.0 - output))
            })
            .collect();

        let mut outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )
        .unwrap();
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )
        .unwrap();

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                output_samples.as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();
        queue
            .enqueue_write_buffer(
                &mut expected_outputs_buf,
                CL_NON_BLOCKING,
                0,
                expected_outputs.as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        let buf = gpu_loss
            .compute_loss_derivative_with_respect_to_output_samples(
                &outputs_buf,
                &expected_outputs_buf,
                samples_amount,
            )
            .unwrap();
        let mut derivatives_vec = vec![0.0; samples_amount * outputs_amount];
        let derivatives_slice = derivatives_vec.as_mut_slice();

        queue
            .enqueue_read_buffer(&buf, CL_NON_BLOCKING, 0, derivatives_slice, &[])
            .unwrap()
            .wait()
            .unwrap();

        assert_approx_equal_distance(&expected_derivatives, &derivatives_vec, 0.01);
    }

    #[test]
    fn should_compute_loss_up_to_a_certain_precision() {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU).unwrap();
        let context = &opencl_state.context;

        let mut loss = CategoricalCrossEntropy::new();
        loss.init(&opencl_state).unwrap();

        let mut rng = thread_rng();
        let samples_amount = 1000;
        let outputs_amount = 290;
        let outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();
        let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();

        let expected_loss: f32 = expected_outputs
            .iter()
            .zip(&outputs)
            .map(|(expected_output, output)| {
                (expected_output, output.max(0.0000001).min(0.9999999))
            })
            .map(|(expected_output, output)| {
                -(expected_output * output.ln()
                    + (1.0 - expected_output) * (1.0 - output).ln())
            })
            .sum::<f32>()
            / samples_amount as f32;
        let mut outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )
        .unwrap();
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )
        .unwrap();

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                outputs.as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();
        queue
            .enqueue_write_buffer(
                &mut expected_outputs_buf,
                CL_NON_BLOCKING,
                0,
                expected_outputs.as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        let actual_loss = loss
            .compute_loss(&outputs_buf, &expected_outputs_buf, samples_amount)
            .unwrap();

        let largest_loss = expected_loss.max(actual_loss);
        println!(
            "|({} - {}) / {}| <= 0.1%",
            expected_loss, actual_loss, largest_loss
        );
        assert!((expected_loss - actual_loss).abs() / largest_loss <= 0.001);
    }
}