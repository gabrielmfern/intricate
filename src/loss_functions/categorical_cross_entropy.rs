use std::mem;
use std::ptr;

use opencl3::{
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};

use crate::loss_functions::LossFunction;
use crate::types::ModelLossFunction;
use crate::utils::opencl::EnsureKernelsAndProgramError;
use crate::utils::opencl::ensure_program;
use crate::utils::BufferOperations;
use crate::utils::OpenCLState;

const PROGRAM_NAME: &str = "CATEGORICAL_CROSS_ENTROPY";
const PROGRAM_SOURCE: &str = include_str!("kernels/categorical_cross_entropy.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";

pub(crate) fn compile_categorical_cross_entropy(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        COMPUTE_LOSS_KERNEL.to_string(),
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
pub struct CategoricalCrossEntropy<'a> {
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> CategoricalCrossEntropy<'a> {
    pub fn new() -> ModelLossFunction<'a> {
        Self::new_raw().into()
    }

    pub fn new_raw() -> CategoricalCrossEntropy<'a> {
        CategoricalCrossEntropy { opencl_state: None }
    }
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
    ) -> Result<f32, ClError> {
        assert!(self.opencl_state.is_some());
        assert!(!self.opencl_state.unwrap().queues.is_empty());
        assert_eq!(output_samples.size()?, expected_outputs.size()?);

        let state = self.opencl_state.unwrap();
        let context = &state.context;
        let queue = state.queues.first().unwrap();

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();

        let sample_losses_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            samples_amount,
            ptr::null_mut(),
        )?;

        let compute_loss_kernel = state
            .programs
            .get(PROGRAM_NAME)
            .unwrap()
            .kernels
            .get(COMPUTE_LOSS_KERNEL)
            .unwrap();

        ExecuteKernel::new(compute_loss_kernel)
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&sample_losses_buffer)
            .set_arg(&(outputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(queue)?
            .wait()?;

        // Ok(0.0)
        Ok(sample_losses_buffer
            .sum(self.opencl_state.unwrap())
            .unwrap()
            / samples_amount as f32)
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, ClError> {
        assert!(self.opencl_state.is_some());
        assert!(!self.opencl_state.unwrap().queues.is_empty());
        assert_eq!(output_samples.size()?, expected_outputs.size()?);

        let state = self.opencl_state.unwrap();
        let context = &state.context;
        let queue = state.queues.first().unwrap();

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();
        let derivatives_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_WRITE,
            output_samples.size()? / mem::size_of::<cl_float>(),
            ptr::null_mut(),
        )?;

        let loss_to_output_deriv_kernel = state
            .programs
            .get(PROGRAM_NAME)
            .unwrap()
            .kernels
            .get(COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)
            .unwrap();

        ExecuteKernel::new(loss_to_output_deriv_kernel)
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&derivatives_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(outputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, outputs_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

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
    use crate::{
        loss_functions::LossFunction, types::CompilationOrOpenCLError, utils::opencl::DeviceType,
    };

    #[test]
    fn should_compute_derivatives_up_to_a_certain_precision() -> Result<(), CompilationOrOpenCLError>
    {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU)?;

        let context = &opencl_state.context;

        let mut gpu_loss = CategoricalCrossEntropy::new();
        gpu_loss.init(&opencl_state)?;

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
            .map(|(expected_output, actual_output)| -expected_output / actual_output)
            .collect();

        let mut outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                output_samples.as_slice(),
                &[],
            )?
            .wait()?;
        queue
            .enqueue_write_buffer(
                &mut expected_outputs_buf,
                CL_NON_BLOCKING,
                0,
                expected_outputs.as_slice(),
                &[],
            )?
            .wait()?;

        let buf = gpu_loss.compute_loss_derivative_with_respect_to_output_samples(
            &outputs_buf,
            &expected_outputs_buf,
            samples_amount,
        )?;
        let mut derivatives_vec = vec![0.0; samples_amount * outputs_amount];
        let derivatives_slice = derivatives_vec.as_mut_slice();

        queue
            .enqueue_read_buffer(&buf, CL_NON_BLOCKING, 0, derivatives_slice, &[])?
            .wait()?;

        assert_approx_equal_distance(&expected_derivatives, &derivatives_vec, 0.01);

        Ok(())
    }

    #[test]
    fn should_compute_loss_up_to_a_certain_precision() -> Result<(), CompilationOrOpenCLError> {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU)?;
        let context = &opencl_state.context;

        let mut loss = CategoricalCrossEntropy::new();
        loss.init(&opencl_state)?;

        let mut rng = thread_rng();
        let samples_amount = 1;
        let outputs_amount = 29;
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
            .map(|(expected_output, output)| -expected_output * output.ln())
            .sum::<f32>()
            / samples_amount as f32;
        let mut outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                outputs.as_slice(),
                &[],
            )?
            .wait()?;
        queue
            .enqueue_write_buffer(
                &mut expected_outputs_buf,
                CL_NON_BLOCKING,
                0,
                expected_outputs.as_slice(),
                &[],
            )?
            .wait()?;

        let actual_loss = loss.compute_loss(&outputs_buf, &expected_outputs_buf, samples_amount)?;

        let largest_loss = expected_loss.max(actual_loss);
        println!(
            "|({} - {}) / {}| <= 0.1%",
            expected_loss, actual_loss, largest_loss
        );
        assert!((expected_loss - actual_loss).abs() / largest_loss <= 0.001);

        Ok(())
    }
}