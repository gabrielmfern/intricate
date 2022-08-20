use std::collections::HashMap;
use std::mem;
use std::ptr;

use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    program::Program,
};

use crate::loss_functions::LossFunction;
use crate::types::CompilationOrOpenCLError;
use crate::types::ModelLossFunction;
use crate::utils::opencl::IntricateProgram;
use crate::utils::BufferOperations;
use crate::utils::OpenCLState;

const PROGRAM_NAME: &str = "MEAN_SQUARED";
const PROGRAM_SOURCE: &str = include_str!("kernels/mean_squared.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";

#[derive(Debug)]
/// The Mean Squared loss function, good for some problem with
/// linear regression, because this error is quite free, in comparison
/// to the `Categorical Cross Entropy` loss function which restricts things
/// to be in (0, 1) (a **closed interval** between 0 and 1) to work well.
pub struct MeanSquared<'a> {
    opencl_state: Option<&'a mut OpenCLState>,
}

impl<'a> MeanSquared<'a> {
    pub fn new() -> ModelLossFunction<'a> {
        Self::new_raw().into()
    }

    pub fn new_raw() -> MeanSquared<'a> {
        MeanSquared { opencl_state: None }
    }
}

impl<'a> LossFunction<'a> for MeanSquared<'a> {
    fn init(&mut self, opencl_state: &'a mut OpenCLState) -> Result<(), CompilationOrOpenCLError> {
        if !opencl_state
            .programs
            .contains_key(&PROGRAM_NAME.to_string())
        {
            let cl_program =
                Program::create_and_build_from_source(&opencl_state.context, PROGRAM_SOURCE, "")?;
            opencl_state.programs.insert(
                PROGRAM_NAME.to_string(),
                IntricateProgram {
                    opencl_program: cl_program,
                    kernels: HashMap::default(),
                },
            );
        }

        let program = opencl_state
            .programs
            .get_mut(&PROGRAM_NAME.to_string())
            .unwrap();

        if !program
            .kernels
            .contains_key(&COMPUTE_LOSS_KERNEL.to_string())
        {
            let compute_loss_kernel = Kernel::create(&program.opencl_program, COMPUTE_LOSS_KERNEL)?;
            program
                .kernels
                .insert(COMPUTE_LOSS_KERNEL.to_string(), compute_loss_kernel);
        }

        if !program
            .kernels
            .contains_key(&COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL.to_string())
        {
            let compute_loss_derivative_with_respect_to_output_kernel = Kernel::create(
                &program.opencl_program,
                COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL,
            )?;
            program.kernels.insert(
                COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL.to_string(),
                compute_loss_derivative_with_respect_to_output_kernel,
            );
        }

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
        let context = state.context;
        let queue = state.queues.first().unwrap();

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();

        let sample_losses_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_WRITE,
            samples_amount,
            ptr::null_mut(),
        )?;

        // TODO: treat this error cases
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
            .unwrap() // TODO: treat this BufferOperationError instead of unwraping it here
            / outputs_amount as f32
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

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();
        let derivatives_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            output_samples.size()? / mem::size_of::<cl_float>(),
            ptr::null_mut(),
        )?;

        // TODO: treat this error cases
        let compute_loss_to_output_derivatives_kernel = state
            .programs
            .get(PROGRAM_NAME)
            .unwrap()
            .kernels
            .get(COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)
            .unwrap();

        ExecuteKernel::new(&compute_loss_to_output_derivatives_kernel)
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&derivatives_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(outputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, outputs_amount])
            .enqueue_nd_range(state.queues.first().unwrap())?
            .wait()?;

        Ok(derivatives_buffer)
    }
}

#[cfg(test)]
mod mean_squared_tests {
    use std::ptr;

    use opencl3::{
        memory::{Buffer, CL_MEM_READ_ONLY},
        types::{cl_float, CL_NON_BLOCKING},
    };
    use rand::{thread_rng, Rng};

    use super::MeanSquared;
    use crate::utils::{approx_eq::assert_approx_equal_distance, setup_opencl, OpenCLState};
    use crate::{
        loss_functions::LossFunction, types::CompilationOrOpenCLError, utils::opencl::DeviceType,
    };

    #[test]
    fn should_compute_derivatives_up_to_a_certain_precision() -> Result<(), CompilationOrOpenCLError>
    {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU)?;

        let mut gpu_loss = MeanSquared::new_raw();
        gpu_loss.init(&mut opencl_state)?;

        let outputs_amount: usize = 61;
        let samples_amount: usize = 113;
        let mut rng = rand::thread_rng();

        let output_samples: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(-13123.0_f32..15413_f32))
            .collect();
        let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(-13123.0_f32..15413_f32))
            .collect();

        let expected_derivatives: Vec<f32> = expected_outputs
            .iter()
            .zip(&output_samples)
            .map(|(expected_output, actual_output)| {
                2.0 / outputs_amount as f32 * (actual_output - expected_output)
                // normal_loss.compute_loss_derivative_with_respect_to_output(
                //     outputs_amount,
                //     *actual_output,
                //     *expected_output,
                // )
            })
            .collect();

        let mut outputs_buf = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            &opencl_state.context,
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

        let mut loss = MeanSquared::new();
        loss.init(&mut opencl_state)?;

        let mut rng = thread_rng();
        let samples_amount = 27;
        let outputs_amount = 29;
        let outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1241_f32..2192_f32))
            .collect();
        let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1241_f32..2192_f32))
            .collect();

        let expected_loss: f32 = expected_outputs
            .iter()
            .zip(&outputs)
            .map(|(output, expected_output)| (output - expected_output).powf(2.0))
            .sum::<f32>()
            / outputs_amount as f32
            / samples_amount as f32;
        let mut outputs_buf = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * outputs_amount,
            ptr::null_mut(),
        )?;
        let mut expected_outputs_buf = Buffer::<cl_float>::create(
            &opencl_state.context,
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

        println!(
            "|({} - {}) / {}| <= 0.1%",
            expected_loss,
            actual_loss,
            expected_loss.max(actual_loss)
        );
        assert!((expected_loss - actual_loss).abs() / expected_loss.max(actual_loss) <= 0.001);

        Ok(())
    }
}
