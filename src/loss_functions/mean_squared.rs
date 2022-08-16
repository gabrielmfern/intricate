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
use crate::utils::{opencl::compile_buffer_summation_kernel, OpenCLSummable};

const PROGRAM_SOURCE: &str = include_str!("kernels/mean_squared.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";

#[derive(Debug)]
/// The Mean Squared loss function, good for some problem with
/// linear regression, because this error is quite free, in comparison
/// to the `Categorical Cross Entropy` loss function which restricts things
/// to be in (0, 1) (a **closed interval** between 0 and 1) to work well.
pub struct MeanSquared<'a> {
    opencl_context: Option<&'a Context>,
    oepncl_queue: Option<&'a CommandQueue>,
    opencl_program: Option<Program>,
    opencl_compute_loss_kernel: Option<Kernel>,
    opencl_compute_loss_to_output_derivatives_kernel: Option<Kernel>,
    opencl_sum_buffer_program: Option<Program>,
    opencl_sum_buffer_kernel: Option<Kernel>,
}

impl<'a> MeanSquared<'a> {
    pub fn new() -> ModelLossFunction<'a> {
        MeanSquared {
            opencl_context: None,
            oepncl_queue: None,
            opencl_program: None,
            opencl_compute_loss_kernel: None,
            // TODO: improve this as to not have these kernels lying around
            // where they should not
            // perhaps store the kernel and the program statically somehow?
            opencl_sum_buffer_kernel: None,
            opencl_sum_buffer_program: None,
            opencl_compute_loss_to_output_derivatives_kernel: None,
        }.into()
    }
}

impl<'a> LossFunction<'a> for MeanSquared<'a> {
    fn init(
        &mut self,
        context: &'a Context,
        queue: &'a CommandQueue,
    ) -> Result<(), CompilationOrOpenCLError> {
        let program =
            Program::create_and_build_from_source(context, PROGRAM_SOURCE, "")?;

        let compute_loss_kernel = Kernel::create(&program, COMPUTE_LOSS_KERNEL)?;
        let compute_loss_derivative_with_respect_to_output_kernel =
            Kernel::create(&program, COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)?;

        self.opencl_context = Some(context);
        self.opencl_program = Some(program);
        self.oepncl_queue = Some(queue);
        self.opencl_compute_loss_kernel = Some(compute_loss_kernel);
        self.opencl_compute_loss_to_output_derivatives_kernel =
            Some(compute_loss_derivative_with_respect_to_output_kernel);

        let sum_program_kernel = compile_buffer_summation_kernel(context)?;
        self.opencl_sum_buffer_program = Some(sum_program_kernel.0);
        self.opencl_sum_buffer_kernel = Some(sum_program_kernel.1);

        Ok(())
    }

    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<f32, ClError> {
        assert!(self.opencl_context.is_some());
        assert!(self.oepncl_queue.is_some());
        assert!(self.opencl_program.is_some());
        assert!(self.opencl_compute_loss_kernel.is_some());
        assert!(self
            .opencl_compute_loss_to_output_derivatives_kernel
            .is_some());
        assert!(self.opencl_sum_buffer_kernel.is_some());
        assert!(self.opencl_sum_buffer_program.is_some());
        assert_eq!(output_samples.size()?, expected_outputs.size()?);

        let context = self.opencl_context.unwrap();
        let queue = self.oepncl_queue.unwrap();

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();

        let sample_losses_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount,
            ptr::null_mut(),
        )?;

        ExecuteKernel::new(self.opencl_compute_loss_kernel.as_ref().unwrap())
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&sample_losses_buffer)
            .set_arg(&(outputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(queue)?
            .wait()?;

        // Ok(0.0)
        Ok(sample_losses_buffer.sum(
            context,
            queue,
            self.opencl_sum_buffer_kernel.as_ref().unwrap(),
        )? / outputs_amount as f32
            / samples_amount as f32)
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, ClError> {
        assert!(self.opencl_context.is_some());
        assert!(self.oepncl_queue.is_some());
        assert!(self.opencl_program.is_some());
        assert!(self.opencl_compute_loss_kernel.is_some());
        assert!(self
            .opencl_compute_loss_to_output_derivatives_kernel
            .is_some());

        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();
        let derivatives_buffer = Buffer::<cl_float>::create(
            self.opencl_context.as_ref().unwrap(),
            CL_MEM_READ_WRITE,
            output_samples.size()? / mem::size_of::<cl_float>(),
            ptr::null_mut(),
        )?;

        ExecuteKernel::new(
            self.opencl_compute_loss_to_output_derivatives_kernel
                .as_ref()
                .unwrap(),
        )
        .set_arg(output_samples)
        .set_arg(expected_outputs)
        .set_arg(&derivatives_buffer)
        .set_arg(&(samples_amount as cl_int))
        .set_arg(&(outputs_amount as cl_int))
        .set_global_work_sizes(&[samples_amount, outputs_amount])
        .enqueue_nd_range(self.oepncl_queue.unwrap())?
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
    use crate::{types::CompilationOrOpenCLError, loss_functions::LossFunction, utils::opencl::DeviceType};
    use crate::utils::{approx_eq::assert_approx_equal_distance, setup_opencl, OpenCLState};

    #[test]
    fn should_compute_derivatives_up_to_a_certain_precision() -> Result<(), CompilationOrOpenCLError> {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::CPU)?;

        let mut gpu_loss = MeanSquared::new();
        gpu_loss.init(&opencl_state.context, &opencl_state.queue)?;

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

        opencl_state
            .queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                output_samples.as_slice(),
                &[],
            )?
            .wait()?;
        opencl_state
            .queue
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

        opencl_state
            .queue
            .enqueue_read_buffer(&buf, CL_NON_BLOCKING, 0, derivatives_slice, &[])?
            .wait()?;

        assert_approx_equal_distance(&expected_derivatives, &derivatives_vec, 0.01);

        Ok(())
    }

    #[test]
    fn should_compute_loss_up_to_a_certain_precision() -> Result<(), CompilationOrOpenCLError> {
        let opencl_state: OpenCLState = setup_opencl(DeviceType::CPU)?;

        let mut loss = MeanSquared::new();
        loss.init(&opencl_state.context, &opencl_state.queue)?;

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

        opencl_state
            .queue
            .enqueue_write_buffer(
                &mut outputs_buf,
                CL_NON_BLOCKING,
                0,
                outputs.as_slice(),
                &[],
            )?
            .wait()?;
        opencl_state
            .queue
            .enqueue_write_buffer(
                &mut expected_outputs_buf,
                CL_NON_BLOCKING,
                0,
                expected_outputs.as_slice(),
                &[],
            )?
            .wait()?;

        let actual_loss = loss.compute_loss(&outputs_buf, &expected_outputs_buf, samples_amount)?;

        println!("|{} - {}| <= 0.5", expected_loss, actual_loss);
        assert!((expected_loss - actual_loss).abs() <= 0.5);

        Ok(())
    }
}