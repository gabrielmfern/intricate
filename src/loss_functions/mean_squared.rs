#[allow(unused_imports)]
use opencl3::{
    command_queue::{CommandQueue, CL_BLOCKING, CL_NON_BLOCKING},
    context::Context,
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::mem;
use std::ptr;

use crate::loss_functions::LossFunction;
use crate::utils::{
    opencl::compile_buffer_summation_kernel, vector_operations::VectorOperations, GpuSummable,
};
#[allow(unused_imports)]
use crate::utils::{setup_opencl, OpenCLState};

use super::OpenCLLossFunction;

#[derive(Debug)]
/// The Mean Squared loss function, good for some problem with
/// linear regression, because this error is quite free, in comparison
/// to the Categorical Cross Entropy loss function which restricts things
/// to be from 1.0 to 0.0 to work well
pub struct MeanSquared;

const PROGRAM_SOURCE: &str = include_str!("kernels/mean_squared.cl");
const COMPUTE_LOSS_KERNEL: &str = "compute_loss";
const COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL: &str = "compute_loss_to_output_derivatives";

#[derive(Debug)]
/// The Mean Squared loss function but computed with OpenCL,
/// good for some problem with linear regression,
/// because this error is quite free, in comparison
/// to the Categorical Cross Entropy loss function which restricts things
/// to be from 1.0 to 0.0 to work well
pub struct OpenCLMeanSquared<'a> {
    opencl_context: Option<&'a Context>,
    oepncl_queue: Option<&'a CommandQueue>,
    opencl_program: Option<Program>,
    opencl_compute_loss_kernel: Option<Kernel>,
    opencl_compute_loss_to_output_derivatives_kernel: Option<Kernel>,
    opencl_sum_buffer_program: Option<Program>,
    opencl_sum_buffer_kernel: Option<Kernel>,
}

impl<'a> OpenCLMeanSquared<'a> {
    pub fn new() -> OpenCLMeanSquared<'a> {
        OpenCLMeanSquared {
            opencl_context: None,
            oepncl_queue: None,
            opencl_program: None,
            opencl_compute_loss_kernel: None,
            opencl_sum_buffer_kernel: None,
            opencl_sum_buffer_program: None,
            opencl_compute_loss_to_output_derivatives_kernel: None,
        }
    }
}

impl<'a> OpenCLLossFunction<'a> for OpenCLMeanSquared<'a> {
    fn init(&mut self, context: &'a Context, queue: &'a CommandQueue) -> Result<(), ClError> {
        let program_compilation_result =
            Program::create_and_build_from_source(context, PROGRAM_SOURCE, "");
        if program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the mean_squared.cl Program:\n{:?}",
                program_compilation_result.err().unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }

        let program = program_compilation_result.unwrap();

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
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(queue)?
            .wait()?;

        // Ok(0.0)
        Ok(sample_losses_buffer.sum(
            context,
            queue,
            self.opencl_sum_buffer_kernel.as_ref().unwrap(),
        )? / outputs_amount as f32 / samples_amount as f32)
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
            .set_global_work_sizes(&[samples_amount, outputs_amount])
            .enqueue_nd_range(self.oepncl_queue.unwrap())?
            .wait()?;

        Ok(derivatives_buffer)
    }
}

impl LossFunction for MeanSquared {
    fn compute_loss(&self, outputs: &Vec<f32>, expected_outputs: &Vec<f32>) -> f32 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());

        expected_outputs
            .subtract(outputs)
            .powf(2.0)
            .par_iter()
            .sum::<f32>()
            / outputs_amount as f32
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f32,
        expected_output: f32,
    ) -> f32 {
        2.0 / ouputs_amount as f32 * (output - expected_output)
    }
}

#[test]
fn opencl_mean_squared_computation_of_loss_derivatives_should_be_the_same_as_normal_mean_squred(
) -> Result<(), ClError> {
    let normal_loss = MeanSquared;

    let opencl_state: OpenCLState = setup_opencl()?;

    let mut gpu_loss = OpenCLMeanSquared::new();
    gpu_loss.init(&opencl_state.context, &opencl_state.queue)?;

    let expected_derivative = normal_loss.compute_loss_derivative_with_respect_to_output(5, 0.5, 0.1);
    let mut outputs_buf =
        Buffer::<cl_float>::create(&opencl_state.context, CL_MEM_READ_ONLY, 5, ptr::null_mut())?;
    let mut expected_outputs_buf =
        Buffer::<cl_float>::create(&opencl_state.context, CL_MEM_READ_ONLY, 5, ptr::null_mut())?;

    opencl_state
        .queue
        .enqueue_write_buffer(
            &mut outputs_buf,
            CL_NON_BLOCKING,
            0,
            &[0.5, 0.0, 0.0, 0.0, 0.0],
            &[],
        )?
        .wait()?;
    opencl_state
        .queue
        .enqueue_write_buffer(
            &mut expected_outputs_buf,
            CL_NON_BLOCKING,
            0,
            &[0.1, 0.0, 0.0, 0.0, 0.0],
            &[],
        )?
        .wait()?;

    let buf = gpu_loss.compute_loss_derivative_with_respect_to_output_samples(&outputs_buf, &expected_outputs_buf, 1)?;
    let mut slice = [0.0, 0.0, 0.0, 0.0, 0.0];

    opencl_state.queue.enqueue_read_buffer(&buf, CL_NON_BLOCKING, 0, &mut slice, &[])?.wait()?;

    println!("{} - {} <= 0.1", slice[0], expected_derivative);
    assert!((slice[0] - expected_derivative).abs() <= 0.05);

    Ok(())
}

#[test]
fn opencl_mean_squared_computation_of_loss_should_return_same_value_as_normal_mean_squared(
) -> Result<(), ClError> {
    let normal_loss = MeanSquared;

    let opencl_state: OpenCLState = setup_opencl()?;

    let mut gpu_loss = OpenCLMeanSquared::new();
    gpu_loss.init(&opencl_state.context, &opencl_state.queue)?;

    let outputs = Vec::from([0.4, 0.9, 15.3, 19.3, 10.1]);
    let expected_outputs = Vec::from([0.0, 0.0, 0.0, 0.0, 1.0]);

    let expected_loss = normal_loss.compute_loss(&outputs, &expected_outputs);
    let mut outputs_buf =
        Buffer::<cl_float>::create(&opencl_state.context, CL_MEM_READ_ONLY, 5, ptr::null_mut())?;
    let mut expected_outputs_buf =
        Buffer::<cl_float>::create(&opencl_state.context, CL_MEM_READ_ONLY, 5, ptr::null_mut())?;

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

    let actual_loss = gpu_loss.compute_loss(&outputs_buf, &expected_outputs_buf, 1)?;

    assert!((expected_loss - actual_loss).abs() <= 0.1);

    Ok(())
}
