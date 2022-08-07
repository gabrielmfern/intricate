use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::cl_float;
use opencl3::error_codes::{ClError, cl_int};
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, ClMem};
use opencl3::program::Program;
use std::mem;
use std::ptr;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::loss_functions::LossFunction;
use crate::utils::vector_operations::VectorOperations;

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
    opencl_compute_loss_to_output_derivatives_kernel: Option<Kernel>
}

impl<'a> OpenCLMeanSquared<'a> {
    #[allow(dead_code)]
    fn new() -> OpenCLMeanSquared<'a> {

        OpenCLMeanSquared { 
            opencl_context: None, 
            oepncl_queue: None, 
            opencl_program: None, 
            opencl_compute_loss_kernel: None, 
            opencl_compute_loss_to_output_derivatives_kernel: None 
        }
    }
}

impl<'a> OpenCLLossFunction<'a> for OpenCLMeanSquared<'a> {
    fn init(&mut self, context: &'a Context, queue: &'a CommandQueue) -> Result<(), ClError> {
        let program_compilation_result = Program::create_and_build_from_source(context, PROGRAM_SOURCE, "");
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
        let compute_loss_derivative_with_respect_to_output_kernel = Kernel::create(&program, COMPUTE_LOSS_TO_OUTPUT_DERIVATIVES_KERNEL)?;

        self.opencl_context = Some(context);
        self.opencl_program = Some(program);
        self.oepncl_queue = Some(queue);
        self.opencl_compute_loss_kernel = Some(compute_loss_kernel);
        self.opencl_compute_loss_to_output_derivatives_kernel = Some(compute_loss_derivative_with_respect_to_output_kernel);

        Ok(())
    }

    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize
    ) -> Result<f32, ClError> {
        let sample_losses_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(), 
            CL_MEM_READ_WRITE, 
            samples_amount, 
            ptr::null_mut()
        )?;

        ExecuteKernel::new(self.opencl_compute_loss_kernel.as_ref().unwrap())
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&sample_losses_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(self.oepncl_queue.as_ref().unwrap())?
            .wait()?;

        let mut sample_losses = vec![0.0; samples_amount];
        let sample_losses_slice = sample_losses.as_mut_slice();
        self.oepncl_queue.as_ref().unwrap().enqueue_read_buffer(
            &sample_losses_buffer, 
            CL_BLOCKING, 
            0, 
            sample_losses_slice, 
            &[]
        )?.wait()?;

        Ok(sample_losses.par_iter().sum::<f32>())
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize
    ) -> Result<Buffer<cl_float>, ClError> {
        let outputs_amount = output_samples.size()? / samples_amount / mem::size_of::<cl_float>();
        let derivatives_buffer = Buffer::<cl_float>::create(
            self.opencl_context.as_ref().unwrap(), 
            CL_MEM_READ_WRITE, 
            output_samples.size()? / mem::size_of::<cl_float>(), 
            ptr::null_mut()
        )?;

        ExecuteKernel::new(self.opencl_compute_loss_to_output_derivatives_kernel.as_ref().unwrap())
            .set_arg(output_samples)
            .set_arg(expected_outputs)
            .set_arg(&derivatives_buffer)
            .set_global_work_sizes(&[samples_amount, outputs_amount])
            .enqueue_nd_range(self.oepncl_queue.as_ref().unwrap())?
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
        2.0 / ouputs_amount as f32 * (expected_output - output)
    }
}