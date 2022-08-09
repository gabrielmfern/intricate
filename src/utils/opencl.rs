use std::{mem, ptr};

use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::Device,
    error_codes::ClError,
    kernel::{Kernel, ExecuteKernel},
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    program::Program,
    types::cl_float,
};
use super::gcd;

const SUM_PROGRAM_SOURCE: &str = include_str!("sum.cl");

pub fn compile_buffer_summation_kernel(context: &Context) -> Result<(Program, Kernel), ClError> {
    let program_compilation_result =
        Program::create_and_build_from_source(context, SUM_PROGRAM_SOURCE, "");
    if program_compilation_result.is_err() {
        println!(
            "A compilation error was found in the sum.cl Program:\n{:?}",
            program_compilation_result.err().unwrap()
        );
        println!("Please report this issue at https://github.com/gabrielmfern/intricate");
        panic!();
    }

    let program = program_compilation_result.unwrap();
    let kernel = Kernel::create(&program, "sum_all_values_in_workgroups")?;

    Ok((program, kernel))
}

pub trait GpuSummable
where
    Self: ClMem,
{
    fn sum(&self, context: &Context, queue: &CommandQueue, kernel: &Kernel) -> Result<f32, ClError>;

    fn reduce(
        &self,
        context: &Context,
        queue: &CommandQueue,
        max_local_size: usize,
        reduce_kernel: &Kernel,
    ) -> Result<Buffer<cl_float>, ClError>;
}

impl GpuSummable for Buffer<cl_float> {
    fn sum(&self, context: &Context, queue: &CommandQueue, kernel: &Kernel) -> Result<f32, ClError> {
        let device = Device::new(queue.device()?);
        let max_local_size = device.max_work_group_size()?;

        let mut current_count = self.size()? / mem::size_of::<cl_float>();

        if current_count == 1 {
            let mut buf_slice: [f32; 1] = [0.0];

            queue
                .enqueue_read_buffer(self, CL_NON_BLOCKING, 0, &mut buf_slice, &[])?
                .wait()?;

            Ok(buf_slice[0])
        } else if current_count == 0 {
            Ok(0.0)
        } else {
            let mut current_buf = self.reduce(context, queue, max_local_size, &kernel)?;
            current_count = current_buf.size()? / mem::size_of::<cl_float>();

            while current_count > 1 {
                current_buf = current_buf.reduce(context, queue, max_local_size, &kernel)?;
                current_count = current_buf.size()? / mem::size_of::<cl_float>();
            }

            let mut buf_slice: [f32; 1] = [0.0];

            queue
                .enqueue_read_buffer(&current_buf, CL_NON_BLOCKING, 0, &mut buf_slice, &[])?
                .wait()?;

            Ok(buf_slice[0])
        }
    }

    fn reduce(
        &self,
        context: &Context,
        queue: &CommandQueue,
        max_local_size: usize,
        reduce_kernel: &Kernel,
    ) -> Result<Buffer<cl_float>, ClError> {
        let current_count = self.size()? / mem::size_of::<cl_float>();
        assert!(current_count >= 1);

        let local_size = gcd(current_count, max_local_size);

        let current_reduced_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            current_count / local_size,
            ptr::null_mut(),
        )?;

        ExecuteKernel::new(reduce_kernel)
            .set_arg(self)
            .set_arg(&current_reduced_buffer)
            .set_arg_local_buffer(local_size)
            .set_local_work_size(local_size)
            .set_global_work_size(current_count)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(current_reduced_buffer)
    }
}