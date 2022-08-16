use std::{mem, ptr};

use crate::types::CompilationOrOpenCLError;

use super::gcd;
use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::{
        get_all_devices, Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL,
        CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_GPU,
    },
    error_codes::{cl_int, ClError},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    program::Program,
    types::cl_float,
};

const SUM_PROGRAM_SOURCE: &str = include_str!("sum.cl");

/// Compiles the summation program and builds it and then returns
/// a Ok contaning a tuple with both the Program and the Kernel respectively.
///
/// # Panics
///
/// Panics if the compilation or build process fails.
///
/// # Errors
///
/// This function will return an error if the kernel of summation
/// cannot be found inside th program.
pub fn compile_buffer_summation_kernel(
    context: &Context,
) -> Result<(Program, Kernel), CompilationOrOpenCLError> {
    let program = Program::create_and_build_from_source(context, SUM_PROGRAM_SOURCE, "")?;

    let kernel = Kernel::create(&program, "sum_all_values_in_workgroups")?;

    Ok((program, kernel))
}

/// A trait that is implemented within Intricate for defining if a certain struct
/// is summable using OpenCL and the kernel compiled with the **compile_buffer_summation_kernel**
/// function.
pub trait OpenCLSummable
where
    Self: ClMem,
{
    fn sum(&self, context: &Context, queue: &CommandQueue, kernel: &Kernel)
        -> Result<f32, ClError>;

    fn reduce(
        &self,
        context: &Context,
        queue: &CommandQueue,
        max_local_size: usize,
        reduce_kernel: &Kernel,
    ) -> Result<Buffer<cl_float>, ClError>;
}

impl OpenCLSummable for Buffer<cl_float> {
    /// Sums all of the numbers inside of a buffer and returns an Result enum
    /// containing either the resulting number or an OpenCL error.
    ///
    /// # Params
    ///
    /// - **kernel**: It is the kernel that is written for summing all of the values
    /// of a certain buffer in a workgroup. It can be compiled and built using the
    /// **compile_buffer_summation_kernel** function that is defined in this same module.
    ///
    /// - **context**: Is OpenCL's context over the application.
    ///
    /// - **queue**: The command queue attached to the device that will sum the buffer.
    ///
    /// # Errors
    ///
    /// This function will return an error if something wrong occurs inside of
    /// OpenCL.
    fn sum(
        &self,
        context: &Context,
        queue: &CommandQueue,
        kernel: &Kernel,
    ) -> Result<f32, ClError> {
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

            while dbg!(current_count) > 1 {
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

        let mut local_size = dbg!(gcd(current_count, max_local_size));
        if local_size == 1 && current_count < max_local_size {
            local_size = current_count;
        }

        if local_size == 1 {
            let middle = (current_count as f32).sqrt() as usize;
            for m in (middle..=current_count.min(max_local_size)).rev() {
                if current_count % m == 0 {
                    local_size = m;
                    break;
                }
            }
        }

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
            .set_arg(&(current_count as cl_int))
            .set_local_work_size(local_size)
            .set_global_work_size(current_count)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(current_reduced_buffer)
    }
}

#[derive(Debug)]
pub struct OpenCLState {
    pub context: Context,
    pub queue: CommandQueue,
    pub device: Device,
}

#[derive(Debug)]
pub enum UnableToSetupOpenCLError {
    OpenCL(ClError),
    NoDeviceFound,
}

impl From<ClError> for UnableToSetupOpenCLError {
    fn from(err: ClError) -> Self {
        UnableToSetupOpenCLError::OpenCL(err)
    }
}

#[derive(Debug)]
/// A enum used for telling Intricate what type of device it should try using with OpenCL.
pub enum DeviceType {
    /// Just the normal and usual **Graphics Processing Unit**
    GPU = CL_DEVICE_TYPE_GPU as isize,
    /// The **Central Processing Unit**
    CPU = CL_DEVICE_TYPE_CPU as isize,
    /// This will allow all types, and in turn, as of v0.3.0, will just get the first device
    /// it is able to find in your computer
    ALL = CL_DEVICE_TYPE_ALL as isize,
    CUSTOM = CL_DEVICE_TYPE_CUSTOM as isize,
    ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR as isize,
}

/// Gets the first device of a certain type it can find, starts the Context and the Command Queue
/// and then returns them all on a OpenCLState struct.
///
/// # Errors
///
/// Will return an error if OpenCL is unable to do something with the first device it finds,
/// or will return another type of error in case there is no available device.
pub fn setup_opencl(device_type: DeviceType) -> Result<OpenCLState, UnableToSetupOpenCLError> {
    let device_ids = get_all_devices(device_type as u64)?;
    if device_ids.len() > 0 {
        let first_gpu = Device::new(device_ids[0]);
        let context = Context::from_device(&first_gpu)?;
        // here it can be activated to make profiling on kernels
        let queue = CommandQueue::create_with_properties(&context, first_gpu.id(), 0, 0)?;

        Ok(OpenCLState {
            context,
            queue,
            device: first_gpu,
        })
    } else {
        Err(UnableToSetupOpenCLError::NoDeviceFound)
    }
}

#[cfg(test)]
mod test_gpu_summable {
    use opencl3::{
        command_queue::CL_NON_BLOCKING,
        device::cl_float,
        memory::{Buffer, CL_MEM_READ_WRITE},
    };
    use rand::{thread_rng, Rng};

    use super::{compile_buffer_summation_kernel, setup_opencl, DeviceType, OpenCLSummable};

    #[test]
    fn should_sum_buffer_to_correct_value() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();
        let (_program, kernel) = compile_buffer_summation_kernel(&opencl_state.context).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 276;
        let test_vec = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-41.34_f32..93_f32) })
            .collect::<Vec<f32>>();
        let expected_sum = test_vec.iter().sum::<f32>();

        let mut buff = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_WRITE,
            numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        opencl_state
            .queue
            .enqueue_write_buffer(&mut buff, CL_NON_BLOCKING, 0, test_vec.as_slice(), &[])
            .unwrap()
            .wait()
            .unwrap();

        let actual_result = buff
            .sum(&opencl_state.context, &opencl_state.queue, &kernel)
            .unwrap();

        assert!(dbg!((actual_result - expected_sum).abs()) <= 0.1);
    }
}
