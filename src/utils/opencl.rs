use std::{collections::HashMap, mem, ptr};

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
    types::{cl_device_type, cl_float},
};

const SUM_PROGRAM_SOURCE: &str = include_str!("sum.cl");
const SUM_PROGRAM_NAME: &str = "SUM";
const SUM_KERNEL_NAME: &str = "sum_all_values_in_workgroups";

/// A trait that is implemented within Intricate for defining if a certain struct
/// is summable using OpenCL and the kernel compiled with the **compile_buffer_summation_kernel**
/// function.
pub trait BufferOperations
where
    Self: ClMem,
{
    fn sum(&self, opencl_state: &mut OpenCLState) -> Result<f32, BufferOperationError>;
}

/// Tries to find the optimal local and global work size as to use as much
/// of the device's computational power.
///
/// - **data_size**: The size of the data that will be computed in the end
/// - **max_Local_size**: The max local work size of the device that the sizes are going to be used
/// in
///
/// Be aware that in some cases like data_sizes that are prime numbers there will be a need to have
/// larger global sizes than the data_size to make it divide or be divisble by the max_local_size
pub fn find_optimal_local_and_global_work_sizes(
    data_size: usize,
    max_local_size: usize,
) -> (usize, usize) {
    let mut local_size = gcd(data_size, max_local_size);
    if local_size == 1 && data_size < max_local_size {
        local_size = data_size;
    }

    if local_size == 1 {
        let middle = (data_size as f32).sqrt() as usize;
        for m in (middle..=data_size.min(max_local_size)).rev() {
            if data_size % m == 0 {
                local_size = m;
                break;
            }
        }
    }

    let global_size: usize;

    if local_size == 1 {
        let mut temp_size = data_size + 1;
        let mut temp_local_size = gcd(temp_size, max_local_size);
        while temp_local_size == 1 {
            temp_size += 1;
            temp_local_size = gcd(temp_size, max_local_size);
        }
        global_size = temp_size;
        local_size = temp_local_size;
    } else {
        global_size = data_size;
    }

    (local_size, global_size)
}

#[derive(Debug)]
pub enum BufferOperationError {
    OpenCLError(ClError),
    CompilationError(String),
    NoDeviceFoundError,
    NoCommandQueueFoundError,
}

impl From<ClError> for BufferOperationError {
    fn from(err: ClError) -> Self {
        BufferOperationError::OpenCLError(err)
    }
}

impl From<String> for BufferOperationError {
    fn from(err: String) -> Self {
        BufferOperationError::CompilationError(err)
    }
}

fn reduce_buffer_by_summation(
    buffer: &Buffer<cl_float>,
    context: &Context,
    queue: &CommandQueue,
    max_local_size: usize,
    reduce_kernel: &Kernel,
) -> Result<Buffer<cl_float>, ClError> {
    let current_count = buffer.size()? / mem::size_of::<cl_float>();
    assert!(current_count >= 1);

    let (local_size, global_size) =
        find_optimal_local_and_global_work_sizes(current_count, max_local_size);
    dbg!(local_size);
    dbg!(global_size);

    let current_reduced_buffer = Buffer::<cl_float>::create(
        context,
        CL_MEM_READ_WRITE,
        global_size / local_size,
        ptr::null_mut(),
    )?;

    ExecuteKernel::new(reduce_kernel)
        .set_arg(buffer)
        .set_arg(&current_reduced_buffer)
        .set_arg_local_buffer(local_size)
        .set_arg(&(current_count as cl_int))
        .set_local_work_size(local_size)
        .set_global_work_size(global_size)
        .enqueue_nd_range(queue)?
        .wait()?;

    Ok(current_reduced_buffer)
}

impl BufferOperations for Buffer<cl_float> {
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
    fn sum(&self, opencl_state: &mut OpenCLState) -> Result<f32, BufferOperationError> {
        if opencl_state.devices.is_empty() {
            return Err(BufferOperationError::NoDeviceFoundError);
        }

        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let device = opencl_state.devices.first().unwrap();
        let queue = opencl_state.queues.first().unwrap();

        let mut reduce_program_option = opencl_state.programs.get(&SUM_PROGRAM_NAME.to_string());
        let reduce_program;
        let reduce_kernel;
        if reduce_program_option.is_some() {
            reduce_program = reduce_program_option.unwrap();
            reduce_kernel = reduce_program
                .kernels
                .get(&SUM_KERNEL_NAME.to_string())
                .unwrap();
        } else {
            let cl_program = Program::create_and_build_from_source(
                &opencl_state.context,
                SUM_PROGRAM_SOURCE,
                "",
            )?;
            let cl_kernel = Kernel::create(&cl_program, SUM_KERNEL_NAME)?;

            let temp_map = HashMap::new();
            temp_map.insert(SUM_KERNEL_NAME.to_string(), cl_kernel);

            opencl_state.programs.insert(
                SUM_PROGRAM_NAME.to_string(),
                IntricateProgram {
                    opencl_program: cl_program,
                    kernels: temp_map,
                },
            );
            reduce_program = opencl_state
                .programs
                .get(&SUM_PROGRAM_NAME.to_string())
                .unwrap();
            reduce_kernel = &cl_kernel;
        }

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
            let context = &opencl_state.context;
            let mut current_buf =
                reduce_buffer_by_summation(self, context, queue, max_local_size, reduce_kernel)?;
            current_count = current_buf.size()? / mem::size_of::<cl_float>();

            while current_count > 1 {
                current_buf = reduce_buffer_by_summation(
                    &current_buf,
                    context,
                    queue,
                    max_local_size,
                    reduce_kernel,
                )?;
                current_count = current_buf.size()? / mem::size_of::<cl_float>();
            }

            let mut buf_slice: [f32; 1] = [0.0];

            queue
                .enqueue_read_buffer(&current_buf, CL_NON_BLOCKING, 0, &mut buf_slice, &[])?
                .wait()?;

            Ok(buf_slice[0])
        }
    }
}

#[derive(Debug)]
pub struct IntricateProgram {
    pub opencl_program: Program,
    pub kernels: HashMap<String, Kernel>,
}

#[derive(Debug)]
pub struct OpenCLState {
    pub context: Context,
    pub queues: Vec<CommandQueue>,
    pub devices: Vec<Device>,
    pub programs: HashMap<String, IntricateProgram>,
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

/// Finds all of the devices of a certain **device_type**, starts a context for all of the devices,
/// and creates a CommandQueue for each one of the devices.
///
/// # Errors
///
/// Will return an NoDeviceFound error if it could not find any device of the specified type, or
/// will return an ClError with the respective OpenCL error code if something goes wrong while
/// creating the context or the queues.
pub fn setup_opencl(device_type: DeviceType) -> Result<OpenCLState, UnableToSetupOpenCLError> {
    let device_ids = get_all_devices(device_type as cl_device_type)?;
    if device_ids.is_empty() {
        let devices: Vec<Device> = device_ids.into_iter().map(|id| Device::new(id)).collect();
        let context = Context::from_devices(&device_ids, &[], None, ptr::null_mut())?;

        // here it can be activated to make profiling on kernels
        let queues: Vec<CommandQueue> = devices
            .iter()
            .map(|device| CommandQueue::create_with_properties(&context, device.id(), 0, 0))
            .collect::<Result<Vec<CommandQueue>, ClError>>()?;

        Ok(OpenCLState {
            context,
            queues,
            devices,
            programs: HashMap::default(),
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
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    use super::{setup_opencl, DeviceType, BufferOperations};

    #[test]
    fn should_sum_buffer_to_correct_value() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 1234;
        let test_vec: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-123.31_f32..3193.31_f32) })
            .collect();
        let expected_sum: f32 = test_vec.par_iter().sum();

        let mut buff = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_WRITE,
            numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        let first_device_queue = opencl_state.queues.first().unwrap();

        first_device_queue
            .enqueue_write_buffer(&mut buff, CL_NON_BLOCKING, 0, test_vec.as_slice(), &[])
            .unwrap()
            .wait()
            .unwrap();

        let actual_result = buff
            .sum(&mut opencl_state)
            .unwrap();

        assert!(
            ((actual_result - expected_sum) / (actual_result.max(expected_sum))).abs() <= 0.0001
        );
    }
}