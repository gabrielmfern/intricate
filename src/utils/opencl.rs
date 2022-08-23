//! A module with some utilities for dealing with OpenCL such as an **OpenCLState** that holds the
//! current programs and kernels that have been compiled.

use std::{collections::HashMap, mem, ptr};

use crate::{layers::compile_layers, loss_functions::compile_losses};

use super::gcd;
use intricate_macros::ErrorsEnum;
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

const BUFFER_OPERATIONS_PROGRAM_SOURCE: &str = include_str!("sum.cl");
const BUFFER_OPERATIONS_PROGRAM_NAME: &str = "SUM";
const REDUCE_BUFFER_KERNEL_NAME: &str = "sum_all_values_in_workgroups";

#[derive(Debug, ErrorsEnum)]
/// An error that happens in the `ensure_program` function, if either the compilation goes wrong of
/// the program or one of the kernels could not be found inside of the program being compiled.
#[allow(missing_docs)]
pub enum EnsureKernelsAndProgramError {
    OpenCL(ClError),
    Compilation(String),
}

/// Will compile all of the kernels listed in **kernel_names** inside of the
/// program with source **program_source**, with the options **compile_options**
/// and will insert that program as well with the kernels inside of the **opencl_state**
/// for later usage.
///
/// # Errors
///
/// - Will yield an error if the compilation goes wrong.
/// - Will yield an error if a specified kernel could not be found inside of the program's source.
pub(crate) fn ensure_program(
    opencl_state: &mut OpenCLState,
    program_name: String,
    program_source: String,
    compile_options: String,
    kernel_names: &[String],
) -> Result<(), EnsureKernelsAndProgramError> {
    let context = &opencl_state.context;

    if !opencl_state.programs.contains_key(&program_name) {
        let new_cl_program = Program::create_and_build_from_source(
            context,
            program_source.as_str(),
            &compile_options,
        )?;
        opencl_state.programs.insert(
            program_name.clone(),
            IntricateProgram {
                opencl_program: new_cl_program,
                kernels: HashMap::default(),
            },
        );
    }

    let program = opencl_state.programs.get_mut(&program_name).unwrap();

    for kernel_name in kernel_names.iter() {
        if !program.kernels.contains_key(kernel_name) {
            let kernel = Kernel::create(&program.opencl_program, kernel_name.as_str())?;
            program.kernels.insert(kernel_name.to_string(), kernel);
        }
    }

    Ok(())
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
pub(crate) fn find_optimal_local_and_global_work_sizes(
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

pub(crate) fn compile_buffer_operations_program(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    ensure_program(
        opencl_state,
        BUFFER_OPERATIONS_PROGRAM_NAME.to_string(),
        BUFFER_OPERATIONS_PROGRAM_SOURCE.to_string(),
        "".to_string(),
        &[REDUCE_BUFFER_KERNEL_NAME.to_string()],
    )
}

#[derive(Debug, ErrorsEnum)]
/// All of the possible errors that may happen while trying to run any buffer operation on a
/// certain buffer
pub enum BufferOperationError {
    /// Just a plain old OpenCL C error
    OpenCLError(ClError),
    /// This means that the program for the buffer operations
    /// has not yet been compiled because it could not be found
    ProgramNotFoundError,
    /// This means that the Kernel (OpenCL's shader) for the operation in question was not found,
    /// that may mean there is a problem in Intricate's code, so you should report this as an
    /// issue.
    KernelNotFoundError,
    /// This just means that the operation did find any device for it to run on.
    NoDeviceFoundError,
    /// This means that there is no command queue associated with the device, this may be a problem
    /// in Intricate's source code, so please report this in an issue.
    NoCommandQueueFoundError,
}

/// A trait that is implemented within Intricate for defining if a certain struct
/// is summable using OpenCL and the kernel compiled with the **compile_buffer_summation_kernel**
/// function.
pub trait BufferOperations
where
    Self: ClMem,
{
    /// Sums all of the numbers inside of a buffer and returns an Result enum
    /// containing either the resulting number or an OpenCL error.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the summation kernel was not foudn in the program for buffer operations.
    fn sum(&self, opencl_state: &OpenCLState) -> Result<f32, BufferOperationError>;
}

impl BufferOperations for Buffer<cl_float> {
    fn sum(&self, opencl_state: &OpenCLState) -> Result<f32, BufferOperationError> {
        if opencl_state.devices.is_empty() {
            return Err(BufferOperationError::NoDeviceFoundError);
        }

        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let device = opencl_state.devices.first().unwrap();
        let queue = opencl_state.queues.first().unwrap();

        let operations_program;
        if opencl_state.programs.contains_key(BUFFER_OPERATIONS_PROGRAM_NAME) {
            operations_program = opencl_state.programs.get(BUFFER_OPERATIONS_PROGRAM_NAME).unwrap();
        } else {
            return Err(BufferOperationError::ProgramNotFoundError);
        }

        let reduce_kernel;
        if operations_program.kernels.contains_key(REDUCE_BUFFER_KERNEL_NAME) {
            reduce_kernel = operations_program.kernels.get(REDUCE_BUFFER_KERNEL_NAME).unwrap();
        } else {
            return Err(BufferOperationError::KernelNotFoundError);
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
/// Just a struct that contains both the original OpenCL program and a HashMap containing all of
/// the kernels with their keys as just the names of the kernels and the values as the actual
/// kernels.
pub struct IntricateProgram {
    /// The original OpenCL Program struct.
    pub opencl_program: Program,
    /// A HashMap with the keys as the names of the kernels and the values as the original OpenCL
    /// structs that represent the kernels.
    pub kernels: HashMap<String, Kernel>,
}

#[derive(Debug)]
/// The state that contains useful OpenCL information that is necessary to keep track of the
/// compilled OpenCL programs and kernels.
pub struct OpenCLState {
    /// OpenCL's Context object that contains some useful information
    pub context: Context,
    /// A vec containing the corresponding Command Queue's for each one of the devices
    pub queues: Vec<CommandQueue>,
    /// A vec containing all of the devices that were found by OpenCL for a ceratin **DeviceType**
    pub devices: Vec<Device>,
    /// A HashMap where the key is the name of the program and value is a struct that contains both
    /// the original OpenCL program and another HashMap with all of the kernels.
    pub programs: HashMap<String, IntricateProgram>,
}

#[derive(Debug, ErrorsEnum)]
/// An error that happens when the `setup_opencl` function fails.
#[allow(missing_docs)]
pub enum UnableToSetupOpenCLError {
    OpenCL(ClError),
    CompilationErrors(EnsureKernelsAndProgramError),
    NoDeviceFound,
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
    /// A custom device, you can write custom drivers and use them in here if you have them in your
    /// computer.
    CUSTOM = CL_DEVICE_TYPE_CUSTOM as isize,
    #[allow(missing_docs)]
    ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR as isize,
}

/// Finds all of the devices of a certain **device_type**, starts a context for all of the devices,
/// and creates a CommandQueue for each one of the devices.
/// Also will compile some basic Intricate programs after setting up.
///
/// # Errors
///
/// Will return an NoDeviceFound error if it could not find any device of the specified type, or
/// will return an ClError with the respective OpenCL error code if something goes wrong while
/// creating the context or the queues.
pub fn setup_opencl(device_type: DeviceType) -> Result<OpenCLState, UnableToSetupOpenCLError> {
    let device_ids = get_all_devices(device_type as cl_device_type)?;
    if !device_ids.is_empty() {
        let devices: Vec<Device> = device_ids.iter().map(|id| Device::new(*id)).collect();
        let context = Context::from_devices(&device_ids, &[], None, ptr::null_mut())?;

        // here it can be activated to make profiling on kernels
        let queues: Vec<CommandQueue> = devices
            .iter()
            .map(|device| CommandQueue::create_with_properties(&context, device.id(), 0, 0))
            .collect::<Result<Vec<CommandQueue>, ClError>>()?;

        let mut state = OpenCLState {
            context,
            queues,
            devices,
            programs: HashMap::default(),
        };

        compile_buffer_operations_program(&mut state)?;

        compile_layers(&mut state)?;

        compile_losses(&mut state)?;

        Ok(state)
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

    use super::{setup_opencl, BufferOperations, DeviceType};

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

        let actual_result = buff.sum(&opencl_state).unwrap();

        assert!(
            ((actual_result - expected_sum) / (actual_result.max(expected_sum))).abs() <= 0.0001
        );
    }
}