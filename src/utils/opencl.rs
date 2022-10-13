//! A module with some utilities for dealing with OpenCL such as an **OpenCLState** that holds the
//! current programs and kernels that have been compiled.

use std::{collections::HashMap, mem, ptr};

use crate::{
    layers::compile_layers,
    loss_functions::compile_losses,
    model::compile_model,
    types::{KernelNotFoundError, ProgramNotFoundError},
};

use super::gcd;
use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    command_queue::{CommandQueue, CL_BLOCKING, CL_NON_BLOCKING},
    context::Context,
    device::{
        get_all_devices, Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL,
        CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_GPU,
    },
    error_codes::{cl_int, ClError},
    event::Event,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_device_type, cl_event, cl_float, cl_mem_flags},
};

const BUFFER_OPERATIONS_PROGRAM_SOURCE: &str = include_str!("buffer_operations.cl");
const BUFFER_OPERATIONS_PROGRAM_NAME: &str = "BUFFER_OPERATIONS";

const REDUCE_BUFFER_KERNEL_NAME: &str = "sum_all_values_in_workgroups";

const SCALE_INPLACE_BUFFER_KERNEL_NAME: &str = "scale_inplace";
const SCALE_BUFFER_KERNEL_NAME: &str = "scale";

const INVERSE_SQRT_BUFFER_KERNEL_NAME: &str = "inverse_sqrt";
const INVERSE_SQRT_INPLACE_BUFFER_KERNEL_NAME: &str = "inverse_sqrt_inplace";

const ADD_BUFFER_KERNEL_NAME: &str = "add";
const ADD_NUM_BUFFER_KERNEL_NAME: &str = "add_num";
const ADD_INPLACE_BUFFER_KERNEL_NAME: &str = "add_inplace";
const SHIFT_INPLACE_BUFFER_KERNEL_NAME: &str = "shift_inplace";
const SUBTRACT_BUFFER_KERNEL_NAME: &str = "subtract";
const SUBTRACT_INPLACE_BUFFER_KERNEL_NAME: &str = "subtract_inplace";
const MULTIPLY_BUFFER_KERNEL_NAME: &str = "multiply";
const MULTIPLY_INPLACE_BUFFER_KERNEL_NAME: &str = "multiply_inplace";
const DIVIDE_BUFFER_KERNEL_NAME: &str = "divide";
const DIVIDE_INPLACE_BUFFER_KERNEL_NAME: &str = "divide_inplace";

#[derive(Debug, FromForAllUnnamedVariants)]
/// An error that happens in the `ensure_program` function, if either the compilation goes wrong of
/// the program or one of the kernels could not be found inside of the program being compiled.
#[allow(missing_docs)]
pub enum EnsureKernelsAndProgramError {
    OpenCL(ClError),
    /// An error that will occur when something goes wrong in kernel compilation
    /// returning a tuple with the error in the code itself and the name of the program
    /// in which it failed
    Compilation(String, String),
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
        let cl_program_result = Program::create_and_build_from_source(
            context,
            program_source.as_str(),
            &compile_options,
        );
        if let Ok(new_cl_program) = cl_program_result {
            opencl_state.programs.insert(
                program_name.clone(),
                IntricateProgram {
                    opencl_program: new_cl_program,
                    kernels: HashMap::default(),
                },
            );
        } else {
            return Err(EnsureKernelsAndProgramError::Compilation(
                cl_program_result.err().unwrap(),
                program_name,
            ));
        }
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
    if data_size <= max_local_size {
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
    opencl_state: &OpenCLState,
    max_local_size: usize,
    reduce_kernel: &Kernel,
    wait_list: &[Event],
) -> Result<(Event, Buffer<cl_float>), ClError> {
    let current_count = buffer.size()? / mem::size_of::<cl_float>();
    assert!(current_count >= 1);

    let (local_size, global_size) =
        find_optimal_local_and_global_work_sizes(current_count, max_local_size);

    let current_reduced_buffer =
        empty_buffer(global_size / local_size, CL_MEM_READ_WRITE, opencl_state)?;
    let queue = opencl_state.queues.first().unwrap();

    let event = ExecuteKernel::new(reduce_kernel)
        .set_arg(buffer)
        .set_arg(&current_reduced_buffer)
        .set_arg_local_buffer(local_size * mem::size_of::<cl_int>())
        .set_arg(&(current_count as cl_int))
        .set_event_wait_list(&wait_list.iter().map(|e| e.get()).collect::<Vec<cl_event>>())
        .set_local_work_size(local_size)
        .set_global_work_size(global_size)
        .enqueue_nd_range(queue)?;

    Ok((event, current_reduced_buffer))
}

pub(crate) fn compile_buffer_operations_program(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        REDUCE_BUFFER_KERNEL_NAME.to_string(),
        ADD_BUFFER_KERNEL_NAME.to_string(),
        SUBTRACT_BUFFER_KERNEL_NAME.to_string(),
        MULTIPLY_BUFFER_KERNEL_NAME.to_string(),
        DIVIDE_BUFFER_KERNEL_NAME.to_string(),
        SCALE_BUFFER_KERNEL_NAME.to_string(),
        ADD_NUM_BUFFER_KERNEL_NAME.to_string(),
        INVERSE_SQRT_BUFFER_KERNEL_NAME.to_string(),
        SCALE_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        SHIFT_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        INVERSE_SQRT_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        ADD_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        SUBTRACT_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        MULTIPLY_INPLACE_BUFFER_KERNEL_NAME.to_string(),
        DIVIDE_INPLACE_BUFFER_KERNEL_NAME.to_string(),
    ];

    ensure_program(
        opencl_state,
        BUFFER_OPERATIONS_PROGRAM_NAME.to_string(),
        BUFFER_OPERATIONS_PROGRAM_SOURCE.to_string(),
        "".to_string(),
        kernels,
    )
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// All of the possible errors that may happen while trying to run any buffer operation on a
/// certain buffer
pub enum BufferOperationError {
    /// Just a plain old OpenCL C error
    OpenCLError(ClError),
    /// This means that the program for the buffer operations
    /// has not yet been compiled because it could not be found
    ProgramNotFoundError(ProgramNotFoundError),
    /// This means that the Kernel (OpenCL's shader) for the operation in question was not found,
    /// that may mean there is a problem in Intricate's code, so you should report this as an
    /// issue.
    KernelNotFoundError(KernelNotFoundError),
    /// An error that happens when doing an operation that requires two buffers and that requires
    /// that both buffers are of the same size and count.
    BuffersAreNotOfSameSize(usize, usize),
    /// This just means that the operation did ot find any device for it to run on.
    NoDeviceFoundError,
    /// This means that there is no command queue associated with the device, this may be a problem
    /// in Intricate's source code, so please report this in an issue.
    NoCommandQueueFoundError,
}

/// A trait that is implemented within Intricate for doing inplace buffer operations that instead
/// of the normal buffer operations does not duplicate data, but mutates the original buffer.
pub trait InplaceBufferOperations
where
    Self: ClMem + Sized,
{
    /// Scales the buffer by a certain number or scaler.
    ///
    /// As an example, if you had a buffer with
    /// the number **[4, 5, 10]**, and you scaled it by **3** this method would change &self to
    /// **[12, 15, 30]**.
    fn scale_inplc(
        &mut self,
        scaler: f32,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Will add all of the values of the `other` buffer into self.
    fn add_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Will just subtract all of the numbers of the `other` buffer into self.
    fn subtract_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Adds a number to every single number inside of Self
    fn shift_inplc(
        &mut self,
        num: f32,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Takes the inverse sqrt of Self
    fn inverse_sqrt_inplc(
        &mut self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Will just multiply all of the numbers of the `other` buffer into self.
    fn multiply_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;

    /// Will divide all of the numbers of &self by the numbers of the `other` buffer.
    fn divide_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError>;
}

impl InplaceBufferOperations for Buffer<cl_float> {
    fn scale_inplc(
        &mut self,
        scaler: f32,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(SCALE_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&(scaler as cl_float))
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(())
    }

    fn inverse_sqrt_inplc(
        &mut self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(INVERSE_SQRT_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(())
    }

    fn shift_inplc(
        &mut self,
        num: f32,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(SHIFT_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&(num as cl_float))
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(())
    }

    fn add_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(ADD_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(())
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn subtract_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(SUBTRACT_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(())
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn divide_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(DIVIDE_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(())
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn multiply_inplc(
        &mut self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(MULTIPLY_INPLACE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(())
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }
}

/// A trait that is implemented within Intricate for doing buffer operations that somewhat of
/// duplicate data. An example of this is if you subtract a buffer from another it will not change
/// any of these two buffers, but it will create a new one with the results and give it back.
pub trait BufferOperations
where
    Self: ClMem + Sized,
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

    /// Scales the buffer by a certain number or scaler.
    ///
    /// As an example, if you had a buffer with
    /// the number **[4, 5, 10]**, and you scaled it by **3** this method would give you ``[12, 15,
    /// 30]`.
    fn scale(&self, scaler: f32, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

    /// Will just add all of the numbers of two buffers together into a new one.
    fn add(&self, other: &Self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

    /// Will just subtract all of the numbers from the current buffer to the other.
    fn subtract(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// Multiplies each respective number of the current buffer and another buffer.
    fn multiply(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// Adds a number to every single number inside of Self
    fn shift(&self, num: f32, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

    /// Takes the inverse sqrt of each one of the numbers
    fn inverse_sqrt(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

    /// Divides each respective number of the current buffer and another buffer.
    fn divide(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// A function that prints a Vec that contains the information of SElf
    fn dbg(&self, state: &OpenCLState) -> Result<(), BufferConversionError>;

    /// Clones the current buffer into another new buffer with a certain memory flag.
    fn clone(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;
}

impl BufferOperations for Buffer<cl_float> {
    fn clone(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if let Some(queue) = opencl_state.queues.first() {
            let context = &opencl_state.context;
            let size = self.size()?;
            let count = size / std::mem::size_of::<cl_float>();
            let mut copied_buff =
                Buffer::create(context, CL_MEM_READ_WRITE, count, ptr::null_mut())?;

            queue
                .enqueue_copy_buffer(self, &mut copied_buff, 0, 0, size, &[])?
                .wait()?;

            Ok(copied_buff)
        } else {
            Err(BufferOperationError::NoCommandQueueFoundError)
        }
    }

    fn dbg(&self, state: &OpenCLState) -> Result<(), BufferConversionError> {
        let vec = Vec::<f32>::from_buffer(self, false, state)?;

        println!("{:?}", vec);

        Ok(())
    }

    fn scale(&self, scaler: f32, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(SCALE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(scaler as cl_float))
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn shift(&self, num: f32, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(ADD_NUM_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(num as cl_float))
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn inverse_sqrt(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(INVERSE_SQRT_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn multiply(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(MULTIPLY_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&result)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(result)
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn divide(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(DIVIDE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&result)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(result)
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn subtract(
        &self,
        other: &Self,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(SUBTRACT_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&result)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(result)
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn add(&self, other: &Self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let context = &opencl_state.context;
        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(ADD_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = Buffer::create(context, CL_MEM_READ_WRITE, count_self, ptr::null_mut())?;

            ExecuteKernel::new(kernel)
                .set_arg(self)
                .set_arg(other)
                .set_arg(&result)
                .set_arg(&(count_self as cl_int))
                .set_global_work_size(count_self)
                .enqueue_nd_range(queue)?
                .wait()?;

            Ok(result)
        } else {
            Err(BufferOperationError::BuffersAreNotOfSameSize(
                count_self,
                count_other,
            ))
        }
    }

    fn sum(&self, opencl_state: &OpenCLState) -> Result<f32, BufferOperationError> {
        if opencl_state.devices.is_empty() {
            return Err(BufferOperationError::NoDeviceFoundError);
        }

        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let device = opencl_state.devices.first().unwrap();
        let queue = opencl_state.queues.first().unwrap();

        let operations_program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let reduce_kernel = operations_program.get_krnl(REDUCE_BUFFER_KERNEL_NAME)?;

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
            let (mut ev, mut current_buf) =
                reduce_buffer_by_summation(self, opencl_state, max_local_size, reduce_kernel, &[])?;
            current_count = current_buf.size()? / mem::size_of::<cl_float>();

            while current_count > 1 {
                (ev, current_buf) = reduce_buffer_by_summation(
                    &current_buf,
                    opencl_state,
                    max_local_size,
                    reduce_kernel,
                    &[ev],
                )?;
                current_count = current_buf.size()? / mem::size_of::<cl_float>();
            }

            let mut buf_slice = [0.0];

            queue.enqueue_read_buffer(
                &current_buf,
                CL_NON_BLOCKING,
                0,
                &mut buf_slice,
                &[ev.get()],
            )?;

            queue.finish()?;

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

impl IntricateProgram {
    /// Safely gets the kernel by name inside of the program.
    pub fn get_krnl(&self, kernel_name: &str) -> Result<&Kernel, KernelNotFoundError> {
        if !self.kernels.contains_key(&kernel_name.to_string()) {
            Err(kernel_name.to_string().into())
        } else {
            Ok(self.kernels.get(&kernel_name.to_string()).unwrap())
        }
    }
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

impl OpenCLState {
    /// Safely gets a program by name inside of the OpenCLState.
    pub fn get_prgm(&self, program_name: &str) -> Result<&IntricateProgram, ProgramNotFoundError> {
        if !self.programs.contains_key(&program_name.to_string()) {
            Err(program_name.to_string().into())
        } else {
            Ok(self.programs.get(&program_name.to_string()).unwrap())
        }
    }
}

#[derive(Debug, FromForAllUnnamedVariants)]
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
    if !&device_ids.is_empty() {
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

        compile_model(&mut state)?;

        compile_losses(&mut state)?;

        Ok(state)
    } else {
        Err(UnableToSetupOpenCLError::NoDeviceFound)
    }
}

pub(crate) trait BufferLike<T>
where
    Self: Sized,
{
    fn to_buffer(
        &self,
        blocking: bool,
        opencl_state: &OpenCLState,
    ) -> Result<Buffer<T>, BufferConversionError>;

    fn from_buffer(
        buffer: &Buffer<T>,
        blocking: bool,
        opencl_state: &OpenCLState,
    ) -> Result<Self, BufferConversionError>;
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that may happen when trying to create a buffer
/// from a flat Vec's content
pub enum BufferConversionError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when there is no command queue inside of the OpenCLState.
    NoCommandQueueFound,
}

pub(crate) fn empty_buffer(
    count: usize,
    flags: cl_mem_flags,
    opencl_state: &OpenCLState,
) -> Result<Buffer<cl_float>, ClError> {
    Buffer::create(&opencl_state.context, flags, count, ptr::null_mut())
}

impl BufferLike<cl_float> for Vec<f32> {
    fn to_buffer(
        &self,
        blocking: bool,
        opencl_state: &OpenCLState,
    ) -> Result<Buffer<cl_float>, BufferConversionError> {
        if let Some(queue) = opencl_state.queues.first() {
            let context = &opencl_state.context;

            let mut buffer =
                Buffer::create(context, CL_MEM_READ_WRITE, self.len(), ptr::null_mut())?;

            if blocking {
                queue
                    .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, self.as_slice(), &[])?
                    .wait()?;
            } else {
                queue
                    .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, self.as_slice(), &[])?
                    .wait()?;
            }

            Ok(buffer)
        } else {
            Err(BufferConversionError::NoCommandQueueFound)
        }
    }

    fn from_buffer(
        buffer: &Buffer<cl_float>,
        blocking: bool,
        opencl_state: &OpenCLState,
    ) -> Result<Vec<f32>, BufferConversionError> {
        if let Some(queue) = opencl_state.queues.first() {
            let size = buffer.size()?;
            let count = size / mem::size_of::<cl_float>();

            let mut vec = vec![0.0; count];

            if blocking {
                queue
                    .enqueue_read_buffer(&buffer, CL_BLOCKING, 0, vec.as_mut_slice(), &[])?
                    .wait()?;
            } else {
                queue
                    .enqueue_read_buffer(&buffer, CL_NON_BLOCKING, 0, vec.as_mut_slice(), &[])?
                    .wait()?;
            }

            Ok(vec)
        } else {
            Err(BufferConversionError::NoCommandQueueFound)
        }
    }
}

#[cfg(test)]
mod test_opencl_utils {
    use rand::{thread_rng, Rng};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    use super::{setup_opencl, BufferLike, BufferOperations, DeviceType};

    #[test]
    fn should_add_buffers_correctly() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 5123;

        let vec1: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let vec2: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a + b).collect();

        let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
        let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

        let actual = Vec::<f32>::from_buffer(
            &buff1.add(&buff2, &opencl_state).unwrap(),
            true,
            &opencl_state,
        )
        .unwrap();

        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
        });
    }

    #[test]
    fn should_subtract_buffers_correctly() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 5123;

        let vec1: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let vec2: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a - b).collect();

        let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
        let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

        let actual = Vec::<f32>::from_buffer(
            &buff1.subtract(&buff2, &opencl_state).unwrap(),
            true,
            &opencl_state,
        )
        .unwrap();

        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
        });
    }

    #[test]
    fn should_multiply_buffers_correctly() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 5123;

        let vec1: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-153_f32..141_f32) })
            .collect();
        let vec2: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-151_f32..121_f32) })
            .collect();
        let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a * b).collect();

        let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
        let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

        let actual = Vec::<f32>::from_buffer(
            &buff1.multiply(&buff2, &opencl_state).unwrap(),
            true,
            &opencl_state,
        )
        .unwrap();

        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
        });
    }

    #[test]
    fn should_divide_buffers_correctly() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 5123;

        let vec1: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let vec2: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();
        let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a / b).collect();

        let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
        let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

        let actual = Vec::<f32>::from_buffer(
            &buff1.divide(&buff2, &opencl_state).unwrap(),
            true,
            &opencl_state,
        )
        .unwrap();

        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
        });
    }

    #[test]
    fn should_scale_buffers_correctly() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 5123;

        let vec1: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
            .collect();

        let scaler = 0.123;
        let expected: Vec<f32> = vec1.iter().map(|a| a * scaler).collect();

        let buff = vec1.to_buffer(true, &opencl_state).unwrap();

        let actual = Vec::<f32>::from_buffer(
            &buff.scale(scaler, &opencl_state).unwrap(),
            true,
            &opencl_state,
        )
        .unwrap();

        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
        });
    }

    #[test]
    fn should_sum_buffer_to_correct_value() {
        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut rng = thread_rng();
        let numbers_amount = 256;
        let test_vec: Vec<f32> = (0..numbers_amount)
            .map(|_| -> f32 { rng.gen_range(-123.31_f32..3193.31_f32) })
            .collect();
        let expected_sum: f32 = test_vec.par_iter().sum();

        let buff = test_vec.to_buffer(true, &opencl_state).unwrap();

        let actual_result = buff.sum(&opencl_state).unwrap();

        println!("{} - {}", actual_result, expected_sum);
        assert!(
            ((actual_result - expected_sum) / (actual_result.max(expected_sum))).abs() <= 0.0001
        );
    }
}