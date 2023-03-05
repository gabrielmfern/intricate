//! The module that contains standard buffer operations.
//!
//! Not recommended to be used more than once in a row, instead a kernel should be used for that.

use std::mem;

use opencl3::{
    error_codes::ClError,
    event::Event,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    types::{cl_event, cl_float, cl_int, CL_NON_BLOCKING, cl_uint},
};

use crate::utils::{find_divsor_of_n_closest_to_m, gcd, opencl::BufferLike};

use super::{
    empty_buffer, find_optimal_local_and_global_work_sizes,
    opencl_state::{ensure_program, EnsureKernelsAndProgramError},
    BufferConversionError, BufferOperationError,
};

use super::opencl_state::OpenCLState;

const BUFFER_OPERATIONS_PROGRAM_SOURCE: &str = include_str!("kernels/buffer_operations.cl");
const BUFFER_OPERATIONS_PROGRAM_NAME: &str = "BUFFER_OPERATIONS";

const REDUCE_BUFFER_KERNEL_NAME: &str = "sum_all_values_in_workgroups";
const SUM_ALL_VALUES_IN_ROW_WOIRK_GROUPS: &str = "sum_all_values_in_row_work_groups";
const SCALE_BUFFER_KERNEL_NAME: &str = "scale";
const FFT_1D_BUFFER_KERNEL_NAME: &str = "fft_1d";
const INVERSE_SQRT_BUFFER_KERNEL_NAME: &str = "inverse_sqrt";
const SQRT_BUFFER_KERNEL_NAME: &str = "squareroot";
const ADD_BUFFER_KERNEL_NAME: &str = "add";
const ADD_NUM_BUFFER_KERNEL_NAME: &str = "add_num";
const SUBTRACT_BUFFER_KERNEL_NAME: &str = "subtract";
const MULTIPLY_BUFFER_KERNEL_NAME: &str = "multiply";
const DIVIDE_BUFFER_KERNEL_NAME: &str = "divide";

pub(crate) fn compile_buffer_operations_program(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        REDUCE_BUFFER_KERNEL_NAME,
        SUM_ALL_VALUES_IN_ROW_WOIRK_GROUPS,
        SCALE_BUFFER_KERNEL_NAME,
        FFT_1D_BUFFER_KERNEL_NAME,
        INVERSE_SQRT_BUFFER_KERNEL_NAME,
        SQRT_BUFFER_KERNEL_NAME,
        ADD_BUFFER_KERNEL_NAME,
        ADD_NUM_BUFFER_KERNEL_NAME,
        SUBTRACT_BUFFER_KERNEL_NAME,
        MULTIPLY_BUFFER_KERNEL_NAME,
        DIVIDE_BUFFER_KERNEL_NAME,
    ];

    ensure_program(
        opencl_state,
        BUFFER_OPERATIONS_PROGRAM_NAME,
        BUFFER_OPERATIONS_PROGRAM_SOURCE,
        "",
        kernels,
    )
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

fn reduce_buffer_by_row_wise_summation(
    buffer: &Buffer<cl_float>,
    width: usize,
    height: usize,
    state: &OpenCLState,
    max_local_size: usize,
    kernel: &Kernel,
    wait_list: &[Event],
) -> Result<(Event, Buffer<cl_float>), ClError> {
    let queue = state.queues.first().unwrap();

    let mut global_size_1 = width;
    let mut local_size_1 = gcd(global_size_1, max_local_size);
    while local_size_1 == 1 {
        global_size_1 += 1;
        local_size_1 = gcd(global_size_1, max_local_size);
    }

    let mut global_size_0 = height;
    let mut local_size_0 = find_divsor_of_n_closest_to_m(height, max_local_size / local_size_1);
    while local_size_0 == 1 && local_size_1 != max_local_size {
        global_size_0 += 1;
        local_size_0 = find_divsor_of_n_closest_to_m(global_size_0, max_local_size / local_size_1);
    }

    let reduced_width = global_size_1 / local_size_1;
    let mut current_reduced_buffer =
        empty_buffer(height * reduced_width, CL_MEM_READ_WRITE, state)?;
    let event = ExecuteKernel::new(kernel)
        .set_arg(buffer)
        .set_arg(&mut current_reduced_buffer)
        .set_arg_local_buffer(
            local_size_0.min(height) * local_size_1.min(width) * std::mem::size_of::<cl_float>(),
        )
        .set_arg(&(reduced_width as cl_int))
        .set_arg(&(width as cl_int))
        .set_arg(&(height as cl_int))
        .set_event_wait_list(&wait_list.iter().map(|e| e.get()).collect::<Vec<cl_event>>())
        .set_global_work_sizes(&[global_size_0, global_size_1])
        .set_local_work_sizes(&[local_size_0, local_size_1])
        .enqueue_nd_range(queue)?;

    Ok((event, current_reduced_buffer))
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
    /// - If the summation kernel was not found in the program for buffer operations.
    fn sum(&self, opencl_state: &OpenCLState) -> Result<f32, BufferOperationError>;

    /// Sums all of the numbers inside of a buffer separated by rows (in which, their width
    /// are received as parameters) and returns a resulting buffer containing the summation per rows.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the summation kernel was not found in the program for buffer operations.
    /// - If the specified width does not match the &self's width
    fn sum_2d_per_row(
        &self,
        state: &OpenCLState,
        width: usize,
    ) -> Result<Self, BufferOperationError>;

    /// The Cooley Tuukey implemtantion of the fast discrete fourier transform.
    /// The FFT will return a new buffer that is twice the size of the original
    /// beucase it contains the imaginary parts of the frequencies.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If the count of &self is not a power of two.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the fft kernel was not found in the program for buffer operations.
    fn fft_1d(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

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

    /// Takes the sqrt of each one of the numbers inside Self
    /// and returns a new Buffer with the resultign nubmers
    fn sqrt(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError>;

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
            let size = self.size()?;
            let count = size / std::mem::size_of::<cl_float>();
            let mut copied_buff = empty_buffer(count, CL_MEM_READ_WRITE, opencl_state)?;

            queue
                .enqueue_copy_buffer(self, &mut copied_buff, 0, 0, size, &[])?
                .wait()?;

            Ok(copied_buff)
        } else {
            Err(BufferOperationError::NoCommandQueueFoundError)
        }
    }

    fn dbg(&self, state: &OpenCLState) -> Result<(), BufferConversionError> {
        let vec = Vec::from_buffer(self, false, state)?;

        println!("{:?}", vec);

        Ok(())
    }

    fn fft_1d(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(FFT_1D_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        if !count_self.is_power_of_two() {
            return Err(BufferOperationError::BufferIsNotOfExpectedSize(
                count_self,
                "The buffer's length must be a power of two to optimize the calculations!",
            ));
        }

        let result = empty_buffer(2 * count_self, CL_MEM_READ_WRITE, opencl_state)?;

        let log_of_count = (count_self as f32).log2().floor() as usize;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(count_self as cl_uint))
            .set_arg(&(log_of_count as cl_uint))
            .set_global_work_size(count_self >> 1)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn scale(&self, scaler: f32, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(SCALE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(ADD_NUM_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

    fn sqrt(&self, opencl_state: &OpenCLState) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(SQRT_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(INVERSE_SQRT_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(MULTIPLY_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(DIVIDE_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(SUBTRACT_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let kernel = program.get_krnl(ADD_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let size_other = other.size()?;

        let count_self = size_self / mem::size_of::<cl_float>();
        let count_other = size_other / mem::size_of::<cl_float>();
        if size_self == size_other {
            let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

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

    fn sum_2d_per_row(
        &self,
        state: &OpenCLState,
        initial_width: usize,
    ) -> Result<Self, BufferOperationError> {
        if state.devices.is_empty() {
            return Err(BufferOperationError::NoDeviceFoundError);
        }

        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let device = state.devices.first().unwrap();

        let operations_program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;

        let reduce_kernel = operations_program.get_krnl(SUM_ALL_VALUES_IN_ROW_WOIRK_GROUPS)?;

        let max_local_size = device.max_work_group_size()?;

        let mut current_count = self.size()? / mem::size_of::<cl_float>();
        let mut width = initial_width;
        if current_count % initial_width != 0 {
            return Err(BufferOperationError::DimensionWasNotAsSpecified("width"));
        }
        let height = current_count / width;

        if current_count == height || current_count == 0 {
            self.clone(state)
        } else {
            let (mut ev, mut current_buf) = reduce_buffer_by_row_wise_summation(
                self,
                width,
                height,
                state,
                max_local_size,
                reduce_kernel,
                &[],
            )?;
            current_count = current_buf.size()? / mem::size_of::<cl_float>();
            width = current_count / height;

            while width > 1 {
                (ev, current_buf) = reduce_buffer_by_row_wise_summation(
                    &current_buf,
                    width,
                    height,
                    state,
                    max_local_size,
                    reduce_kernel,
                    &[ev],
                )?;
                current_count = current_buf.size()? / mem::size_of::<cl_float>();
                width = current_count / height;
            }

            //             queue.finish()?;

            Ok(current_buf)
        }
    }
}