//! The module that contains standard buffer operations.
//!
//! Not recommended to be used more than once in a row, instead a kernel should be used for that.

use std::{mem, ops::Range};

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
    BufferConversionError, BufferOperationError, InplaceBufferOperations,
};

use super::opencl_state::OpenCLState;

const BUFFER_OPERATIONS_PROGRAM_SOURCE: &str = include_str!("kernels/buffer_operations.cl");
const BUFFER_OPERATIONS_PROGRAM_NAME: &str = "BUFFER_OPERATIONS";

const REDUCE_BUFFER_KERNEL_NAME: &str = "sum_all_values_in_workgroups";
const SUM_ALL_VALUES_IN_ROW_WOIRK_GROUPS: &str = "sum_all_values_in_row_work_groups";
const SCALE_BUFFER_KERNEL_NAME: &str = "scale";
const COMPLEX_POINT_WISE_MULTIPLY_KERNEL_NAME: &str = "complex_point_wise_multiply";
const COMPLEX_POINT_WISE_MULTIPLY_FOR_SAMPLED_CONVOLUTION_KERNEL_NAME: &str = "sampled_complex_pointwise_mutliply";
const PADD_2D_KERNEL_NAME: &str = "padd_2d";
const TO_COMPLEX_FLOAT_TWO_BUFFER_KERNEL_NAME: &str = "to_complex_float2_buffer";
const SLICE_2D_KERNEL_NAME: &str = "slice_2d";
const GET_REAL_PART_KERNEL_NAME: &str = "get_real_part";
const FFT_1D_BUFFER_KERNEL_NAME: &str = "fft";
const IFFT_1D_BUFFER_KERNEL_NAME: &str = "ifft";
const COMPLEX_TRANSPOSE_KERNEL_NAME: &str = "complex_transpose";
const TRANSPOSE_KERNEL_NAME: &str = "transpose";
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
        PADD_2D_KERNEL_NAME,
        SLICE_2D_KERNEL_NAME,
        FFT_1D_BUFFER_KERNEL_NAME,
        GET_REAL_PART_KERNEL_NAME,
        TO_COMPLEX_FLOAT_TWO_BUFFER_KERNEL_NAME,
        COMPLEX_POINT_WISE_MULTIPLY_KERNEL_NAME,
        COMPLEX_POINT_WISE_MULTIPLY_FOR_SAMPLED_CONVOLUTION_KERNEL_NAME,
        IFFT_1D_BUFFER_KERNEL_NAME,
        COMPLEX_TRANSPOSE_KERNEL_NAME,
        TRANSPOSE_KERNEL_NAME,
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

    /// Padds the current buffer by appending zeroes into the end of each row
    /// and each column of the matrix saved into &self
    ///
    /// This does work with multiple matrices as once, that is, with multiple samples.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the padd_2d kernel was not found in the program for buffer operations.
    /// - If the specified width and height do not agree with &self's volume.
    fn padd_2d(
        &self,
        width: usize,
        height: usize,
        new_width: usize,
        new_height: usize,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// Slices the current buffer by creating a new buffer of data that contains
    /// the data of &self inside a certain range.
    ///
    /// This does work with multiple matrices as once, that is, with multiple samples.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the slice_2d kernel was not found in the program for buffer operations.
    /// - If the specified width and height do not agree with &self's volume.
    fn slice_2d(
        &self,
        x_range: Range<usize>,
        y_range: Range<usize>,
        width: usize,
        height: usize,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// Computes the 2d convolution between &self and other using the 2d FFT.
    /// All of the samples of &other are convolved with each and every one of the samples of &self,
    /// generating a new buffer with the amount of samples of both mutliplied.
    /// 
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the fft, ifft, complex tranpose or complex point-wise multiplication kernels were not found 
    /// in the program for buffer operations.
    /// - If the specified width and height do not agree with &self's volume.
    /// - If the specified filter_width and filter_height do not agre with  
    /// &other's volume
    fn convolve_2d(
        &self,
        state: &OpenCLState,
        other: &Self,
        self_width: usize,
        self_height: usize,
        filter_width: usize,
        filter_height: usize,
        range: (Range<usize>, Range<usize>)
    ) -> Result<Self, BufferOperationError>;

    /// Computes the sampled 2d convolution between &self and other using the 2d FFT.
    /// 
    /// This convolution is made for when &other is much larger than &self but its
    /// amount of samples is a multiple of the samples of &self.
    ///
    /// This computes the a point-wise convolution per image-other sample and then
    /// repeats this process depending on the amount of filters in &other.
    /// 
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the fft, ifft, complex tranpose or sampled complex point-wise multiplication 
    /// kernels were not found.
    /// in the program for buffer operations.
    /// - If the specified width and height do not agree with &self's volume.
    /// - If the specified filter_width and filter_height do not agree with &other's volume.
    /// - If the amount of samples of &self is not a divisor of the amount of samples of &other.
    fn sampled_convolve_2d(
        &self,
        state: &OpenCLState,
        other: &Self,
        self_width: usize,
        self_height: usize,
        filter_width: usize,
        filter_height: usize,
        range: (Range<usize>, Range<usize>)
    ) -> Result<Self, BufferOperationError>;

    /// Computes a sampled 2d complex point-wise multiplication between &self and &other.
    ///
    /// This is almost the same as the point-wise multiply operation but it matches the samples
    /// of &self and &other instead of repeating each sample of &other into each and every one
    /// sample of &self.
    ///
    /// This is mostly used in the sampled_convolve_2d buffer operation for the calculation of
    /// gradients in the Conv2d layer.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - if the sampled complex point-wise multiplication kernel is not found inside the program.
    /// - If the specified width and height do not agree with &self's volume.
    /// - If the width and height of each sample of &self are not the same of &other.
    /// - If the amount of samples of &self is not a divisor of the amount of samples of &other.
    fn sampled_complex_pointwise_mutliply(
        &self,
        other: &Self,
        width: usize,
        height: usize,
        state: &OpenCLState
    ) -> Result<Self, BufferOperationError>;

    /// Returns the real part of a complex buffer into a new buffer of half the size of the
    /// original buffer.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If the count of &self is not a even number, meaning it is not made of float2 numbers.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the kernel was not found in the program for buffer operations.
    fn real_part(
        &self,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// Returns a new buffer with twice the volume of &self that contanins the imaginary part
    /// of all the values of &self being zero.
    ///
    /// # Errors
    ///
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the to_complex_float2_buffer kernel was not found in the program for buffer operations.
    fn to_complex_float2_buffer(
        &self,
        state: &OpenCLState
    ) -> Result<Self, BufferOperationError>;

    /// Evaluates a point-wise complex multiplication between two float2 buffers: &self and &other.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If the count of &self or of &other is not a even number, meaning it is not made of float2 numbers.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the kernel was not found in the program for buffer operations.
    fn complex_multiply(
        &self,
        other_samples_amount: usize,
        self_samples_amount: usize,
        other: &Self,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError>;

    /// The Cooley Tuukey implemtantion of the fast discrete fourier transform.
    /// The FFT will return a new buffer that is twice the size of the original
    /// beucase it contains the imaginary parts of the frequencies. This buffer is intended
    /// to generally be used in a kernel with a `float2 *buffer` representation.
    ///
    /// Assumes that the input buffer is a float2 buffer over which the X is the real part and the
    /// Y is the imaginary part.
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
    fn fft(&self, opencl_state: &OpenCLState, samples_amount: usize) -> Result<Self, BufferOperationError>;

    /// The Cooley Tukey implementation of the fast discrete inverse fourier trasnform.
    /// The FFT will return a new buffer that is the size of the original including imaginary
    /// numbers.
    /// Assumes that the input buffer is a float2 buffer over which the X is the real part and the
    /// Y is the imaginary part.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If the count of &self is not a power of two.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the ifft kernel was not found in the program for buffer operations.
    fn ifft(&self, opencl_state: &OpenCLState, samples_amount: usize) -> Result<Self, BufferOperationError>;

    /// Transposes the original matrix, assuming it is a complex nnumber matrix
    /// that contains only **float2** numbers.
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the complex tranpose kernel was not found in the program for buffer operations.
    fn complex_tranpose(
        &self, 
        opencl_state: &OpenCLState, 
        width: usize,
        height: usize
    ) -> Result<Self, BufferOperationError>;

    /// Tranposes &self and returns the transposed &self
    ///
    /// # Errors
    ///
    /// This method will yield an error in the following cases:
    /// - There is no device in the **opencl_state**.
    /// - There is no command queue in the **opencl_state**.
    /// - If something goes wrong while executing the kernels.
    /// - If the program for buffer operations was not compiled in **opencl_state**.
    /// - If the tranpose kernel was not found in the program for buffer operations.
    fn tranpose(
        &self, 
        opencl_state: &OpenCLState, 
        width: usize,
        height: usize
    ) -> Result<Self, BufferOperationError>;

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
    fn dbg(self, state: &OpenCLState) -> Result<Self, BufferConversionError>;

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

    fn padd_2d(
        &self,
        width: usize,
        height: usize,
        new_width: usize,
        new_height: usize,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(PADD_2D_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        if count_self / width % height != 0 {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "Cannot determine the amount of samples of the buffer for padding the matrices since the width and height were not as specified!"
            ));
        }
        let samples_amount = count_self / width / height;

        let result = empty_buffer(samples_amount * new_width * new_height, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(width as cl_uint))
            .set_arg(&(height as cl_uint))
            .set_global_work_sizes(&[new_width, new_height, samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }
    
    fn slice_2d(
        &self,
        x_range: Range<usize>,
        y_range: Range<usize>,
        width: usize,
        height: usize,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(SLICE_2D_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        if count_self / width % height != 0 {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "Cannot determine the amount of samples of the buffer for slicing the matrices since the width and height were not as specified!"
            ));
        }
        let samples_amount = count_self / width / height;

        let slice_width = x_range.end - x_range.start + 1;
        let slice_height = y_range.end - y_range.start + 1;
        let result = empty_buffer(samples_amount * slice_width * slice_height, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(x_range.start as cl_uint))
            .set_arg(&(y_range.start as cl_uint))
            .set_arg(&(width as cl_uint))
            .set_arg(&(height as cl_uint))
            .set_global_work_sizes(&[slice_width, slice_height, samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn sampled_complex_pointwise_mutliply(
        &self,
        other: &Self,
        width: usize,
        height: usize,
        state: &OpenCLState
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoDeviceFoundError);
        }

        let queue = state.queues.first().unwrap();

        let self_count = self.size()? / mem::size_of::<cl_float>() / 2;
        let other_count = other.size()? / mem::size_of::<cl_float>() / 2;

        if self_count % width != 0 || self_count % height != 0 {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "It seems that the volume of &self does not match the provided width and height!"
            ));
        }
        
        if other_count % width != 0 || other_count % height != 0 {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "It seems that the volume of &other does not match the provided width and height!"
            ));
        }

        let samples_amount = self_count / width / height;
        if other_count % samples_amount != 0 {
            return Err(BufferOperationError::BufferIsNotOfExpectedSize(
                other_count, 
                "The samples amount of &self do not divide evenly into the filter"
            ));
        }
        let other_sub_samples_amount = other_count / width / height / dbg!(samples_amount);

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let multiply_kernel = program.get_krnl(COMPLEX_POINT_WISE_MULTIPLY_FOR_SAMPLED_CONVOLUTION_KERNEL_NAME)?;

        let multiplication = empty_buffer(2 * self_count * other_sub_samples_amount, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(multiply_kernel)
            .set_arg(self)
            .set_arg(other)
            .set_arg(&multiplication)
            .set_global_work_sizes(&[self_count, other_sub_samples_amount, samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(multiplication)
    }

    fn sampled_convolve_2d(
        &self,
        state: &OpenCLState,
        filter: &Self,
        self_width: usize,
        self_height: usize,
        filter_width: usize,
        filter_height: usize,
        range: (Range<usize>, Range<usize>)
    ) -> Result<Self, BufferOperationError> {
        let self_count = self.size()? / mem::size_of::<cl_float>();
        let filter_pixel_count = filter.size()? / mem::size_of::<cl_float>();

        let samples_amount = self_count / self_width / self_height;
        if filter_pixel_count % samples_amount != 0 {
            return Err(BufferOperationError::BufferIsNotOfExpectedSize(
                filter_pixel_count, 
                "The samples amount of &self do not divide evenly into the filter"
            ));
        }
        let filters_amount = filter_pixel_count / filter_width / filter_height / samples_amount;

        let padded_width = self_width + filter_width - 1;
        let padded_height = self_height + filter_height - 1;
        let even_padded_width = padded_width.next_power_of_two();
        let even_padded_height = padded_height.next_power_of_two();

        let fft_self = self
            .padd_2d(self_width, self_height, even_padded_width, even_padded_height, state)?
            .to_complex_float2_buffer(state)?
            .fft(state, samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .fft(state, samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?;

        let fft_filter = filter
            .padd_2d(filter_width, filter_height, even_padded_width, even_padded_height, state)?
            .to_complex_float2_buffer(state)?
            .fft(state, filters_amount * samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .fft(state, filters_amount * samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?;

        let x_range = range.0;
        let y_range = range.1;

        let convolution = fft_self
            .sampled_complex_pointwise_mutliply(
                &fft_filter, 
                even_padded_width, 
                even_padded_height, 
                state
            )?
            .ifft(state, filters_amount * samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .ifft(state, filters_amount * samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?
            .real_part(state)?
            .dbg(state).unwrap()
            .slice_2d(
                x_range,
                y_range,
                even_padded_width,
                even_padded_height,
                state,
            )?;

        Ok(convolution)
    }

    fn convolve_2d(
        &self,
        state: &OpenCLState,
        other: &Self,
        self_width: usize,
        self_height: usize,
        other_width: usize,
        other_height: usize,
        range: (Range<usize>, Range<usize>)
    ) -> Result<Self, BufferOperationError> {
        let self_count = self.size()? / mem::size_of::<cl_float>();
        let other_count = other.size()? / mem::size_of::<cl_float>();

        let samples_amount = self_count / self_width / self_height;
        let other_samples_amount = other_count / other_width / other_height;


        let padded_width = self_width + other_width - 1;
        let padded_height = self_height + other_height - 1;
        let even_padded_width = padded_width.next_power_of_two();
        let even_padded_height = padded_height.next_power_of_two();

        let fft_self = self
            .padd_2d(self_width, self_height, even_padded_width, even_padded_height, state)?
            .to_complex_float2_buffer(state)?
            .fft(state, samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .fft(state, samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?;

        let fft_other = other
            .padd_2d(other_width, other_height, even_padded_width, even_padded_height, state)?
            .to_complex_float2_buffer(state)?
            .fft(state, samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .fft(state, samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?;

        let x_range = range.0;
        let y_range = range.1;



        let convolution = fft_self
            .complex_multiply(other_samples_amount, samples_amount, &fft_other, state)?
            .ifft(state, other_samples_amount * samples_amount * even_padded_height)?
            .complex_tranpose(state, even_padded_width, even_padded_height)?
            .ifft(state, other_samples_amount * samples_amount * even_padded_width)?
            .complex_tranpose(state, even_padded_height, even_padded_width)?
            .real_part(state)?
            .slice_2d(
                x_range,
                y_range,
                even_padded_width,
                even_padded_height,
                state,
            )?;

        Ok(convolution)
    }

    fn dbg(self, state: &OpenCLState) -> Result<Self, BufferConversionError> {
        let vec = Vec::from_buffer(&self, false, state)?;
        println!("{:?}", vec);
        Ok(self)
    }

    fn to_complex_float2_buffer(
        &self,
        state: &OpenCLState
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(TO_COMPLEX_FLOAT_TWO_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        let result = empty_buffer(count_self << 1, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_global_work_sizes(&[count_self])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn real_part(
        &self,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(GET_REAL_PART_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        if count_self % 2 != 0 {
            return Err(
                BufferOperationError::BufferIsNotOfExpectedSize(
                    count_self, 
                    "&self must be a float2 buffer for it to store complex numbers, it seems to be odd"
                )
            );
        }

        let result = empty_buffer(count_self >> 1, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_global_work_sizes(&[count_self >> 1])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn complex_multiply(
        &self,
        other_samples_amount: usize,
        self_samples_amount: usize,
        other: &Self,
        state: &OpenCLState,
    ) -> Result<Self, BufferOperationError> {
        if state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(COMPLEX_POINT_WISE_MULTIPLY_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        if count_self % 2 != 0 {
            return Err(
                BufferOperationError::BufferIsNotOfExpectedSize(
                    count_self, 
                    "&self must be a float2 buffer for the complex multiplication to work!"
                )
            );
        }

        let size_other = other.size()?;
        let count_other = size_other / mem::size_of::<cl_float>();

        let matrix_volume = count_self / self_samples_amount / 2;
        if matrix_volume != count_other / other_samples_amount / 2 {
            return Err(
                BufferOperationError::BuffersAreNotOfSameSize(
                    count_other,
                    count_self,
                )
            );
        }

        let result = empty_buffer(other_samples_amount * count_self, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(other)
            .set_arg(&result)
            .set_global_work_sizes(&[matrix_volume, self_samples_amount, other_samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn tranpose(
        &self, 
        opencl_state: &OpenCLState, 
        width: usize,
        height: usize
    ) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(TRANSPOSE_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        let samples_amount = count_self / width / height;
        if width * height * samples_amount != count_self {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "The samples amount, width and height specified to transpose the samples of matrices do not match the buffer's volume",
            ));
        }

        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(height as cl_uint))
            .set_arg(&(width as cl_uint))
            .set_global_work_sizes(&[width, height, samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn complex_tranpose(
        &self, 
        opencl_state: &OpenCLState, 
        width: usize,
        height: usize
    ) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(COMPLEX_TRANSPOSE_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        let samples_amount = count_self / width / height / 2;
        if width * height * samples_amount * 2 != count_self {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "The samples amount, width and height specified to transpose the samples of matrices do not match the buffer's volume",
            ));
        }

        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(height as cl_uint))
            .set_arg(&(width as cl_uint))
            .set_global_work_sizes(&[width, height, samples_amount])
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(result)
    }

    fn ifft(&self, opencl_state: &OpenCLState, samples_amount: usize) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(IFFT_1D_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        let width = (count_self / samples_amount) >> 1;

        if width * samples_amount * 2 != count_self {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "The samples amount specified to do a IFFT over does not match the volume of the buffer!",
            ));
        }

        if !width.is_power_of_two() {
            return Err(BufferOperationError::BufferIsNotOfExpectedSize(
                width, 
                "THe width of the samples inside of the buffer to do a IFFT must be a power of two!"
            ));
        }

        //                         this size already includes complex numbers
        let mut result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

        let log_width = (width as f32).log2().floor() as usize;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(width as cl_uint))
            .set_arg(&(log_width as cl_uint))
            .set_global_work_sizes(&[samples_amount, width >> 1])
            .enqueue_nd_range(queue)?
            .wait()?;

        result.scale_inplc(1.0 / width as f32, opencl_state)?;

        Ok(result)
    }

    fn fft(&self, opencl_state: &OpenCLState, samples_amount: usize) -> Result<Self, BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(BUFFER_OPERATIONS_PROGRAM_NAME)?;
        let kernel = program.get_krnl(FFT_1D_BUFFER_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();
        let width = count_self / samples_amount >> 1;

        if width * samples_amount << 1 != count_self {
            return Err(BufferOperationError::DimensionWasNotAsSpecified(
                "The samples amount specified to do a FFT over does not match the volume of the buffer!",
            ));
        }

        if !width.is_power_of_two() {
            return Err(BufferOperationError::BufferIsNotOfExpectedSize(
                width, 
                "THe width of the samples inside of the buffer to do a FFT must be a power of two!"
            ));
        }

        let result = empty_buffer(count_self, CL_MEM_READ_WRITE, opencl_state)?;

        let log_width = (width as f32).log2().floor() as usize;

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&result)
            .set_arg(&(width as cl_uint))
            .set_arg(&(log_width as cl_uint))
            .set_global_work_sizes(&[samples_amount, width >> 1])
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