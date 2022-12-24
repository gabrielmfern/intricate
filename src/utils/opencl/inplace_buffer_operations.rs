//! The module that contains all of the available Intricate inplace buffer operations.
//!
//! Inplace because it will not create a new buffer but rather mutate the &self instead of the
//! `BufferOperations` trait.

use std::mem;

use opencl3::{
    memory::{Buffer, ClMem},
    types::{cl_float, cl_int}, kernel::ExecuteKernel,
};

use super::{opencl_state::{ensure_program, EnsureKernelsAndProgramError}, BufferOperationError, OpenCLState};

const PROGRAM_SOURCE: &str = include_str!("kernels/inplace_buffer_operations.cl");
const PROGRAM_NAME: &str = "INPLACE_BUFFER_OPERATIONS";

const CLIP_MIN_MAX_KERNEL_NAME: &str = "clip_min_max";
const SCALE_KERNEL_NAME: &str = "scale";
const SQRT_KERNEL_NAME: &str = "sqrt";
const INVERSE_SQRT_KERNEL_NAME: &str = "inverse_sqrt";
const SHIFT_KERNEL_NAME: &str = "shift";
const ADD_KERNEL_NAME: &str = "add";
const SUBTRACT_KERNEL_NAME: &str = "subtract";
const MULTIPLY_KERNEL_NAME: &str = "multiply";
const DIVIDE_KERNEL_NAME: &str = "divide";

pub(crate) fn compile_inplace_buffer_operations_program(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        CLIP_MIN_MAX_KERNEL_NAME,
        SCALE_KERNEL_NAME,
        SQRT_KERNEL_NAME,
        INVERSE_SQRT_KERNEL_NAME,
        SHIFT_KERNEL_NAME,
        ADD_KERNEL_NAME,
        SUBTRACT_KERNEL_NAME,
        MULTIPLY_KERNEL_NAME,
        DIVIDE_KERNEL_NAME
    ];

    ensure_program(
        opencl_state,
        PROGRAM_NAME,
        PROGRAM_SOURCE,
        "",
        kernels,
    )
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

    /// Clips all of the values inside of the buffer using the min and max function
    fn clip_min_max_inplace(
        &mut self,
        min: f32,
        max: f32,
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

    /// Takes the sqrt of Self
    fn sqrt_inplc(&mut self, opencl_state: &OpenCLState) -> Result<(), BufferOperationError>;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(SCALE_KERNEL_NAME)?;

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

    fn clip_min_max_inplace(
        &mut self,
        min: f32,
        max: f32,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(CLIP_MIN_MAX_KERNEL_NAME)?;

        let size_self = self.size()?;
        let count_self = size_self / mem::size_of::<cl_float>();

        ExecuteKernel::new(kernel)
            .set_arg(self)
            .set_arg(&(min as cl_float))
            .set_arg(&(max as cl_float))
            .set_arg(&(count_self as cl_int))
            .set_global_work_size(count_self)
            .enqueue_nd_range(queue)?
            .wait()?;

        Ok(())
    }

    fn sqrt_inplc(&mut self, opencl_state: &OpenCLState) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(SQRT_KERNEL_NAME)?;

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

    fn inverse_sqrt_inplc(
        &mut self,
        opencl_state: &OpenCLState,
    ) -> Result<(), BufferOperationError> {
        if opencl_state.queues.is_empty() {
            return Err(BufferOperationError::NoCommandQueueFoundError);
        }

        let queue = opencl_state.queues.first().unwrap();

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(INVERSE_SQRT_KERNEL_NAME)?;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(SHIFT_KERNEL_NAME)?;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(ADD_KERNEL_NAME)?;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(SUBTRACT_KERNEL_NAME)?;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(DIVIDE_KERNEL_NAME)?;

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

        let program = opencl_state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(MULTIPLY_KERNEL_NAME)?;

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