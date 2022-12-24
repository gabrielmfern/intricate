//! The module that contains utilities for converting from and to buffers with Vecs

use std::{mem, ptr};

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    error_codes::ClError,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    types::{cl_float, CL_BLOCKING, CL_NON_BLOCKING},
};

use super::OpenCLState;

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