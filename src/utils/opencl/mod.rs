//! The module that contains all of the OpenCL utilities written for Intricate

use std::ptr;

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{types::{cl_mem_flags, cl_float}, memory::Buffer, error_codes::ClError};

use crate::types::{ProgramNotFoundError, KernelNotFoundError};

use super::gcd;

pub mod opencl_state;
pub mod buffer_operations;
#[cfg(test)]
mod buffer_operations_test;
pub mod buffer_like;
pub mod inplace_buffer_operations;

pub use opencl_state::DeviceType;
pub use opencl_state::OpenCLState;
pub(crate) use buffer_like::BufferLike;
pub use buffer_like::BufferConversionError;
pub(crate) use opencl_state::ensure_program;
pub use opencl_state::setup_opencl;
pub use buffer_operations::BufferOperations;
pub use inplace_buffer_operations::InplaceBufferOperations;

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

pub(crate) fn empty_buffer(
    count: usize,
    flags: cl_mem_flags,
    opencl_state: &OpenCLState,
) -> Result<Buffer<cl_float>, ClError> {
    let buf = Buffer::create(&opencl_state.context, flags, count, ptr::null_mut())?;

    Ok(buf)
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