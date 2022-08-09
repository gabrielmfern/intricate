pub mod matrix_operations;
pub mod vector_operations;
pub mod approx_eq;

pub mod opencl;
pub use opencl::GpuSummable;
use opencl3::{
    error_codes::ClError,
    device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU},
    context::Context,
    command_queue::CommandQueue,
    
};

pub fn gcd(n: usize, m: usize) -> usize {
    let mut largest_divisor: usize = 1;
    let middle_point: usize = (n.min(m) as f32).sqrt().floor() as usize;

    for k in (1..=middle_point).rev() {
        if n % k == 0 && m % k == 0 {
            largest_divisor = k;
            break;
        }
    }

    largest_divisor
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct OpenCLState {
    pub context: Context,
    pub queue: CommandQueue,
    pub device: Device
}

pub fn setup_opencl() -> Result<OpenCLState, ClError> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
    let first_gpu = Device::new(device_ids[0]);
    let context = Context::from_device(&first_gpu)?;
    // here it can be activated to make profiling on kernels
    let queue = CommandQueue::create_with_properties(&context, first_gpu.id(), 0, 0)?;

    Ok(OpenCLState {
        context, 
        queue, 
        device: first_gpu
    })
} 