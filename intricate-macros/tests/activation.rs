use intricate_macros::ActivationLayer;

use opencl3::{
    command_queue::CommandQueue, context::Context, device::cl_float, kernel::Kernel,
    memory::Buffer, program::Program,
};
#[allow(dead_code)]
use savefile_derive::Savefile;

const PROGRAM_SOURCE: &str = "";
const PROPAGATE_KERNEL_NAME: &str = "propagate";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

// Here the only expected error is that Softmax is not included in ModelLayer
#[derive(Debug, Savefile, ActivationLayer)]
pub struct Softmax<'a> {
    pub inputs_amount: usize,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_context: Option<&'a Context>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_queue: Option<&'a CommandQueue>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_program: Option<Program>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_propagate_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_back_propagate_kernel: Option<Kernel>,
}

fn main() {}
