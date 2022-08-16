use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::cl_float,
    kernel::Kernel,
    memory::Buffer,
    program::Program,
};

use intricate_macros::ActivationLayer;

use savefile_derive::Savefile;

const PROGRAM_SOURCE: &str = include_str!("kernels/relu.cl");
const PROPAGATE_KERNEL_NAME: &str = "propagate";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

#[derive(Debug, Savefile, ActivationLayer)]
pub struct ReLU<'a> {
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

#[cfg(test)]
mod relu_tests {
    use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY}, device::cl_float, command_queue::CL_BLOCKING};
    use rand::{thread_rng, Rng};

    use crate::{utils::{setup_opencl, opencl::DeviceType, approx_eq::assert_approx_equal_distance}, layers::Layer};

    use super::ReLU;

    #[test]
    fn should_propagate_to_correct_values() {
        let samples_amount = 30;
        let numbers_amount = 20;

        let mut rng = thread_rng();

        let inputs: Vec<f32> = (0..(samples_amount * numbers_amount)).map(|_| {
            rng.gen_range(-1234.41_f32..51312.93_f32)
        }).collect();

        let expected_outputs: Vec<f32> = inputs.iter().map(|input| input.max(0.0)).collect();

        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut relu = ReLU::new(numbers_amount);
        relu.init(&opencl_state.queue, &opencl_state.context).unwrap();

        let mut inputs_buffer = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * numbers_amount,
            std::ptr::null_mut(),
        ).unwrap();

        opencl_state.queue.enqueue_write_buffer(
            &mut inputs_buffer,
            CL_BLOCKING,
            0,
            inputs.as_slice(),
            &[],
        ).unwrap().wait().unwrap();

        let outputs_buffer = relu.propagate(&inputs_buffer).unwrap();

        let mut actual_outputs = vec![0.0; samples_amount * numbers_amount];
        
        opencl_state.queue.enqueue_read_buffer(
            &outputs_buffer,
            CL_BLOCKING,
            0,
            actual_outputs.as_mut_slice(),
            &[]
        ).unwrap().wait().unwrap();

        assert_approx_equal_distance(&actual_outputs, &expected_outputs, 0.05);
    }
}
