//! The module that contains the SoftMax activation function.

use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
};

use savefile_derive::Savefile;

use crate::layers::Layer;

const PROGRAM_SOURCE: &str = include_str!("kernels/softmax.cl");
const PROPAGATE_KERNEL_NAME: &str = "propagate";
const CALCULATE_EXPONENTIALS_KERNEL_NAME: &str = "calculate_exponentials";
const SUM_EXPONENTIALS_PER_SAMPLE_KERNEL_NAME: &str = "sum_exponentials_per_sample";
const CALCULATE_MAX_INPUT_PER_SAMPLE: &str = "calculate_max_input_per_sample";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

#[derive(Debug, Savefile)]
/// The SoftMax activation function, this function will squash its inputs in such a way that only
/// the numbers that are very close to the largest number be more "considered" than others.
/// It is good for classification problems because it is very rigid.
pub struct SoftMax<'a> {
    /// The amount of inputs this instance of TanH expects.
    pub inputs_amount: usize,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The cloned inputs last forward passed into this TaNH.
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The outputs that came out from the last forward pass into this TanH.
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL context used for managing OpenCL devices and queues.
    pub opencl_context: Option<&'a Context>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL queue, there exists one queue for each device,
    /// so currently Intricate does not have support for multiple devices
    /// doing computations on the data
    pub opencl_queue: Option<&'a CommandQueue>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL program for the SoftMax, this contains the kernsl (OpenCL GPU shaders)
    /// that will be needed for doing calculations with OpenCL
    pub opencl_program: Option<Program>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL kernel that will take the `exp` for each of the inputs.
    pub opencl_calculate_exponentials_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL kernel that will sum all of the exponentials calculated for each sample.
    pub opencl_sum_exponentials_per_sample_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL kernel that will find the largest input per sample.
    pub opencl_calculate_max_input_per_sample_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL kernel that will actually propagate through with the calculated data from other
    /// kernels and give the output of a forward pass into the SoftMax activation function.
    pub opencl_propagate_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The OpenCL kernel that will calculate the differentials of the loss with respect to each of
    /// the inputs given in a forward pass to this SoftMax.
    pub opencl_back_propagate_kernel: Option<Kernel>,
}

impl<'a> SoftMax<'a> {
    /// Creates a raw version of the SoftMax activation function, this is good for
    /// being used when you don't want to use the layer in a Model.
    pub fn new_raw(inputs_amount: usize) -> SoftMax<'a> {
        SoftMax {
            inputs_amount,
            opencl_context: None,
            opencl_queue: None,
            opencl_program: None,
            opencl_propagate_kernel: None,
            opencl_calculate_exponentials_kernel: None,
            opencl_back_propagate_kernel: None,
            opencl_sum_exponentials_per_sample_kernel: None,
            opencl_calculate_max_input_per_sample_kernel: None,
            last_outputs_buffer: None,
            last_inputs_buffer: None,
        }
    }

    /// Creates a ModelLayer version of the SotMax activation function, to be
    /// used with a Model.
    pub fn new(inputs_amount: usize) -> crate::types::ModelLayer<'a> {
        Self::new_raw(inputs_amount).into()
    }
}

impl<'a> Layer<'a> for SoftMax<'a> {
    fn init(
        &mut self,
        queue: &'a CommandQueue,
        context: &'a Context,
    ) -> Result<(), crate::types::CompilationOrOpenCLError> {
        let program =
            opencl3::program::Program::create_and_build_from_source(context, PROGRAM_SOURCE, "")?;

        let propagation_kernel = Kernel::create(&program, PROPAGATE_KERNEL_NAME)?;
        let exponentials_calculation_kernel =
            Kernel::create(&program, CALCULATE_EXPONENTIALS_KERNEL_NAME)?;
        let back_propagation_kernel = Kernel::create(&program, BACK_PROPAGATE_KERNEL_NAME)?;
        let sum_exponentials_per_sample_kernel =
            Kernel::create(&program, SUM_EXPONENTIALS_PER_SAMPLE_KERNEL_NAME)?;
        let calculate_max_input_per_sample_kernel =
            Kernel::create(&program, CALCULATE_MAX_INPUT_PER_SAMPLE)?;

        self.opencl_program = Some(program);
        self.opencl_propagate_kernel = Some(propagation_kernel);
        self.opencl_calculate_exponentials_kernel = Some(exponentials_calculation_kernel);
        self.opencl_sum_exponentials_per_sample_kernel = Some(sum_exponentials_per_sample_kernel);
        self.opencl_calculate_max_input_per_sample_kernel =
            Some(calculate_max_input_per_sample_kernel);
        self.opencl_back_propagate_kernel = Some(back_propagation_kernel);
        self.opencl_queue = Some(queue);
        self.opencl_context = Some(context);

        Ok(())
    }

    fn get_last_inputs(&self) -> Option<&Buffer<cl_float>> {
        self.last_inputs_buffer.as_ref()
    }

    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>> {
        self.last_outputs_buffer.as_ref()
    }

    fn get_inputs_amount(&self) -> usize {
        self.inputs_amount
    }

    fn get_outputs_amount(&self) -> usize {
        self.inputs_amount
    }

    fn clean_up_gpu_state(&mut self) -> () {
        if self.last_inputs_buffer.is_some() {
            drop(self.last_inputs_buffer.as_ref().unwrap());
        }

        if self.last_outputs_buffer.is_some() {
            drop(self.last_outputs_buffer.as_ref().unwrap());
        }
    }

    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError> {
        Ok(())
    }

    fn propagate(&mut self, inputs: &Buffer<cl_float>) -> Result<&Buffer<cl_float>, ClError> {
        assert!(self.opencl_context.is_some());
        assert!(self.opencl_queue.is_some());

        let context = self.opencl_context.unwrap();
        let queue = self.opencl_queue.unwrap();

        let inputs_size = inputs.size()?;
        let inputs_total_count = inputs_size / std::mem::size_of::<cl_float>();
        let samples_amount = inputs_total_count / self.inputs_amount;

        let mut copied_last_inputs_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            inputs_total_count,
            std::ptr::null_mut(),
        )?;

        // TODO: make copying this into the last inputs optional since this is only needed
        // for fitting a model as to make everything more optimized both in RAM usage and computation
        queue
            .enqueue_copy_buffer(
                inputs,
                &mut copied_last_inputs_buffer,
                0,
                0,
                inputs_size,
                &[],
            )?;

        queue.finish()?;

        self.last_inputs_buffer = Some(copied_last_inputs_buffer);

        let max_input_per_sample_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount,
            std::ptr::null_mut(),
        )?;

        ExecuteKernel::new(
            self.opencl_calculate_max_input_per_sample_kernel
                .as_ref()
                .unwrap(),
        )
        .set_arg(inputs)
        .set_arg(&max_input_per_sample_buffer)
        .set_arg(&(samples_amount as cl_int))
        .set_arg(&(self.inputs_amount as cl_int))
        .set_global_work_size(samples_amount)
        .enqueue_nd_range(self.opencl_queue.unwrap())?;

        queue.finish()?;

        let exponentials_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            inputs_total_count,
            std::ptr::null_mut(),
        )?;

        ExecuteKernel::new(self.opencl_calculate_exponentials_kernel.as_ref().unwrap())
            .set_arg(inputs)
            .set_arg(&exponentials_buffer)
            .set_arg(&max_input_per_sample_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .enqueue_nd_range(self.opencl_queue.unwrap())?;

        queue.finish()?;

        let exponentials_sum_per_sample = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount,
            std::ptr::null_mut(),
        )?;

        ExecuteKernel::new(
            self.opencl_sum_exponentials_per_sample_kernel
                .as_ref()
                .unwrap(),
        )
        .set_arg(&exponentials_buffer)
        .set_arg(&exponentials_sum_per_sample)
        .set_arg(&(samples_amount as cl_int))
        .set_arg(&(self.inputs_amount as cl_int))
        .set_global_work_size(samples_amount)
        .enqueue_nd_range(self.opencl_queue.unwrap())?;

        queue.finish()?;

        let outputs_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            inputs_total_count,
            std::ptr::null_mut(),
        )?;

        ExecuteKernel::new(self.opencl_propagate_kernel.as_ref().unwrap())
            .set_arg(&exponentials_buffer)
            .set_arg(&outputs_buffer)
            .set_arg(&exponentials_sum_per_sample)
            .set_arg(&(self.inputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .enqueue_nd_range(self.opencl_queue.as_ref().unwrap())?;

        queue.finish()?;

        self.last_outputs_buffer = Some(outputs_buffer);

        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &opencl3::memory::Buffer<opencl3::device::cl_float>,
        _: opencl3::device::cl_float,
    ) -> Result<
        Option<opencl3::memory::Buffer<opencl3::device::cl_float>>,
        opencl3::error_codes::ClError,
    > {
        if should_calculate_input_to_error_derivative {
            assert!(self.opencl_context.is_some());
            assert!(self.opencl_queue.is_some());

            let samples_amount = self.last_outputs_buffer.as_ref().unwrap().size()?
                / self.inputs_amount
                / std::mem::size_of::<opencl3::device::cl_float>();

            assert_eq!(samples_amount % 1, 0);

            let context = self.opencl_context.unwrap();
            let queue = self.opencl_queue.unwrap();

            let loss_to_input_derivatives_buffer =
                opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                    context,
                    opencl3::memory::CL_MEM_READ_WRITE,
                    self.inputs_amount * samples_amount,
                    std::ptr::null_mut(),
                )?;

            opencl3::kernel::ExecuteKernel::new(
                self.opencl_back_propagate_kernel.as_ref().unwrap(),
            )
            .set_arg(layer_output_to_error_derivative)
            .set_arg(self.last_outputs_buffer.as_ref().unwrap())
            .set_arg(&loss_to_input_derivatives_buffer)
            .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
            .set_arg(&(samples_amount as opencl3::error_codes::cl_int))
            .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .enqueue_nd_range(queue)?;

            queue.finish()?;

            Ok(Some(loss_to_input_derivatives_buffer))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod softmax_tests {
    use std::f32::consts::E;

    use opencl3::{
        command_queue::CL_BLOCKING,
        device::cl_float,
        memory::{Buffer, CL_MEM_READ_ONLY},
    };
    use rand::{thread_rng, Rng};

    use crate::{
        layers::Layer,
        utils::{approx_eq::assert_approx_equal_distance, opencl::DeviceType, setup_opencl},
    };

    use super::SoftMax;

    #[test]
    fn should_propagate_to_correct_values() {
        let samples_amount = 123;
        let numbers_amount = 19;

        let mut rng = thread_rng();

        let inputs: Vec<Vec<f32>> = (0..samples_amount)
            .map(|_| {
                (0..numbers_amount)
                    .map(|_| rng.gen_range(0.0_f32..10.93_f32))
                    .collect()
            })
            .collect();

        let expected_outputs: Vec<Vec<f32>> = inputs
            .iter()
            .map(|inputs| {
                let max = inputs.iter().copied().fold(f32::NAN, f32::max);
                let exponentials: Vec<f32> = inputs.iter().map(|x| E.powf(x - max)).collect();
                let exponential_sum: f32 = exponentials.iter().sum::<f32>();
                exponentials
                    .iter()
                    .map(|exponential| exponential / exponential_sum)
                    .collect()
            })
            .collect();

        let opencl_state = setup_opencl(DeviceType::CPU).unwrap();

        let mut softmax = SoftMax::new(numbers_amount);
        softmax
            .init(&opencl_state.queue, &opencl_state.context)
            .unwrap();

        let mut inputs_buffer = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        opencl_state
            .queue
            .enqueue_write_buffer(
                &mut inputs_buffer,
                CL_BLOCKING,
                0,
                inputs
                    .iter()
                    .map(|v| v.to_vec())
                    .flatten()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &[],
            )
            .unwrap();

        opencl_state.queue.finish().unwrap();

        let outputs_buffer = softmax.propagate(&inputs_buffer).unwrap();

        let mut actual_outputs = vec![0.0; samples_amount * numbers_amount];

        opencl_state
            .queue
            .enqueue_read_buffer(
                &outputs_buffer,
                CL_BLOCKING,
                0,
                actual_outputs.as_mut_slice(),
                &[],
            )
            .unwrap();

        opencl_state.queue.finish().unwrap();

        assert_approx_equal_distance(
            &actual_outputs,
            &expected_outputs
                .iter()
                .map(|v| v.to_vec())
                .flatten()
                .collect(),
            0.05,
        );
    }
}