//! The module that contains the SoftMax activation function.

use opencl3::{
    device::cl_float,
    error_codes::cl_int,
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
};

use savefile_derive::Savefile;

use crate::{
    layers::{
        Gradient, Layer, LayerLossToInputDifferentiationError, LayerPropagationError,
        SyncDataError, ParametersOptimizationError, LayerInitializationError,
    },
    utils::{
        opencl::{empty_buffer, ensure_program, BufferOperations, EnsureKernelsAndProgramError},
        OpenCLState,
    }, optimizers::Optimizer,
};

const PROGRAM_NAME: &str = "SOFTMAX";
const PROGRAM_SOURCE: &str = include_str!("kernels/softmax.cl");
const PROPAGATE_KERNEL_NAME: &str = "propagate";

const CALCULATE_EXPONENTIALS_KERNEL_NAME: &str = "calculate_exponentials";
const SUM_EXPONENTIALS_PER_SAMPLE_KERNEL_NAME: &str = "sum_exponentials_per_sample";
const FIND_MAX_INPUT_PER_SAMPLE_KERNEL_NAME: &str = "calculate_max_input_per_sample";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

pub(crate) fn compile_softmax(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[
        PROPAGATE_KERNEL_NAME.to_string(),
        CALCULATE_EXPONENTIALS_KERNEL_NAME.to_string(),
        SUM_EXPONENTIALS_PER_SAMPLE_KERNEL_NAME.to_string(),
        FIND_MAX_INPUT_PER_SAMPLE_KERNEL_NAME.to_string(),
        BACK_PROPAGATE_KERNEL_NAME.to_string(),
    ];

    ensure_program(
        opencl_state,
        PROGRAM_NAME.to_string(),
        PROGRAM_SOURCE.to_string(),
        "".to_string(),
        kernels,
    )?;

    Ok(())
}

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
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> SoftMax<'a> {
    /// Creates a raw version of the SoftMax activation function, this is good for
    /// being used when you don't want to use the layer in a Model.
    pub fn new_raw(inputs_amount: usize) -> SoftMax<'a> {
        SoftMax {
            inputs_amount,

            last_outputs_buffer: None,
            last_inputs_buffer: None,

            opencl_state: None,
        }
    }

    /// Creates a ModelLayer version of the SotMax activation function, to be
    /// used with a Model.
    pub fn new(inputs_amount: usize) -> crate::types::ModelLayer<'a> {
        Self::new_raw(inputs_amount).into()
    }
}

impl<'a> Layer<'a> for SoftMax<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), LayerInitializationError> {
        self.opencl_state = Some(opencl_state);

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

    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), SyncDataError> {
        Ok(())
    }

    fn propagate(
        &mut self,
        inputs: &Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
        if self.opencl_state.is_none() {
            return Err(LayerPropagationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.len() == 0 {
            return Err(LayerPropagationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        let inputs_size = inputs.size()?;
        let inputs_total_count = inputs_size / std::mem::size_of::<cl_float>();

        if inputs_total_count % self.inputs_amount != 0 {
            return Err(LayerPropagationError::InputsDontMatchExpectedShape);
        }

        let samples_amount = inputs_total_count / self.inputs_amount;

        let copied_last_inputs_buffer = inputs.clone(CL_MEM_READ_ONLY, state)?;

        self.last_inputs_buffer = Some(copied_last_inputs_buffer);

        let max_input_per_sample_buffer = empty_buffer(samples_amount, CL_MEM_READ_WRITE, state)?;

        let program = state.get_prgm(PROGRAM_NAME)?;

        let max_input_per_sample_kernel =
            program.get_krnl(FIND_MAX_INPUT_PER_SAMPLE_KERNEL_NAME)?;

        let find_max_input_event = ExecuteKernel::new(max_input_per_sample_kernel)
            .set_arg(inputs)
            .set_arg(&max_input_per_sample_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_size(samples_amount)
            .enqueue_nd_range(queue)?;

        let exponentials_buffer = empty_buffer(inputs_total_count, CL_MEM_READ_WRITE, state)?;

        let calculate_exponentials_kernel = program.get_krnl(CALCULATE_EXPONENTIALS_KERNEL_NAME)?;

        let calculate_exponentials_event = ExecuteKernel::new(calculate_exponentials_kernel)
            .set_arg(inputs)
            .set_arg(&exponentials_buffer)
            .set_arg(&max_input_per_sample_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .set_wait_event(&find_max_input_event)
            .enqueue_nd_range(queue)?;

        let exponentials_sum_per_sample = empty_buffer(samples_amount, CL_MEM_READ_WRITE, state)?;

        let sum_exponentials_kernel = program.get_krnl(SUM_EXPONENTIALS_PER_SAMPLE_KERNEL_NAME)?;

        let sum_exponentials_event = ExecuteKernel::new(sum_exponentials_kernel)
            .set_arg(&exponentials_buffer)
            .set_arg(&exponentials_sum_per_sample)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_size(samples_amount)
            .set_wait_event(&calculate_exponentials_event)
            .enqueue_nd_range(queue)?;

        let outputs_buffer = empty_buffer(inputs_total_count, CL_MEM_READ_WRITE, state)?;

        let propagate_kernel = program.get_krnl(PROPAGATE_KERNEL_NAME)?;

        ExecuteKernel::new(propagate_kernel)
            .set_arg(&exponentials_buffer)
            .set_arg(&outputs_buffer)
            .set_arg(&exponentials_sum_per_sample)
            .set_arg(&(self.inputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .set_wait_event(&sum_exponentials_event)
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        self.last_outputs_buffer = Some(outputs_buffer);

        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn apply_gradients(
        &mut self,
        _per_parameter_type_gradients: &[Gradient],
        _optimizer: &mut dyn Optimizer<'a>,
        _layer_index: usize,
    ) -> Result<(), crate::layers::LayerGradientApplicationError> {
        Ok(())
    }

    fn compute_gradients(
        &self,
        _layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Vec<Gradient>, crate::layers::LayerGradientComputationError> {
        Ok(Vec::default())
    }

    fn optimize_parameters(
        &mut self,
        _optimizer: &dyn Optimizer<'a>,
        _layer_index: usize,
    ) -> Result<(), ParametersOptimizationError> {
        Ok(())
    }

    fn compute_loss_to_input_derivatives(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, LayerLossToInputDifferentiationError> {
        if self.opencl_state.is_none() {
            return Err(LayerLossToInputDifferentiationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.len() == 0 {
            return Err(LayerLossToInputDifferentiationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        let outputs_size = self.last_outputs_buffer.as_ref().unwrap().size()?;
        let outputs_total_count = outputs_size / std::mem::size_of::<cl_float>();
        if outputs_total_count % self.inputs_amount != 0 {
            return Err(LayerLossToInputDifferentiationError::DerivativesDontMatchExpectedShape);
        }
        let samples_amount = outputs_total_count / self.inputs_amount;

        let loss_to_input_derivatives_buffer = empty_buffer(
            self.inputs_amount * samples_amount,
            CL_MEM_READ_WRITE,
            state,
        )?;

        let program = state.get_prgm(PROGRAM_NAME)?;
        let backprop_kernel = program.get_krnl(BACK_PROPAGATE_KERNEL_NAME)?;

        opencl3::kernel::ExecuteKernel::new(backprop_kernel)
            .set_arg(layer_output_to_error_derivative)
            .set_arg(self.last_outputs_buffer.as_ref().unwrap())
            .set_arg(&loss_to_input_derivatives_buffer)
            .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
            .set_arg(&(samples_amount as opencl3::error_codes::cl_int))
            .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        Ok(loss_to_input_derivatives_buffer)
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
    fn should_calculate_loss_to_input_derivatives_correctly() {
        let samples_amount = 1;
        let numbers_amount = 19;

        let mut rng = thread_rng();
        let loss_to_output_derivatives: Vec<Vec<f32>> = (0..samples_amount)
            .map(|_| {
                (0..numbers_amount)
                    .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
                    .collect()
            })
            .collect();
        let last_outputs: Vec<Vec<f32>> = (0..samples_amount)
            .map(|_| {
                (0..numbers_amount)
                    .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
                    .collect()
            })
            .collect();

        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut softmax = SoftMax::new_raw(numbers_amount);
        softmax.init(&opencl_state).unwrap();

        let mut loss_to_output_derivatives_buffer = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        let mut last_outputs_buffer = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut last_outputs_buffer,
                CL_BLOCKING,
                0,
                last_outputs
                    .iter()
                    .map(|v| v.to_vec())
                    .flatten()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        queue
            .enqueue_write_buffer(
                &mut loss_to_output_derivatives_buffer,
                CL_BLOCKING,
                0,
                loss_to_output_derivatives
                    .iter()
                    .map(|v| v.to_vec())
                    .flatten()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        softmax.last_outputs_buffer = Some(last_outputs_buffer);

        let expected_loss_to_input_derivatives: Vec<Vec<f32>> = (0..samples_amount)
            .map(|sample_index| {
                (0..numbers_amount)
                    .map(|input_index| {
                        let input_associated_output = last_outputs[sample_index][input_index];
                        last_outputs[sample_index]
                            .iter()
                            .enumerate()
                            .map(|(output_index, output)| {
                                let output_to_input_derivative;
                                if input_index == output_index {
                                    output_to_input_derivative = output * (1.0 - output);
                                } else {
                                    output_to_input_derivative = -input_associated_output * output;
                                }

                                output_to_input_derivative
                                    * loss_to_output_derivatives[sample_index][output_index]
                            })
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect();
        let loss_to_input_derivatives_buffer = softmax
            .compute_loss_to_input_derivatives(&loss_to_output_derivatives_buffer)
            .unwrap();

        let mut loss_to_input_derivatives = vec![0.0; samples_amount * numbers_amount];

        queue
            .enqueue_read_buffer(
                &loss_to_input_derivatives_buffer,
                CL_BLOCKING,
                0,
                loss_to_input_derivatives.as_mut_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        queue.finish().unwrap();

        println!("GPU's dE/dI: {:?}", loss_to_input_derivatives);
        println!("\nCPU's dE/dI: {:?}", expected_loss_to_input_derivatives);

        loss_to_input_derivatives
            .iter()
            .zip(expected_loss_to_input_derivatives.iter().flatten())
            .for_each(|(x, y)| {
                assert!(((x - y) / x.max(*y)).abs() <= 0.01);
            });
    }

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

        let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

        let mut softmax = SoftMax::new_raw(numbers_amount);
        softmax.init(&opencl_state).unwrap();

        let mut inputs_buffer = Buffer::<cl_float>::create(
            &opencl_state.context,
            CL_MEM_READ_ONLY,
            samples_amount * numbers_amount,
            std::ptr::null_mut(),
        )
        .unwrap();

        let queue = opencl_state.queues.first().unwrap();

        queue
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
            .unwrap()
            .wait()
            .unwrap();

        let outputs_buffer = softmax.propagate(&inputs_buffer).unwrap();

        let mut actual_outputs = vec![0.0; samples_amount * numbers_amount];

        queue
            .enqueue_read_buffer(
                &outputs_buffer,
                CL_BLOCKING,
                0,
                actual_outputs.as_mut_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

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