//! The module that defines the Dense layer.

use opencl3::{
    command_queue::CL_NON_BLOCKING,
    device::cl_float,
    error_codes::{cl_int, ClError},
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use savefile_derive::Savefile;
use std::mem;
use std::ptr;

use crate::{
    optimizers::Optimizer,
    types::ModelLayer,
    utils::{
        opencl::{empty_buffer, ensure_program, EnsureKernelsAndProgramError},
        OpenCLState,
    },
};

use super::{
    Gradient, Gradients, Layer, LayerGradientApplicationError, LayerGradientComputationError,
    LayerLossToInputDifferentiationError, LayerPropagationError, LayerSyncDataError,
};

const DENSE_PROP_PROGRAM_NAME: &str = "DENSE_PROPAGATION";
const DENSE_BACKPROP_PROGRAM_NAME: &str = "DENSE_BACKPROPAGATION";

const PROPAGATION_PROGRAM_SORUCE: &str = include_str!("kernels/dense_propagation.cl");
const BACK_PROPAGATION_PROGRAM_SOURCE: &str = include_str!("kernels/dense_back_propagation.cl");

const PROPAGATION_KERNEL_NAME: &str = "dense_propagate";

const WEIGHTS_GRADIENT_COMPUTATION_KERNEL_NAME: &str = "weights_gradient_calculation";
const BIAS_GRADIENT_APPLICATION_KERNEL_NAME: &str = "bias_gradient_calculation";
const LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME: &str =
    "compute_loss_derivative_with_respect_to_inputs";

pub(crate) fn compile_dense(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let prop_kernels = &[PROPAGATION_KERNEL_NAME.to_string()];
    let backprop_kernels = &[
        WEIGHTS_GRADIENT_COMPUTATION_KERNEL_NAME.to_string(),
        BIAS_GRADIENT_APPLICATION_KERNEL_NAME.to_string(),
        LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME.to_string(),
    ];

    ensure_program(
        opencl_state,
        DENSE_PROP_PROGRAM_NAME.to_string(),
        PROPAGATION_PROGRAM_SORUCE.to_string(),
        "".to_string(),
        prop_kernels,
    )?;
    ensure_program(
        opencl_state,
        DENSE_BACKPROP_PROGRAM_NAME.to_string(),
        BACK_PROPAGATION_PROGRAM_SOURCE.to_string(),
        "".to_string(),
        backprop_kernels,
    )?;

    Ok(())
}

#[derive(Debug, Savefile)]
/// A densely connected layer, this layer consists of some inputs
/// and the weights that connect each input to all outputs,
/// its propagation results in a dot product between these weights
/// and the inputs received in the propagation method
/// added with some biases that are trainable on backprop
///
/// # Examples
///
/// ```
/// use intricate::layers::Dense;
///
/// let my_layer: Dense = Dense::new_raw(5, 5);
/// ```
pub struct Dense<'a> {
    /// The expected inputs to this Dense layer.
    pub inputs_amount: usize,
    /// The expected outputs to this Dense layer.
    pub outputs_amount: usize,

    /// The weights of this Dense layer, but stored in the CPU instead of in a OpenCL buffer.
    pub weights: Vec<Vec<f32>>,
    /// The biases of this Dense layer, but stored in the CPU instead of in a OpenCL buffer.
    pub biases: Vec<f32>, // TODO: make biases optional

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The allocated buffer with OpenCL that contains the flattened weights of this Dense layer.
    pub weights_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The allocated buffer with OpenCL that contains the biases of this Dense layer.
    pub biases_buffer: Option<Buffer<cl_float>>,

    // Had to take a choice with this, not having a reference here
    // needs to be unless there needs to be unsafe code in the Model
    // so duplicating things in the RAM is better off than perhaps having
    // some memory errors that would be extremely hard to debug
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The buffer that contains the flattened inputs per sample that were last forwad passed into
    /// this Dense layer.
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The buffer that contains the flattened outputs per sample that last came out of a forward
    /// pass into this Dense layer.
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Dense<'a> {
    /// Creates a new Dense layer but without being inside of the ModelLayer enum.
    pub fn new_raw(inputs_amount: usize, outputs_amount: usize) -> Dense<'a> {
        let mut rng = rand::thread_rng(); //                much more convenient

        let weights = (0..inputs_amount)
            .into_iter()
            .map(|_| {
                (0..outputs_amount)
                    .into_iter()
                    .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let biases = (0..outputs_amount)
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
            .collect::<Vec<f32>>();

        Dense {
            inputs_amount,
            outputs_amount,

            weights,
            biases,

            weights_buffer: None,
            biases_buffer: None,

            last_inputs_buffer: None,
            last_outputs_buffer: None,

            opencl_state: None,
        }
        .into() // because ModelLayer implements From<Dense>
    }

    /// Creates a new Dense layer with random weights and biases and empty OpenCL values.
    pub fn new(inputs_amount: usize, outputs_amount: usize) -> ModelLayer<'a> {
        Self::new_raw(inputs_amount, outputs_amount).into()
    }
}

pub struct DenseGradients<'a> {
    opencl_state: &'a OpenCLState,
    weights_gradients: Buffer<cl_float>,
    bias_gradients: Buffer<cl_float>,
}

impl<'a> Gradients<'a> for DenseGradients<'a> {
    fn get_gradients(&self) -> &[Gradient] {
        return &[
            Gradient {
                value: self.weights_gradients,
                optimizable: true,
            },
            Gradient {
                value: self.bias_gradients,
                optimizable: true,
            },
        ];
    }

    fn get_opencl_state(&self) -> &'a OpenCLState {
        self.opencl_state
    }
}

impl<'a> Layer<'a, DenseGradients<'a>> for Dense<'a> {
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
        self.outputs_amount
    }

    fn clean_up_gpu_state(&mut self) -> () {
        if self.weights_buffer.is_some() {
            drop(self.weights_buffer.as_ref().unwrap());
        }

        if self.biases_buffer.is_some() {
            drop(self.biases_buffer.as_ref().unwrap());
        }

        if self.last_inputs_buffer.is_some() {
            drop(self.last_inputs_buffer.as_ref().unwrap());
        }

        if self.last_outputs_buffer.is_some() {
            drop(self.last_outputs_buffer.as_ref().unwrap());
        }
    }

    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), LayerSyncDataError> {
        if self.weights_buffer.is_none() {
            Err(LayerSyncDataError::NotAllocatedInDevice {
                field_name: "weights_buffer".to_string(),
            })
        }

        if self.biases_buffer.is_none() {
            Err(LayerSyncDataError::NotAllocatedInDevice {
                field_name: "biases_buffer".to_string(),
            })
        }

        if self.opencl_state.is_none {
            Err(LayerSyncDataError::LayerNotInitialized)
        }

        if self.opencl_state.unwrap().queues.is_empty() {
            Err(LayerSyncDataError::NoCommandQueue)
        }

        let mut weights_flat = vec![0.0; self.inputs_amount * self.outputs_amount];
        let mut biases = vec![0.0; self.outputs_amount];

        let queue = self.opencl_state.unwrap().queues.first().unwrap();

        let read_weights_event = queue.enqueue_read_buffer(
            self.weights_buffer.as_ref().unwrap(),
            CL_NON_BLOCKING,
            0,
            weights_flat.as_mut_slice(),
            &[],
        )?;

        let read_biases_event = queue.enqueue_read_buffer(
            self.biases_buffer.as_ref().unwrap(),
            CL_NON_BLOCKING,
            0,
            biases.as_mut_slice(),
            &[],
        )?;

        read_weights_event.wait()?;
        read_biases_event.wait()?;

        self.biases = biases;
        self.weights = (0..self.inputs_amount)
            .into_par_iter()
            .map(|i| {
                let row_part = i * self.outputs_amount;
                (0..self.outputs_amount)
                    .into_iter()
                    .map(|j| {
                        let flat_index = row_part + j;
                        weights_flat[flat_index]
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        Ok(())
    }

    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError> {
        assert!(!self.weights.is_empty());
        assert!(!self.biases.is_empty());
        assert!(!opencl_state.queues.is_empty());

        let context = &opencl_state.context;

        let mut weights_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.inputs_amount * self.outputs_amount,
            ptr::null_mut(),
        )?;
        let mut biases_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.outputs_amount,
            ptr::null_mut(),
        )?;

        let queue = opencl_state.queues.first().unwrap();

        queue
            .enqueue_write_buffer(
                &mut weights_buffer,
                CL_NON_BLOCKING,
                0,
                self.weights
                    .par_iter()
                    .map(|x| x.to_vec())
                    .flatten()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &[],
            )?
            .wait()?;
        queue
            .enqueue_write_buffer(
                &mut biases_buffer,
                CL_NON_BLOCKING,
                0,
                self.biases.as_slice(),
                &[],
            )?
            .wait()?;

        self.weights_buffer = Some(weights_buffer);
        self.biases_buffer = Some(biases_buffer);

        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn propagate(
        &mut self,
        input_samples: &Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
        if self.opencl_state.is_none() {
            return Err(LayerPropagationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerPropagationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();
        let context = &state.context;

        let inputs_size = input_samples.size()?;
        let inputs_total_count = inputs_size / mem::size_of::<cl_float>();

        let mut copied_last_inputs_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_ONLY,
            inputs_total_count,
            ptr::null_mut(),
        )?;

        // TODO: make copying this into the last inputs optional since this is only needed
        // for fitting a model as to make everything more optimized both in RAM usage and computation
        queue.enqueue_copy_buffer(
            input_samples,
            &mut copied_last_inputs_buffer,
            0,
            0,
            inputs_size,
            &[],
        )?;

        self.last_inputs_buffer = Some(copied_last_inputs_buffer);

        let samples_amount =
            input_samples.size()? / self.inputs_amount / mem::size_of::<cl_float>();

        let outputs_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.outputs_amount * samples_amount,
            ptr::null_mut(),
        )?;

        if !state.programs.contains_key(DENSE_PROP_PROGRAM_NAME) {
            return Err(LayerPropagationError::ProgramNotFound(
                DENSE_PROP_PROGRAM_NAME,
            ));
        }

        let program = state.programs.get(DENSE_PROP_PROGRAM_NAME).unwrap();

        if !program.kernels.contains_key(PROPAGATION_KERNEL_NAME) {
            return Err(LayerPropagationError::KernelNotFound(
                PROPAGATION_KERNEL_NAME,
            ));
        }

        let kernel = program.kernels.get(PROPAGATION_KERNEL_NAME).unwrap();

        ExecuteKernel::new(kernel)
            .set_arg(input_samples)
            .set_arg(self.biases_buffer.as_ref().unwrap())
            .set_arg(self.weights_buffer.as_ref().unwrap())
            .set_arg(&outputs_buffer)
            .set_arg(&(self.inputs_amount as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.outputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.outputs_amount])
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        self.last_outputs_buffer = Some(outputs_buffer);
        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn compute_gradients(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<DenseGradients<'a>, LayerGradientComputationError> {
        if self.opencl_state.is_none() {
            return Err(LayerGradientComputationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerGradientComputationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        if !state.programs.contains_key(DENSE_BACKPROP_PROGRAM_NAME) {
            return Err(LayerGradientComputationError::ProgramNotFound(
                DENSE_BACKPROP_PROGRAM_NAME,
            ));
        }

        let backprop_program = state.programs.get(DENSE_BACKPROP_PROGRAM_NAME).unwrap();

        if !backprop_program
            .kernels
            .contains_key(WEIGHTS_GRADIENT_COMPUTATION_KERNEL_NAME)
        {
            return Err(LayerGradientComputationError::KernelNotFound(
                WEIGHTS_GRADIENT_COMPUTATION_KERNEL_NAME,
            ));
        }

        let weights_gradient_computation_kernel = backprop_program
            .kernels
            .get(WEIGHTS_GRADIENT_COMPUTATION_KERNEL_NAME)
            .unwrap();

        if !backprop_program
            .kernels
            .contains_key(BIAS_GRADIENT_APPLICATION_KERNEL_NAME)
        {
            return Err(LayerGradientComputationError::KernelNotFound(
                BIAS_GRADIENT_APPLICATION_KERNEL_NAME,
            ));
        }

        let bias_gradient_computation_kernel = backprop_program
            .kernels
            .get(BIAS_GRADIENT_APPLICATION_KERNEL_NAME)
            .unwrap();

        let weights_gradients = empty_buffer(
            self.inputs_amount * self.outputs_amount,
            CL_MEM_READ_WRITE,
            self.opencl_state,
        )?;
        let bias_gradients =
            empty_buffer(self.outputs_amount, CL_MEM_READ_WRITE, self.opencl_state)?;

        let samples_amount = layer_output_to_error_derivative.size()?
            / self.outputs_amount
            / mem::size_of::<cl_float>();

        let weight_gradients_event = ExecuteKernel::new(weights_gradient_computation_kernel)
            .set_arg(layer_output_to_error_derivative)
            .set_arg(self.last_inputs_buffer.as_ref().unwrap())
            .set_arg(&weights_gradients)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.outputs_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_sizes(&[self.inputs_amount, self.outputs_amount])
            .enqueue_nd_range(queue)?;

        let bias_gradients_event = ExecuteKernel::new(bias_gradient_computation_kernel)
            .set_arg(layer_output_to_error_derivative)
            .set_arg(&bias_gradients)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.outputs_amount as cl_int))
            .set_wait_event(&weight_gradients_event)
            .set_global_work_size(self.outputs_amount)
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        Ok(DenseGradients {
            opencl_state: state,
            weights_gradients,
            bias_gradients,
        })
    }

    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: DenseGradients<'a>,
        optimizer: dyn Optimizer,
    ) -> Result<(), LayerGradientApplicationError> {
        let update_vectors = per_parameter_type_gradients.compute_update_vectors(optimizer)?;

        let weights_buffer = self.weights_buffer.unwrap();
        let biases_buffer = self.biases_buffer.unwrap();

        weights_buffer.subtract(update_vectors[0])?;
        biases_buffer.subtract(update_vectors[1])?;

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
            return Err(LayerLossToInputDifferentiationError::NoCommandQueue);
        }

        let queue = state.queues.first().unwrap();

        if !state.programs.contains_key(DENSE_BACKPROP_PROGRAM_NAME) {
            return Err(LayerLossToInputDifferentiationError::ProgramNotFound(
                DENSE_BACKPROP_PROGRAM_NAME,
            ));
        }

        let program = state.programs.get(DENSE_BACKPROP_PROGRAM_NAME).unwrap();

        if !program
            .kernels
            .contains_key(LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME)
        {
            return Err(LayerLossToInputDifferentiationError::KernelNotFound(
                LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME,
            ));
        }

        let kernel = program
            .kernels
            .get(LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME)
            .unwrap();

        let samples_amount = layer_output_to_error_derivative.size()? / mem::size_of::<cl_float>();
        let loss_to_input_derivatives = empty_buffer(samples_amount, CL_MEM_READ_WRITE, state)?;

        ExecuteKernel::new(kernel)
            .set_arg(self.weights_buffer.as_ref().unwrap())
            .set_arg(layer_output_to_error_derivative)
            .set_arg(&loss_to_input_derivatives)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(self.outputs_amount as cl_int))
            .set_arg(&(self.inputs_amount as cl_int))
            .set_global_work_sizes(&[samples_amount, self.inputs_amount])
            .enqueue_nd_range(queue);

        queue.finish()?;

        Ok(())
    }
}

#[cfg(test)]
mod dense_tests {
    use std::ptr;

    use opencl3::{
        command_queue::{CL_BLOCKING, CL_NON_BLOCKING},
        device::cl_float,
        memory::{Buffer, CL_MEM_READ_ONLY},
    };
    use rand::{thread_rng, Rng};

    use crate::{
        layers::{dense::Dense, Layer},
        types::CompilationOrOpenCLError,
        utils::{
            opencl::{BufferLike, DeviceType},
            setup_opencl,
        },
    };

    #[test]
    fn should_apply_gradients_correctly() -> () {
        let state = setup_opencl(DeviceType::GPU).unwrap();

        let queue = state.queues.first().unwrap();
        let context = &state.context;

        let inputs_amount = 500;
        let outputs_amount = 500;

        let mut gpu_dense = Dense::new_raw(inputs_amount, outputs_amount);
        gpu_dense.init(&state).unwrap();

        let mut rng = thread_rng();
        let loss_to_output_derivatives: Vec<f32> = (0..outputs_amount)
            .map(|_| rng.gen_range(-134_f32..314_f32))
            .collect();

        let inputs: Vec<f32> = (0..inputs_amount)
            .map(|_| rng.gen_range(-134_f32..314_f32))
            .collect();

        let expected_gradients: Vec<Vec<f32>> = (0..inputs_amount)
            .map(|input_index| {
                (0..outputs_amount)
                    .map(|output_index| {
                        let loss_to_output_derivative = loss_to_output_derivatives[output_index];
                        let input = inputs[input_index];

                        loss_to_output_derivative * input
                    })
                    .collect()
            })
            .collect();

        let expected_bias_gradients: Vec<f32> = loss_to_output_derivatives.to_vec();

        let mut input_samples_buffer = inputs.to_buffer(CL_MEM_READ_ONLY, true, &state).unwrap();
        gpu_dense.last_inputs_buffer = Some(input_samples_buffer);

        let mut loss_to_output_derivatives_buffer = loss_to_output_derivatives
            .to_buffer(CL_MEM_READ_ONLY, true, &state)
            .unwrap();

        let actual_gradients = gpu_dense
            .compute_gradients(&loss_to_output_derivatives_buffer)
            .unwrap();

        let flat_actual_weights_gradients =
            Vec::<f32>::from_buffer(&actual_gradients.weights_gradients, true, &state).unwrap();

        let actual_weights_gradients: Vec<Vec<f32>> = (0..inputs_amount).map(|input_index| {
            (0..outputs_amount).map(|output_index| {
                let i = input_index * outputs_amount + output_index;

                flat_actual_weights_gradients[i]
            }).collect()
        }).collect();
        let actual_bias_gradients =
            Vec::<f32>::from_buffer(&actual_gradients.bias_gradients, true, &state).unwrap();

        let max_dist = 0.01;

        {
            expected_gradients.iter().zip(actual_weights_gradients).for_each(|(input_to_output_gradients, actual_input_to_output_gradients)| {
                input_to_output_gradients.iter().zip(actual_input_to_output_gradients).for_each(|(expected_gradient, gradient)| {
                    assert!((expected_gradient - gradient).abs() / expected_gradient.max(gradient) <= 0.0001);
                });
            });
        };

        {
            expected_bias_gradients.iter().zip(actual_bias_gradients).for_each(|(expected_bias, bias)| {
                assert!((expected_bias - bias).abs() / expected_bias.max(bias) <= 0.0001);
            })
        };
    }

    #[test]
    fn should_propagate_to_correct_value() -> Result<(), CompilationOrOpenCLError> {
        let state = setup_opencl(DeviceType::GPU).unwrap();

        let queue = state.queues.first().unwrap();
        let context = &state.context;

        let samples_amount = 4;
        let inputs_amount = 5;
        let outputs_amount = 5;

        let mut gpu_dense: Dense = Dense::new_raw(inputs_amount, outputs_amount);
        gpu_dense.init(&state)?;

        let mut rng = thread_rng();
        let input_samples: Vec<Vec<f32>> = (0..samples_amount)
            .into_iter()
            .map(|_| {
                (0..inputs_amount)
                    .into_iter()
                    .map(|_| rng.gen_range(-1231.0_f32..=15151.0_f32))
                    .collect()
            })
            .collect();

        let mut expected_outputs = vec![vec![0.0; outputs_amount]; samples_amount];
        input_samples.iter().enumerate().for_each(|(i, inputs)| {
            for (j, input_to_outputs) in gpu_dense.weights.iter().enumerate() {
                for (k, weight) in input_to_outputs.iter().enumerate() {
                    expected_outputs[i][k] += weight * inputs[j]; // + gpu_dense.biases[k];
                }
            }
            for (k, bias) in gpu_dense.biases.iter().enumerate() {
                expected_outputs[i][k] += bias;
            }
        });

        let mut input_samples_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            samples_amount * inputs_amount,
            ptr::null_mut(),
        )?;

        let input_samples_gpu_write_event = queue.enqueue_write_buffer(
            &mut input_samples_buffer,
            CL_BLOCKING,
            0,
            input_samples
                .iter()
                .map(|x| x.to_vec())
                .flatten()
                .collect::<Vec<f32>>()
                .as_slice(),
            &[],
        )?;

        input_samples_gpu_write_event.wait()?;

        let gpu_outputs_buffer = gpu_dense.propagate(&input_samples_buffer)?;

        let mut outputs_vec = vec![0.0; samples_amount * outputs_amount];
        let gpu_flattend_outputs = outputs_vec.as_mut_slice();

        let read_flattened_outputs_gpu = queue.enqueue_read_buffer(
            &gpu_outputs_buffer,
            CL_NON_BLOCKING,
            0,
            gpu_flattend_outputs,
            &[],
        )?;

        read_flattened_outputs_gpu.wait()?;

        let flattened_expected_outputs: Vec<f32> = expected_outputs
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect();

        // println!("CPU prediction: {:?}", flattened_expected_outputs);
        // println!("\nGPU prediction: {:?}", outputs_vec);

        {
            let a = &outputs_vec;
            let b = &flattened_expected_outputs;
            let max_dist = 0.01;
            assert_eq!(a.len(), b.len());

            a.iter().zip(b).for_each(|(x, y)| {
                // println!("x:{}\ny:{}", x, y);
                assert!((x - y).abs() / x.max(*y) <= max_dist);
            });
        };

        Ok(())
    }
}
