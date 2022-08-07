use std::time::Instant;

use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::{cl_float, get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    error_codes::ClError,
    event::wait_for_events,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use savefile_derive::Savefile;
use std::mem;
use std::ptr;

use crate::{
    layers::{activations::tanh_gpu::TanHGPU, dense_gpu::DenseGPU, OpenCLLayer},
    loss_functions::{mean_squared::OpenCLMeanSquared, OpenCLLossFunction},
};

#[derive(Debug, Savefile)]
pub enum GPUModelLayer<'a> {
    Dense(DenseGPU<'a>),
    TanH(TanHGPU<'a>),
}

#[derive(Debug)]
pub enum GPUModelLossFunction<'a> {
    MeanSquared(OpenCLMeanSquared<'a>),
}

impl<'a> OpenCLLossFunction<'a> for GPUModelLossFunction<'a> {
    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<f32, ClError> {
        match self {
            GPUModelLossFunction::MeanSquared(lossfn) => {
                lossfn.compute_loss(output_samples, expected_outputs, samples_amount)
            }
        }
    }

    fn init(&mut self, context: &'a Context, queue: &'a CommandQueue) -> Result<(), ClError> {
        match self {
            GPUModelLossFunction::MeanSquared(lossfn) => {
                lossfn.init(context, queue)
            }
        }
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, ClError> {
        match self {
            GPUModelLossFunction::MeanSquared(lossfn) => lossfn
                .compute_loss_derivative_with_respect_to_output_samples(
                    output_samples,
                    expected_outputs,
                    samples_amount,
                ),
        }
    }
}

impl<'a> OpenCLLayer<'a> for GPUModelLayer<'a> {
    fn get_last_inputs(&self) -> Option<&Buffer<cl_float>> {
        match self {
            GPUModelLayer::Dense(layer) => layer.get_last_inputs(),
            GPUModelLayer::TanH(layer) => layer.get_last_inputs(),
        }
    }

    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>> {
        match self {
            GPUModelLayer::Dense(layer) => layer.get_last_outputs(),
            GPUModelLayer::TanH(layer) => layer.get_last_outputs(),
        }
    }

    fn get_inputs_amount(&self) -> usize {
        match self {
            GPUModelLayer::Dense(layer) => layer.get_inputs_amount(),
            GPUModelLayer::TanH(layer) => layer.get_inputs_amount(),
        }
    }

    fn get_outputs_amount(&self) -> usize {
        match self {
            GPUModelLayer::Dense(layer) => layer.get_outputs_amount(),
            GPUModelLayer::TanH(layer) => layer.get_outputs_amount(),
        }
    }

    fn init(&mut self, queue: &'a CommandQueue, context: &'a Context) -> Result<(), ClError> {
        match self {
            GPUModelLayer::Dense(layer) => layer.init(queue, context),
            GPUModelLayer::TanH(layer) => layer.init(queue, context),
        }
    }

    fn clean_up_gpu_state(&mut self) -> () {
        match self {
            GPUModelLayer::Dense(layer) => layer.clean_up_gpu_state(),
            GPUModelLayer::TanH(layer) => layer.clean_up_gpu_state(),
        }
    }

    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError> {
        match self {
            GPUModelLayer::Dense(layer) => layer.sync_data_from_gpu_with_cpu(),
            GPUModelLayer::TanH(layer) => layer.sync_data_from_gpu_with_cpu(),
        }
    }

    fn propagate(&mut self, inputs: &Buffer<cl_float>) -> Result<&Buffer<cl_float>, ClError> {
        match self {
            GPUModelLayer::Dense(layer) => layer.propagate(inputs),
            GPUModelLayer::TanH(layer) => layer.propagate(inputs),
        }
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Buffer<cl_float>,
        learning_rate: cl_float,
    ) -> Result<Option<Buffer<cl_float>>, ClError> {
        match self {
            GPUModelLayer::Dense(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            GPUModelLayer::TanH(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
        }
    }
}

pub struct GPUTrainingOptions<'a> {
    pub loss_algorithm: GPUModelLossFunction<'a>,
    // TODO: implement optimizers
    pub learning_rate: f32,
    pub should_print_information: bool,
    pub epochs: usize,
}

#[allow(dead_code)]
#[derive(Debug, Savefile)]
/// An Intricate GPUModel can be defined as just an ordering
/// of some layers with their inputs and outputs, the GPUModel receives
/// the inputs for the first layer and results in the outputs of the last layer,
///
/// the only difference from an ordinary Model is that thourgh its propagation and
/// backprop process it just moves around GPU buffers instead of Vec's
///
/// it also back_propagates returning the new loss for the Model based on the
/// defined Loss Function and calls the back_propagate method on each layer
/// going from the last to the first layer
///
/// once it is instantiated using the `new` method, it will get the first GPU device
/// it can find and use it for all the computations, in the future Intricate will
/// support multiple GPU's
pub struct GPUModel<'a> {
    pub layers: Vec<GPUModelLayer<'a>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_context: Option<Context>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_queue: Option<CommandQueue>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_device: Option<Device>,
}

impl<'a> GPUModel<'a> {
    pub fn new(layers: Vec<GPUModelLayer<'a>>) -> GPUModel<'a> {
        GPUModel {
            layers,
            opencl_device: None,
            opencl_queue: None,
            opencl_context: None,
        }
    }

    pub fn init(&'a mut self) -> Result<(), ClError> {
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
        let first_gpu = Device::new(device_ids[0]);
        let context = Context::from_device(&first_gpu)?;
        // here it can be activated to make profiling on kernels
        let queue = CommandQueue::create_with_properties(&context, first_gpu.id(), 0, 0)?;

        self.opencl_device = Some(first_gpu);
        self.opencl_queue = Some(queue);
        self.opencl_context = Some(context);

        for layer in self.layers.iter_mut() {
            layer.init(self.opencl_queue.as_ref().unwrap(), self.opencl_context.as_ref().unwrap())?;
        }

        Ok(())
    }

    pub fn predict(
        &mut self,
        input_samples: &Vec<Vec<f32>>,
    ) -> Result<Buffer<cl_float>, ClError> {
        let samples_amount = input_samples.len();
        let mut first_input_samples_buffer = Buffer::<cl_float>::create(
            self.opencl_context.as_ref().unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount * input_samples[0].len(),
            ptr::null_mut(),
        )?;
        let queue = self.opencl_queue.as_ref().unwrap();

        queue
            .enqueue_write_buffer(
                &mut first_input_samples_buffer,
                CL_NON_BLOCKING,
                0,
                input_samples
                    .par_iter()
                    .map(|x| x.to_vec())
                    .flatten()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &[],
            )?
            .wait()?;

        self.predict_with_buffer(first_input_samples_buffer)
    }

    pub fn predict_with_buffer(
        &mut self,
        input_samples: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, ClError> {
        let mut current_values: &Buffer<cl_float>;

        current_values = input_samples;

        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(current_values)?;
        }

        Ok(current_values)
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss
    pub fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: GPUTrainingOptions,
    ) -> Result<(), ClError> {
        let samples_amount = training_input_samples.len();
        let mut input_samples_buffer = Buffer::<cl_float>::create(
            self.opencl_context.as_ref().unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount * training_input_samples[0].len(),
            ptr::null_mut(),
        )?;

        let mut expected_output_samples_buffer = Buffer::<cl_float>::create(
            self.opencl_context.as_ref().unwrap(),
            CL_MEM_READ_WRITE,
            samples_amount * training_expected_output_samples[0].len(),
            ptr::null_mut(),
        )?;
        let queue = self.opencl_queue.as_ref().unwrap();

        let mut events = Vec::default();
        events.push(
            queue
                .enqueue_write_buffer(
                    &mut input_samples_buffer,
                    CL_NON_BLOCKING,
                    0,
                    training_input_samples
                        .par_iter()
                        .map(|x| x.to_vec())
                        .flatten()
                        .collect::<Vec<f32>>()
                        .as_slice(),
                    &[],
                )?
                .get(),
        );
        events.push(
            queue
                .enqueue_write_buffer(
                    &mut expected_output_samples_buffer,
                    CL_NON_BLOCKING,
                    0,
                    training_expected_output_samples
                        .par_iter()
                        .map(|x| x.to_vec())
                        .flatten()
                        .collect::<Vec<f32>>()
                        .as_slice(),
                    &[],
                )?
                .get(),
        );
        wait_for_events(events.as_slice())?;

        for epoch_index in 0..training_options.epochs {
            if training_options.should_print_information {
                println!("epoch #{}", epoch_index + 1);
            }

            self.back_propagate(
                samples_amount,
                &input_samples_buffer,
                &expected_output_samples_buffer,
                &training_options,
            )?;
        }

        Ok(())
    }

    pub fn back_propagate(
        &mut self,
        samples_amount: usize,
        training_input_samples: &Buffer<cl_float>,
        training_expected_output_samples: &Buffer<cl_float>,
        training_options: &GPUTrainingOptions,
    ) -> Result<Option<f32>, ClError> {
        assert_eq!(
            training_input_samples.size()?,
            training_expected_output_samples.size()?
        );

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict_with_buffer(training_input_samples)?;

        let outputs_amount = training_expected_output_samples.size()? / mem::size_of::<cl_float>();

        // Not sure if this can be implemented on the GPU because of the
        // computation of the loss bellow being done on dyn LossFunction
        let mut lost_to_outputs_derivatives = training_options
            .loss_algorithm
            .compute_loss_derivative_with_respect_to_output_samples(
                &training_actual_outputs,
                &training_expected_output_samples,
                samples_amount,
            )?;

        for (layer_index, layer) in self.layers.iter_mut().enumerate().rev() {
            if layer_index > 0 {
                // always Some
                lost_to_outputs_derivatives = layer
                    .back_propagate(
                        true,
                        &lost_to_outputs_derivatives,
                        training_options.learning_rate,
                    )?
                    .unwrap();
            } else {
                layer.back_propagate(
                    // always None
                    false,
                    &lost_to_outputs_derivatives,
                    training_options.learning_rate,
                )?;
            }
        }

        let actual_sample_outputs = self.predict_with_buffer(training_input_samples)?;

        if training_options.should_print_information {
            let new_loss = training_options
                .loss_algorithm
                .compute_loss(&actual_sample_outputs, &training_expected_output_samples, outputs_amount)?;
            println!(
                "{}s elapsed, now has loss of {}",
                start_instant.elapsed().as_secs_f32(),
                new_loss
            );
            Ok(Some(new_loss))
        } else {
            Ok(None)
        }
    }
}