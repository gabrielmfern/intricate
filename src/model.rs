use std::time::Instant;

use super::utils::OpenCLState;
#[allow(unused_imports)]
use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::{cl_float, Device},
    error_codes::ClError,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use savefile_derive::Savefile;
use std::mem;
use std::ptr;

use crate::{types::{CompilationOrOpenCLError, ModelLayer, TrainingOptions}, layers::Layer, loss_functions::LossFunction};

#[allow(dead_code)]
#[derive(Debug, Savefile)]
/// An Intricate Model can be defined as just an ordering
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
/// support multiple GPU's here as well.
pub struct Model<'a> {
    pub layers: Vec<ModelLayer<'a>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Model<'a> {
    pub fn new(layers: Vec<ModelLayer<'a>>) -> Model<'a> {
        Model {
            layers,
            opencl_state: None,
        }
    }

    pub fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), CompilationOrOpenCLError> {
        for layer in self.layers.iter_mut() {
            layer.init(&opencl_state.queue, &opencl_state.context)?;
        }

        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    pub fn get_last_prediction(&self) -> Result<Vec<f32>, ClError> {
        assert!(self.opencl_state.is_some());
        let state = self.opencl_state.unwrap();

        let buffer = self.layers.last().unwrap().get_last_outputs().unwrap();

        let size = buffer.size()? / mem::size_of::<cl_float>();
        let mut resulting_vec = vec![0.0; size];
        let resulting_slice = resulting_vec.as_mut_slice();

        state
            .queue
            .enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, resulting_slice, &[])?
            .wait()?;

        Ok(resulting_vec)
    }

    pub fn predict(&mut self, input_samples: &Vec<Vec<f32>>) -> Result<&Buffer<cl_float>, ClError> {
        assert!(self.opencl_state.is_some());

        let state = self.opencl_state.unwrap();

        let samples_amount = input_samples.len();

        let mut first_input_samples_buffer = Buffer::<cl_float>::create(
            &state.context,
            CL_MEM_READ_WRITE,
            samples_amount * input_samples[0].len(),
            ptr::null_mut(),
        )?;

        state
            .queue
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

        let result = self.predict_with_moved_buffer(first_input_samples_buffer)?;

        Ok(result)
    }

    pub fn predict_with_moved_buffer(
        &mut self,
        input_samples: Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, ClError> {
        assert!(!self.layers.is_empty());

        let mut current_value: Option<&Buffer<cl_float>> = None;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == 0 {
                current_value = Some(layer.propagate(&input_samples)?);
            } else if current_value.is_some() {
                current_value = Some(layer.propagate(current_value.unwrap())?);
            }
        }

        Ok(current_value.unwrap())
    }

    pub fn predict_with_buffer<'b>(
        &'b mut self,
        input_samples: &'b Buffer<cl_float>,
    ) -> Result<&'b Buffer<cl_float>, ClError> {
        assert!(!self.layers.is_empty());

        let mut current_values: &Buffer<cl_float> = input_samples;

        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(current_values)?;
        }

        Ok(current_values)
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss, if it is computing the loss
    /// it will return the loss in the last epoch.
    ///
    /// # Errors
    ///
    /// This function will return an error if some compilation error
    /// happens while compiling the OpenCL programs for the Loss Function
    /// defined in the training options, or some error happens running the kernels
    /// at some point in the method calls.
    pub fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: &mut TrainingOptions<'a>,
    ) -> Result<Option<f32>, CompilationOrOpenCLError> {
        assert!(self.opencl_state.is_some());
        let state = self.opencl_state.unwrap();

        let samples_amount = training_input_samples.len();

        training_options
            .loss_algorithm
            .init(&state.context, &state.queue)?;

        let mut input_samples_buffer = Buffer::<cl_float>::create(
            &state.context,
            CL_MEM_READ_WRITE,
            samples_amount * training_input_samples[0].len(),
            ptr::null_mut(),
        )?;

        let mut expected_output_samples_buffer = Buffer::<cl_float>::create(
            &state.context,
            CL_MEM_READ_WRITE,
            samples_amount * training_expected_output_samples[0].len(),
            ptr::null_mut(),
        )?;

        state
            .queue
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
            .wait()?;
        state
            .queue
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
            .wait()?;

        let mut loss = None;

        for epoch_index in 0..training_options.epochs {
            if training_options.should_print_information {
                println!("epoch #{}", epoch_index + 1);
            }

            loss = self.back_propagate(
                samples_amount,
                &input_samples_buffer,
                &expected_output_samples_buffer,
                &training_options,
            )?;
        }

        Ok(loss)
    }

    pub fn back_propagate(
        &mut self,
        samples_amount: usize,
        training_input_samples: &Buffer<cl_float>,
        training_expected_output_samples: &Buffer<cl_float>,
        training_options: &TrainingOptions,
    ) -> Result<Option<f32>, ClError> {
        // dumb
        // assert_eq!(
        //     training_input_samples.size()?,
        //     training_expected_output_samples.size()?
        // );

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict_with_buffer(training_input_samples)?;

        let outputs_amount =
            training_expected_output_samples.size()? / samples_amount / mem::size_of::<cl_float>();

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
            let new_loss = training_options.loss_algorithm.compute_loss(
                &actual_sample_outputs,
                &training_expected_output_samples,
                outputs_amount,
            )?;
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
