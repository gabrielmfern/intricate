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

use crate::{
    layers::Layer,
    loss_functions::LossFunction,
    types::{CompilationOrOpenCLError, ModelLayer, ModelLossFunction, TrainingOptions},
};

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
///
/// # Example
///
/// ```rust
/// use intricate::{
///     types::ModelLayer,
///     layers::{
///         Dense,
///         activations::TanH,
///     },
///     Model,
/// };
///
/// let my_layers: Vec<ModelLayer> = vec![
///     Dense::new(768, 300), // make sure the outputs are the same as the inputs of the next
///                           // one or Intricate will panic when asserting these are of the
///                           // same shape
///     Dense::new(300, 100),
///     TanH::new(100), // Activations are layers by themselves, this makes all calculations
///                     // much simpler under the hood
/// ];
///
/// let my_model: Model = Model::new(my_layers);
/// ```
pub struct Model<'a> {
    pub layers: Vec<ModelLayer<'a>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Model<'a> {
    /// Creates a new Model from a Vec of layers with an empty OpenCLState.
    ///
    /// This does not initialize OpenCL in each of the layers, after calling this method, to do
    /// anything with the Model you **need** to call the `ìnit` method.
    pub fn new(layers: Vec<ModelLayer<'a>>) -> Model<'a> {
        Model {
            layers,
            opencl_state: None,
        }
    }

    /// Sends the trained parameters in each layer from the GPU to the CPU.
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong
    /// while reading the buffers into the CPU.
    pub fn sync_data_from_buffers_to_host(&mut self) -> Result<(), ClError> {
        for layer in self.layers.iter_mut() {
            layer.sync_data_from_buffers_to_host()?;
        }

        Ok(())
    }

    /// Initializes all of the layers inside of the Model and starts holding the reference to the
    /// OpenCL state passed in as parameter.
    ///
    /// # Errors
    ///
    /// Perhaps in one of the layers OpenCL compilation of a program fails, and this will yield a
    /// CompilationError (just a String with some stacktrace to the error).
    /// If the programs were compiled successfully don't put your guard down yet because OpenCL may
    /// yield some error if something it needs to do fails.
    pub fn init(
        &mut self,
        opencl_state: &'a OpenCLState,
    ) -> Result<(), CompilationOrOpenCLError> {
        for layer in self.layers.iter_mut() {
            layer.init(opencl_state)?;
        }

        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    /// Will fetch the outputs of the last layer in the Model.
    ///
    /// This is useful since prediction in the Model will just yield a Buffer (a memory allocation
    /// inside of the GPU) which can't really be manipulated by the CPU and is not the data itself.
    ///
    /// So if you want to do some manipulation with the data after predicting, just call this
    /// method.
    ///
    /// # Errors
    ///
    /// This function will yield an error ClError if something goes wrong while reading the data
    /// inside of the GPU.
    ///
    /// # Panics
    ///
    /// Will panic if the 'init' method was not called setting the **opencl_state**, if there
    /// is no layers in the model or if there is not outputs in the last layer.
    pub fn get_last_prediction(&self) -> Result<Vec<f32>, ClError> {
        // TODO: get rid of all these unwraps and make a customized enum for errors in this
        // function
        assert!(self.opencl_state.is_some());
        assert!(!self.opencl_state.unwrap().queues.is_empty());
        let state = self.opencl_state.unwrap();
        let queue = state.queues.first().unwrap();

        let buffer = self.layers.last().unwrap().get_last_outputs().unwrap();

        let size = buffer.size()? / mem::size_of::<cl_float>();
        let mut resulting_vec = vec![0.0; size];
        let resulting_slice = resulting_vec.as_mut_slice();

        queue
            .enqueue_read_buffer(buffer, CL_NON_BLOCKING, 0, resulting_slice, &[])?
            .wait()?;

        Ok(resulting_vec)
    }

    /// Plain old `predict` function, will receive the inputs for the model and will give out a
    /// OpenCL buffer associated with the outputs in the GPU.
    /// If you need to get the data from the buffer don't worry, just call the `get_last_prediction`
    /// method after predicting. (also the reference to the output may be annoying to work with)
    ///
    /// # Errors
    ///
    /// Will yield an error if something goes wrong with OpenCL, perhaps running some kernels, or
    /// with buffer allocation.
    ///
    /// # Panics
    ///
    /// Will panic if the `init` was not called on the Model, or if the model has no layers.
    pub fn predict(&mut self, input_samples: &Vec<Vec<f32>>) -> Result<&Buffer<cl_float>, ClError> {
        assert!(self.opencl_state.is_some());
        assert!(!self.opencl_state.unwrap().queues.is_empty());
        let state = self.opencl_state.unwrap();
        let queue = state.queues.first().unwrap();

        let samples_amount = input_samples.len();

        let mut first_input_samples_buffer = Buffer::<cl_float>::create(
            &state.context,
            CL_MEM_READ_WRITE,
            samples_amount * input_samples[0].len(),
            ptr::null_mut(),
        )?;

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

        let result = self.predict_with_moved_buffer(first_input_samples_buffer)?;

        Ok(result)
    }

    // Just used for calling predict without having trouble with references
    fn predict_with_moved_buffer(
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
        assert!(!self.opencl_state.unwrap().queues.is_empty());
        let state = self.opencl_state.unwrap();
        let queue = state.queues.first().unwrap();

        let samples_amount = training_input_samples.len();

        training_options.loss_algorithm.init(state)?;

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
            .wait()?;
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
                &training_options.learning_rate,
                &training_options.loss_algorithm,
                &training_options.should_print_information,
            )?;
        }

        Ok(loss)
    }

    pub fn back_propagate(
        &mut self,
        samples_amount: usize,
        training_input_samples: &Buffer<cl_float>,
        training_expected_output_samples: &Buffer<cl_float>,
        learning_rate: &f32,
        loss_function: &ModelLossFunction<'a>,
        verbose: &bool,
    ) -> Result<Option<f32>, ClError> {
        let start_instant = Instant::now();

        let training_actual_outputs = self.predict_with_buffer(training_input_samples)?;

        let outputs_amount =
            training_expected_output_samples.size()? / samples_amount / mem::size_of::<cl_float>();

        let mut lost_to_outputs_derivatives = loss_function
            .compute_loss_derivative_with_respect_to_output_samples(
                &training_actual_outputs,
                &training_expected_output_samples,
                samples_amount,
            )?;

        for (layer_index, layer) in self.layers.iter_mut().enumerate().rev() {
            if layer_index > 0 {
                // always Some
                lost_to_outputs_derivatives = layer
                    .back_propagate(true, &lost_to_outputs_derivatives, *learning_rate)?
                    .unwrap();
            } else {
                layer.back_propagate(
                    // always None
                    false,
                    &lost_to_outputs_derivatives,
                    *learning_rate,
                )?;
            }
        }

        let actual_sample_outputs = self.predict_with_buffer(training_input_samples)?;

        if *verbose {
            let new_loss = loss_function.compute_loss(
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