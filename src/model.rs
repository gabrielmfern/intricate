//! The module that implements a sequential Model, that contains some layers, and forward passes
//! some inputs over and over again from one layer to another.

use std::time::Instant;

use super::utils::OpenCLState;
use intricate_macros::FromForAllUnnamedVariants;
use opencl3::memory::CL_MEM_READ_ONLY;
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

use crate::{
    layers::{
        Gradient, Layer, LayerGradientApplicationError, LayerGradientComputationError,
        LayerLossToInputDifferentiationError, LayerPropagationError, ParametersOptimizationError,
    },
    loss_functions::{LossFunction, LossComputationError, LossToModelOutputsDerivativesComputationError},
    optimizers::Optimizer,
    types::{
        ModelLayer, SyncDataError,
        TrainingOptions,
    },
    utils::opencl::{BufferConversionError, BufferLike},
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
///                           // one or Intricate will yield an error
///     Dense::new(300, 100),
///     TanH::new(100), // Activations are layers by themselves, this makes all calculations
///                     // much simpler under the hood
/// ];
///
/// let my_model: Model = Model::new(my_layers);
/// ```
pub struct Model<'a> {
    /// The list of layers that this Model consits of.
    pub layers: Vec<ModelLayer<'a>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// A optional reference to the current OpenCL state.
    pub opencl_state: Option<&'a OpenCLState>,
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen on a Vec Model prediction.
pub enum ModelPredictionError {
    /// Happens when the Model was not initialized before calling the method
    NotInitialized,
    /// Happens mostly if there is no devide in the current OpenCLState.
    NoCommandQueue,

    /// Happens if something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when converting a Vec into a buffer.
    Conversion(BufferConversionError),
    /// Happens when something goes wrong inside of the propagation of a Layer.
    LayerPropagation(LayerPropagationError),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen when fitting a Model.
pub enum ModelFittingError {
    /// Happens when the Model was not initialized before calling the method.
    NotInitialized,
    /// Happens mostly if there is no device in the current OpenCLState.
    NoCommandQueue,
    /// Happens if there is no device found by OpenCL
    NoDevice,

    /// Happens if something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when converting a Vec into a buffer.
    Conversion(BufferConversionError),
    /// Happens when something goes wrong in the gradient computations of the Model.
    ModelGradientComputation(ModelGradientComputationError),
    /// Happens when something goes wrong in the gradient application of the Model.
    ModelGradientApplication(ModelGradientApplicationError),
    /// Happens when something goes wrong when trying to optimize a Layer's parameters.
    ParameterOptimization(ParametersOptimizationError),
    /// Happens when something goes wrong in the propagation of the Model.
    LayerPropagation(LayerPropagationError),

    /// Happens when something goes wrong while computing the overall loss of the Model
    LossComputation(LossComputationError),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the possible errors that can happen while computing the Model's
/// gradients.
pub enum ModelGradientComputationError {
    /// Happens when the Model was not initialized.
    NotInitialized,
    /// Happens when there is no command queue in the current opencl state.
    NoCommandQueue,
    /// Happens when there is no device in the current opencl state.
    NoDevice,

    /// Happens when there goes something wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when the propagation of a layer goes wrong.
    LayerPropagation(LayerPropagationError),
    /// Happens when the gradient computation of a layer goes wrong.
    LayerGradientComputation(LayerGradientComputationError),
    /// Happens when the differentiation of the inputs of a layer with respect to the loss goes wrong.
    LayerLossToInputDifferentiation(LayerLossToInputDifferentiationError),

    /// Happens when something goes wrong
    LossDerivativesComputation(LossToModelOutputsDerivativesComputationError),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen while applying particular gradients to a
/// Model.
pub enum ModelGradientApplicationError {
    /// Happens when the Model was not initialized.
    NotInitialized,
    /// Happens when there is no command queue in the current opencl state.
    NoCommandQueue,
    /// Happens when there is no device in the current opencl state.
    NoDevice,

    /// Happens when there goes something wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when the propagation of a layer goes wrong.
    LayerPropagation(LayerPropagationError),
    /// Happens when the gradient application of a layer goes wrong.
    LayerGradientApllication(LayerGradientApplicationError),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum contaning all of the possible errors that can happen when trying to get the last
/// prediction of a Model as a Vec.
pub enum ModelGetLastPredictionError {
    /// Happens when the Model was not initialized
    NotInitialized,
    /// Happens when something goes wrong while trying to convert from a buffer to a Vec
    BufferConversion(BufferConversionError),
    /// Happens when the Model has no layers inside of it
    NoLayers,
    /// Happens when the method was called before predicting with the Model
    HasNotPredicted,
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
    pub fn sync_data_from_buffers_to_host(&mut self) -> Result<(), SyncDataError> {
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
    pub fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError> {
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
    pub fn get_last_prediction(&self) -> Result<Vec<f32>, ModelGetLastPredictionError> {
        if self.opencl_state.is_none() {
            return Err(ModelGetLastPredictionError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if self.layers.len() == 0 {
            return Err(ModelGetLastPredictionError::NoLayers);
        }

        let last_layer = self.layers.last().unwrap();

        if last_layer.get_last_outputs().is_none() {
            return Err(ModelGetLastPredictionError::HasNotPredicted);
        }

        let buffer = last_layer.get_last_outputs().unwrap();

        Ok(Vec::<f32>::from_buffer(&buffer, false, state)?)
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
    pub fn predict(
        &mut self,
        input_samples: &Vec<Vec<f32>>,
    ) -> Result<&Buffer<cl_float>, ModelPredictionError> {
        if self.opencl_state.is_none() {
            return Err(ModelPredictionError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelPredictionError::NoCommandQueue);
        }

        let samples_amount = input_samples.len();

        assert!(samples_amount > 0);

        let first_input_samples_buffer = input_samples
            .par_iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect::<Vec<f32>>()
            .to_buffer(CL_MEM_READ_ONLY, false, state)?;

        let result = self.predict_with_moved_buffer(first_input_samples_buffer)?;

        Ok(result)
    }

    // Just used for calling predict without having trouble with references
    fn predict_with_moved_buffer(
        &mut self,
        input_samples: Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
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

    /// This is the same as normal predict but it is made to run with a buffer instead of with a
    /// Vec, this is used on the backprop function, but is pub for being used if needed.
    ///
    /// # Errors
    ///
    /// Will yield an error if something goes wrong in executing kernels inside of any of the
    /// layers inside of the Model.
    pub fn predict_with_buffer<'b>(
        &'b mut self,
        input_samples: &'b Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
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
    ) -> Result<Option<f32>, ModelFittingError> {
        if self.opencl_state.is_none() {
            return Err(ModelFittingError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelFittingError::NoCommandQueue);
        }

        training_options.loss_fn.init(state)?;
        training_options.optimizer.init(state)?;

        let input_samples_buffer = training_input_samples
            .par_iter()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<f32>>()
            .to_buffer(CL_MEM_READ_ONLY, false, state)?;

        let expected_output_samples_buffer = training_expected_output_samples
            .par_iter()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<f32>>()
            .to_buffer(CL_MEM_READ_WRITE, false, state)?;

        let mut last_loss = None;

        let inputs_amount = self.layers[0].get_inputs_amount();
        let samples_amount =
            input_samples_buffer.size()? / mem::size_of::<cl_float>() / inputs_amount;

        for epoch_index in 0..training_options.epochs {
            if training_options.verbose {
                println!("epoch #{}", epoch_index + 1);
            }

            let start = Instant::now();

            for layer in self.layers.iter_mut() {
                layer.optimize_parameters(training_options.optimizer)?;
            }

            let gradients = self.compute_gradients(
                &input_samples_buffer,
                &expected_output_samples_buffer,
                training_options.loss_fn,
            )?;

            self.apply_gradients(gradients.as_slice(), training_options.optimizer)?;

            let actual_outputs = self.layers.last().unwrap().get_last_outputs().unwrap();

            if training_options.verbose || training_options.compute_loss {
                last_loss = Some(training_options.loss_fn.compute_loss(
                    actual_outputs,
                    &expected_output_samples_buffer,
                    samples_amount,
                )?);

                if training_options.verbose {
                    println!(
                        "epoch finished in {:?},\n after updating parameters loss found was {}",
                        start.elapsed(),
                        last_loss.unwrap()
                    );
                }
            }
        }

        Ok(last_loss)
    }

    /// Applies all the gradients calculated per layer calling each layer's respective
    /// **apply_gradients** function.
    ///
    /// # Errors
    ///
    /// This function will return an error if the Model was not initialized, if there is no command
    /// queue in the current `OpenCLState` and if the apply_gradients in any of the layers fails as
    /// well.
    pub fn apply_gradients(
        &mut self,
        gradients_per_layer: &[Vec<Gradient>],
        optimizer: &dyn Optimizer<'a>//ModelOptimizer<'a>,
    ) -> Result<(), ModelGradientApplicationError> {
        if self.opencl_state.is_none() {
            return Err(ModelGradientApplicationError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelGradientApplicationError::NoCommandQueue);
        }

        for (layer, gradients) in self.layers.iter_mut().zip(gradients_per_layer.iter().rev()) {
            layer.apply_gradients(gradients.as_slice(), optimizer)?;
        }

        Ok(())
    }

    /// Computes the gradients for each one of the layers in the Model calling each layer's
    /// `compute_gradients` in conjuction with the `compute_loss_to_input_derivatives`.
    ///
    /// # Errors
    ///
    /// This function will return an error if the Model was not initialized, if there is no command
    /// queue, if the prediction of the Model fails, if the computation of derivatives of inputs
    /// with respect to the loss fail or if the computation of a Layer's gradients fails..
    pub fn compute_gradients(
        &mut self,
        training_input_samples: &Buffer<cl_float>,
        // training_actual_outputs: &Buffer<cl_float>,
        training_expected_output_samples: &Buffer<cl_float>,
        loss_function: &dyn LossFunction, //ModelLossFunction<'a>,
    ) -> Result<Vec<Vec<Gradient>>, ModelGradientComputationError> {
        if self.opencl_state.is_none() {
            return Err(ModelGradientComputationError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelGradientComputationError::NoCommandQueue);
        }

        let first_layer = self.layers.first().unwrap();

        let inputs_amount = first_layer.get_inputs_amount();
        let samples_amount =
            training_input_samples.size()? / mem::size_of::<cl_float>() / inputs_amount;

        let layers_amount = self.layers.len();

        let training_actual_outputs = self.predict_with_buffer(training_input_samples)?;

        let mut gradients: Vec<Vec<Gradient>> = Vec::with_capacity(layers_amount);

        let mut last_loss_to_outputs_derivatives = loss_function
            .compute_loss_derivative_with_respect_to_output_samples(
                &training_actual_outputs,
                &training_expected_output_samples,
                samples_amount,
            )?;
        for layer in self.layers.iter().rev() {
            gradients.push(layer.compute_gradients(&last_loss_to_outputs_derivatives)?);
            last_loss_to_outputs_derivatives =
                layer.compute_loss_to_input_derivatives(&last_loss_to_outputs_derivatives)?;
        }

        Ok(gradients)
    }
}