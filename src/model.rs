//! The module that implements a sequential Model, that contains some layers, and forward passes
//! some inputs over and over again from one layer to another.

use std::{fmt::Write, time::Instant};

use super::utils::OpenCLState;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use intricate_macros::FromForAllUnnamedVariants;
#[allow(unused_imports)]
use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::{cl_float, Device},
    error_codes::ClError,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};
use opencl3::{error_codes::cl_int, kernel::ExecuteKernel, memory::CL_MEM_READ_ONLY};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use savefile_derive::Savefile;
use std::mem;

use crate::{
    layers::{
        Gradient, Layer, LayerGradientApplicationError, LayerGradientComputationError,
        LayerInitializationError, LayerLossToInputDifferentiationError, LayerPropagationError,
        ParametersOptimizationError,
    },
    loss_functions::{
        LossComputationError, LossFunction, LossToModelOutputsDerivativesComputationError,
    },
    optimizers::Optimizer,
    types::{
        HaltingCondition, KernelNotFoundError, ModelLayer, ProgramNotFoundError, SyncDataError,
        TrainingOptions, TrainingResults,
    },
    utils::{
        opencl::{
            empty_buffer, ensure_program, BufferConversionError, BufferLike, BufferOperationError,
            EnsureKernelsAndProgramError,
        },
        BufferOperations,
    },
};

const MODEL_PROGRAM_SOURCE: &str = include_str!("kernels/model.cl");
const MODEL_PROGRAM_NAME: &str = "MODEL";
const COMPUTE_ACCURACIES_KERNEL_NAME: &str = "compute_accuracy_per_output";

pub(crate) fn compile_model(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let kernels = &[COMPUTE_ACCURACIES_KERNEL_NAME.to_string()];

    ensure_program(
        opencl_state,
        MODEL_PROGRAM_NAME.to_string(),
        MODEL_PROGRAM_SOURCE.to_string(),
        "".to_string(),
        kernels,
    )?;

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Savefile)]
/// An Intricate Model can be defined as just an ordering
/// of some layers with their inputs and outputs, the Model receives
/// the inputs for the first layer and results in the outputs of the last layer.
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
    /// Happens when the Model has no layers inside of it
    NoLayers,
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

    /// Happens when a required program was not found
    ProgramNotFound(ProgramNotFoundError),
    /// Happens when a required kernel was not found in a program
    KernelNotFound(KernelNotFoundError),

    /// Happens when the Halting Condition for the training process is the `MinLossReached` which
    /// requires that the loss is computed in the training process
    NoLossForHaltingCondition,
    /// Happens when the Halting Condition for the training process is the `MinAccuracyReached` which
    /// requires that the accuracy is computed in the training process
    NoAccuracyForHaltingCondition,

    /// Happens when something goes wrong in a predefined buffer operation
    BufferOperation(BufferOperationError),

    /// Happens if something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when converting a Vec into a buffer.
    Conversion(BufferConversionError),
    /// Happens when something goes wrong in the gradient computations of the Model.
    ModelGradientComputation(ModelGradientComputationError),
    /// Happens when something goes wrong in the gradient application of the Model.
    ModelGradientApplication(ModelGradientApplicationError),
    /// Happens when something goes wrong in the prediction of the Model.
    ModelPrediction(ModelPredictionError),

    /// Happens when something goes wrong when trying to optimize a Layer's parameters.
    ParameterOptimization(usize, ParametersOptimizationError),

    /// Happens when the Model has no layers inside of it
    NoLayers,
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

    /// Happens when the gradient computation of a layer goes wrong.
    ///
    /// This error also contains the index of the layer at which this error happenned.
    LayerGradientComputation(usize, LayerGradientComputationError),
    /// Happens when the differentiation of the inputs of a layer with respect to the loss goes wrong.
    ///
    /// This error also contains the index of the layer at which this error happenned.
    LayerLossToInputDifferentiation(usize, LayerLossToInputDifferentiationError),

    /// Happens when something goes wrong in the prediction of the Model.
    ModelPrediction(ModelPredictionError),

    /// Happens when the Model has no layers inside of it
    NoLayers,
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

    /// Happens when the Model has no layers inside of it
    NoLayers,
    /// Happens when there goes something wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when the propagation of a layer goes wrong.
    ModelPrediction(ModelPredictionError),
    /// Happens when the gradient application of a layer goes wrong.
    ///
    /// This error contains the index of the layer on which th error happenned and the actual
    /// error.
    LayerGradientApllication(usize, LayerGradientApplicationError),
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
    pub fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), LayerInitializationError> {
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
    /// Yields an error if:
    /// - The Model was not intialized;
    /// - THe Model has no layers;
    /// - The Model has not yet predicted;
    /// - Something goes wrong when reading the data from the outputs buffer.
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
    /// Yields an error if:
    /// - The Model was not initialized;
    /// - There is no command queue in the OpenCLState;
    /// - Something goes wrong in the Vec to Buffer conversion;
    /// - Something goes wrong when predicting with a moved buffer on the Model.
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
            .to_buffer(false, state)?;

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
    /// Yields an error if:
    /// - The Model was not initialized;
    /// - There is no OpenCL command queue in the OpenCLState;
    /// - THere is no layers in the Mode;
    /// - Something goes wrong in the Model's propagation.
    pub fn predict_with_buffer<'b>(
        &'b mut self,
        input_samples: &'b Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, ModelPredictionError> {
        // should this yield an error? since the layers already do yield an error in this case
        if self.opencl_state.is_none() {
            return Err(ModelPredictionError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.len() == 0 {
            return Err(ModelPredictionError::NoCommandQueue);
        }

        if self.layers.len() == 0 {
            return Err(ModelPredictionError::NoLayers);
        }

        let mut current_values: &Buffer<cl_float> = input_samples;

        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(current_values)?;
        }

        Ok(current_values)
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss, if it is computing the loss
    /// it will return the losses after every single **training step**.
    ///
    /// # Errors
    ///
    /// Yields an error if:
    /// - the Model is not initialized;
    /// - there is no command queue in the OpenCLState;
    /// - there are no layers in the Model;
    /// - something goes wrong in the initialization of the loss function;
    /// - something goes wrong in the initialization of the optimizer;
    /// - something goes wrong when trying to convert the training_inputs samples into a buffer;
    /// - something goes wrong when trying to convert the training_expected_output_samples into a
    /// buffer;
    /// - there are no calculated losses for a HaltingCondition of MinLoss;
    /// - there are no calculated accuracies for a HaltingCondition of MinAccuracy;
    /// - something goes wrong inside of parameter optimization;
    /// - something goes wrong in the gradients computation method;
    /// - something goes wrong in the gradients application;
    /// - something goes wrong in the prediction of the Model;
    /// - something goes wrong in the loss computation;
    /// - something goes wrong inside OpenCL.
    pub fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: &mut TrainingOptions<'a>,
    ) -> Result<TrainingResults, ModelFittingError> {
        if self.opencl_state.is_none() {
            return Err(ModelFittingError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelFittingError::NoCommandQueue);
        }

        assert_eq!(training_input_samples.len(), training_expected_output_samples.len());

        training_options.loss_fn.init(state)?;
        training_options.optimizer.init(state)?;

        let inputs_amount = self.layers[0].get_inputs_amount();
        let outputs_amount = self.layers.last().unwrap().get_outputs_amount();
        let samples_amount = training_input_samples.len();

        let input_samples_buffer = training_input_samples
            .par_iter()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<f32>>()
            .to_buffer(false, state)?;

        let expected_output_samples_buffer = training_expected_output_samples
            .par_iter()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<f32>>()
            .to_buffer(false, state)?;

        let steps_amount =
            calculate_training_steps_amount(samples_amount, training_options.batch_size);

        let mut losses: Vec<f32> = Vec::with_capacity(training_options.epochs * steps_amount);
        let mut accuracies: Vec<f32> = Vec::with_capacity(training_options.epochs * steps_amount);

        let per_step_inputs: Vec<Buffer<cl_float>> = separate_into_sub_buffer_batches(
            &input_samples_buffer,
            steps_amount,
            samples_amount,
            training_options.batch_size,
            inputs_amount,
        )?;

        let per_step_outputs: Vec<Buffer<cl_float>> = separate_into_sub_buffer_batches(
            &expected_output_samples_buffer,
            steps_amount,
            samples_amount,
            training_options.batch_size,
            outputs_amount,
        )?;

        let mut timestep: usize = 0;

        for epoch_index in 0..training_options.epochs {
            let start = Instant::now();

            let mut progress = None;
            if training_options.verbosity.show_current_epoch {
                println!("---------");
                println!("epoch #{}", epoch_index + 1);
            }

            if training_options.verbosity.show_epoch_progress
                && training_options.batch_size < samples_amount
            {
                let pbar = ProgressBar::new(
                    (samples_amount as f32 / training_options.batch_size as f32).ceil() as u64,
                );
                pbar.set_style(
                    ProgressStyle::with_template("[{bar:10}] [{per_second}/s] {pos}/{len} {elapsed}/{eta} {msg}")
                        .expect("unable to create epoch training steps progress bar")
                        .with_key("elapsed", |state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "{}", format!("{:.2}s", state.elapsed().as_secs_f32())).unwrap()
                        })
                        .with_key("per_second", |state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "{}", format!("{:.2}", state.per_sec())).unwrap()
                        })
                        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| {
                            write!(w, "{}", format!("{:.2}s", state.eta().as_secs_f32())).unwrap()
                        })
                        .progress_chars("=> "),
                );
                progress = Some(pbar);
            }

            let mut epoch_losses: Vec<f32> = Vec::with_capacity(steps_amount);
            let mut epoch_accuracies: Vec<f32> = Vec::with_capacity(steps_amount);

            for i_batch in 0..steps_amount {
                timestep += 1;

                let batch_inputs = &per_step_inputs[i_batch];
                let batch_outputs = &per_step_outputs[i_batch];

                let local_batch_size;
                if i_batch == steps_amount - 1 && samples_amount % training_options.batch_size != 0
                {
                    local_batch_size = samples_amount % training_options.batch_size;
                } else {
                    local_batch_size = training_options.batch_size;
                }

                let (optional_loss, optional_accuracy) = self.do_training_step(
                    batch_inputs,
                    batch_outputs,
                    local_batch_size,
                    timestep,
                    training_options,
                )?;

                if let Some(loss) = optional_loss {
                    losses.push(loss);
                    epoch_losses.push(loss);
                }

                if let Some(accuracy) = optional_accuracy {
                    accuracies.push(accuracy);
                    epoch_accuracies.push(accuracy);
                }

                if progress.is_some() {
                    let pbar = progress.as_ref().unwrap();
                    pbar.inc(1);
                    if training_options.verbosity.print_loss || training_options.compute_loss {
                        pbar.set_message(format!("(loss: {:.3})", losses.last().unwrap()));
                    }
                }
            }

            if progress.is_some() {
                progress.as_ref().unwrap().finish_and_clear();
            }

            let epoch_loss = epoch_losses.iter().sum::<f32>() / steps_amount as f32;
            let epoch_accuracy = epoch_accuracies.iter().sum::<f32>() / steps_amount as f32;

            if training_options.verbosity.print_loss {
                println!("got a loss of {:.3} after epoch", epoch_loss);
            }

            if training_options.verbosity.print_accuracy {
                println!(
                    "got a accuracy of {:.3} after epoch",
                    epoch_accuracy
                );
            }

            if training_options.verbosity.show_epoch_elapsed {
                println!("{:.3}s elapsed on epoch", start.elapsed().as_secs_f32());
            }

            if let Some(halting_condition) = &training_options.halting_condition {
                match halting_condition {
                    HaltingCondition::MinLossReached(min_loss) => {
                        if losses.is_empty() {
                            return Err(ModelFittingError::NoLossForHaltingCondition);
                        }

                        if min_loss >= &epoch_loss {
                            if training_options.verbosity.halting_condition_warning {
                                println!("stopping training process due to MinLossReached halting condition...");
                            }

                            break;
                        }
                    }
                    HaltingCondition::MinAccuracyReached(min_acc) => {
                        if accuracies.is_empty() {
                            return Err(ModelFittingError::NoAccuracyForHaltingCondition);
                        }

                        if min_acc <= &epoch_accuracy {
                            if training_options.verbosity.halting_condition_warning {
                                println!("stopping training process due to MinAccuracyReached halting condition...");
                            }

                            break;
                        }
                    }
                };
            }
        }

        Ok(TrainingResults {
            loss_per_training_steps: losses,
            accuracy_per_training_steps: accuracies,
        })
    }

    fn do_training_step(
        &mut self,
        input_samples: &Buffer<cl_float>,
        expected_output_samples: &Buffer<cl_float>,
        samples_amount: usize,
        timestep: usize,
        training_options: &mut TrainingOptions<'a>,
    ) -> Result<(Option<f32>, Option<f32>), ModelFittingError> {
        if self.opencl_state.is_none() {
            return Err(ModelFittingError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelFittingError::NoCommandQueue);
        }

        let queue = &state.queues[0];

        if self.layers.len() == 0 {
            return Err(ModelFittingError::NoLayers);
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Err(err) = layer.optimize_parameters(training_options.optimizer, i, timestep) {
                return Err(ModelFittingError::ParameterOptimization(i, err));
            }
        }

        let gradients = self.compute_gradients(
            &input_samples,
            &expected_output_samples,
            training_options.loss_fn,
        )?;

        self.apply_gradients(gradients.as_slice(), training_options.optimizer, timestep)?;

        let loss;
        let accuracy;

        if training_options.verbosity.print_loss
            || training_options.compute_loss
            || training_options.verbosity.print_accuracy
            || training_options.compute_accuracy
        {
            self.predict_with_buffer(input_samples)?;
        }

        if training_options.verbosity.print_loss || training_options.compute_loss {
            let actual_outputs = self.layers.last().unwrap().get_last_outputs().unwrap();

            loss = Some(training_options.loss_fn.compute_loss(
                actual_outputs,
                &expected_output_samples,
                samples_amount,
            )?);
        } else {
            loss = None;
        }

        if training_options.verbosity.print_accuracy || training_options.compute_accuracy {
            let actual_outputs = self.layers.last().unwrap().get_last_outputs().unwrap();

            let program = state.get_prgm(MODEL_PROGRAM_NAME)?;
            let accuracy_kernel = program.get_krnl(COMPUTE_ACCURACIES_KERNEL_NAME)?;

            let outputs_total_count = actual_outputs.size()? / mem::size_of::<cl_float>();

            let accuracies = empty_buffer(outputs_total_count, CL_MEM_READ_WRITE, state)?;

            ExecuteKernel::new(accuracy_kernel)
                .set_arg(actual_outputs)
                .set_arg(expected_output_samples)
                .set_arg(&accuracies)
                .set_arg(&(outputs_total_count as cl_int))
                .set_global_work_size(outputs_total_count)
                .enqueue_nd_range(queue)?;

            queue.finish()?;

            accuracy = Some(accuracies.sum(state)? / outputs_total_count as f32);
        } else {
            accuracy = None;
        }

        Ok((loss, accuracy))
    }

    /// Applies all the gradients calculated per layer calling each layer's respective
    /// **apply_gradients** function.
    ///
    /// # Errors
    ///
    /// Yields an error if:
    /// - the Model was not initialiazed;
    /// - there is no command queue inside the OpenCLState;
    /// - there are no layers inside the Model;
    /// - something goes wrong in the gradient application on a specific layer.
    pub fn apply_gradients(
        &mut self,
        gradients_per_layer: &[Vec<Gradient>],
        optimizer: &mut dyn Optimizer<'a>, //ModelOptimizer<'a>,
        timestep: usize,
    ) -> Result<(), ModelGradientApplicationError> {
        if self.opencl_state.is_none() {
            return Err(ModelGradientApplicationError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(ModelGradientApplicationError::NoCommandQueue);
        }

        if self.layers.len() == 0 {
            return Err(ModelGradientApplicationError::NoLayers);
        }

        for (layer_index, (layer, gradients)) in self
            .layers
            .iter_mut()
            .zip(gradients_per_layer.iter().rev())
            .enumerate()
        {
            let result = layer.apply_gradients(
                gradients.as_slice(), 
                optimizer, 
                layer_index,
                timestep,
            );

            if let Err(err) = result {
                return Err(ModelGradientApplicationError::LayerGradientApllication(
                    layer_index,
                    err,
                ));
            }
        }

        Ok(())
    }

    /// Computes the gradients for each one of the layers in the Model calling each layer's
    /// `compute_gradients` in conjuction with the `compute_loss_to_input_derivatives`.
    ///
    /// # Errors
    ///
    /// Yields an error if:
    /// - the model was not initialized;
    /// - there is no command queue in the OpenClState;
    /// - there are no layers in the Model;
    /// - something goes wrong trying to compute the size of the training_input_samples size;
    /// - something goes wrong in the predicting with a buffer;
    /// - something goes wrong in the initial loss fn gradients computation;
    /// - something goes wrong in the layer gradient computation;
    /// - something goes wrong when trying to pass on the derivatives between the layers.
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

        if self.layers.len() == 0 {
            return Err(ModelGradientComputationError::NoLayers);
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
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let gradients_result = layer.compute_gradients(&last_loss_to_outputs_derivatives);
            if let Ok(layer_gradients) = gradients_result {
                gradients.push(layer_gradients);
            } else if let Err(err) = gradients_result {
                return Err(ModelGradientComputationError::LayerGradientComputation(i, err));
            }

            let derivatives_result = layer.compute_loss_to_input_derivatives(&last_loss_to_outputs_derivatives); 
            if let Ok(derivatives) = derivatives_result {
                last_loss_to_outputs_derivatives = derivatives;
            } else if let Err(err) = derivatives_result {
                return Err(ModelGradientComputationError::LayerLossToInputDifferentiation(i, err));
            } 
        }

        Ok(gradients)
    }
}

fn calculate_training_steps_amount(samples_amount: usize, batch_size: usize) -> usize {
    (samples_amount as f32 / batch_size as f32).ceil() as usize
}

#[test]
fn should_calculate_training_steps_amount_correctly() {
    let samples_amount = 25;
    let batch_size = 4;

    let correct_training_steps = 7;
    let actual_training_steps = calculate_training_steps_amount(samples_amount, batch_size);

    assert_eq!(correct_training_steps, actual_training_steps);
}

fn calculate_batch_origin_and_count(
    steps_amount: usize,
    batch_size: usize,
    batch_index: usize,
    samples_amount: usize, //   origin, count
) -> (usize, usize) {
    let (origin, count);
    if batch_index == steps_amount - 1 && samples_amount % batch_size != 0 {
        count = samples_amount % batch_size;
        origin = samples_amount - count;
    } else {
        count = batch_size;
        origin = batch_index * count;
    }

    (origin, count)
}

#[test]
fn should_calculate_batch_origin_and_count_correctly_for_normal_batches() {
    let samples_amount = 6123;
    let batch_size = 25;

    let steps_amount = 245;
    let batch_index = 123;

    let expected_origin = batch_size * batch_index;
    let expected_count = batch_size;

    let (origin, count) =
        calculate_batch_origin_and_count(steps_amount, batch_size, batch_index, samples_amount);

    assert_eq!(origin, expected_origin);
    assert_eq!(count, expected_count);
}

#[test]
fn should_calculate_batch_origin_and_count_correctly_for_the_last_uneven_batch() {
    let samples_amount = 6123;
    let batch_size = 25;

    let steps_amount = 245;
    let batch_index = 244;

    let expected_origin = 6100;
    let expected_count = 23;

    let (origin, count) =
        calculate_batch_origin_and_count(steps_amount, batch_size, batch_index, samples_amount);

    assert_eq!(origin, expected_origin);
    assert_eq!(count, expected_count);
}

#[test]
fn should_calculate_batch_origin_and_count_correctly_for_the_last_uneven_batch_2() {
    let samples_amount = 60000;
    let batch_size = 64;

    let steps_amount = 938;
    let batch_index = 937;

    let expected_origin = 59968;
    let expected_count = 32;

    let (origin, count) =
        calculate_batch_origin_and_count(steps_amount, batch_size, batch_index, samples_amount);

    assert_eq!(origin, expected_origin);
    assert_eq!(count, expected_count);
}

fn separate_into_sub_buffer_batches(
    buffer: &Buffer<cl_float>,
    steps_amount: usize,

    samples_amount: usize,
    batch_size: usize,

    feature_amount: usize,
) -> Result<Vec<Buffer<cl_float>>, ClError> {
    let buff_size = buffer.size()?;
    assert_eq!(buff_size % samples_amount, 0);
    assert_eq!(buff_size % feature_amount, 0);

    let mut per_step_feature: Vec<Buffer<cl_float>> = Vec::with_capacity(steps_amount);

    for i_batch in 0..steps_amount {
        let (origin, count) =
            calculate_batch_origin_and_count(steps_amount, batch_size, i_batch, samples_amount);

        let batch = buffer.create_sub_buffer(
            CL_MEM_READ_ONLY,
            origin * feature_amount,
            count * feature_amount,
        )?;

        per_step_feature.push(batch);
    }

    Ok(per_step_feature)
}