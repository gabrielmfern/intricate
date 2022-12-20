//! A module containing internal data types for Intricate

use opencl3::error_codes::ClError;
use savefile_derive::Savefile;

use intricate_macros::{EnumLayer, FromForAllUnnamedVariants};

use crate::{
    layers::{
        activations::{ReLU, Sigmoid, SoftMax, TanH},
        Dense, conv2d::Conv2D,
    },
    loss_functions::LossFunction,
    optimizers::Optimizer, utils::opencl::BufferConversionError,
};

#[derive(Debug)]
/// An error that happens when a program is not found.
///
/// It contains a tuple that has the Program's name that was not found.
pub struct ProgramNotFoundError(pub String);

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum that contains all the errors that can happen when trying to sync a buffer from a device
/// to the host.
pub enum SyncDataError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when the state was not setup or passed into the struct that is using it.
    NotInitialized,
    /// Happens when the field trying to be synced is not in the device.
    NotAllocatedInDevice {
        /// The name of the field trying to be synced.
        field_name: String,
    },
    /// Happens when something goes wrong with a buffer operation.
    BufferConversion(BufferConversionError),
    /// Happens when there is no command queue to be used.
    NoCommandQueue,
}

impl From<String> for ProgramNotFoundError {
    fn from(program: String) -> Self {
        ProgramNotFoundError(program)
    }
}

#[derive(Debug)]
/// An error that happens when a kernel is not found inside of a IntricateProgram.
///
/// It contains a tuple that has the Kernel's name that was not found.
pub struct KernelNotFoundError(pub String);

impl From<String> for KernelNotFoundError {
    fn from(kernel: String) -> Self {
        KernelNotFoundError(kernel)
    }
}

#[derive(Debug, Savefile, EnumLayer, FromForAllUnnamedVariants)]
/// All of the possible layers that a usual Sequential Model can have.
#[allow(missing_docs)]
pub enum ModelLayer<'a> {
    Dense(Dense<'a>),
    Conv2D(Conv2D<'a>),

    TanH(TanH<'a>),
    SoftMax(SoftMax<'a>),
    ReLU(ReLU<'a>),
    Sigmoid(Sigmoid<'a>),
}

#[derive(Debug)]
/// Some verbosity options to determine what should appear when training a Model or not.
pub struct TrainingVerbosity {
    /// Weather or not to show a message such as `epoch #5`
    pub(crate) show_current_epoch: bool,

    /// Weather or not to show a progress bar of an epoch with the current steps it has gon through
    /// and the missing steps as well as an elapsed time and the last step's loss
    pub(crate) show_epoch_progress: bool,

    /// Weather or not to show how much time was elapsed going through a whole epoch
    pub(crate) show_epoch_elapsed: bool,

    /// Weather or not the loss of the Model after a epoch should be printed
    pub(crate) print_loss: bool,

    /// Weather or not the loss of the Model after a epoch should be printed
    pub(crate) print_accuracy: bool,

    /// Weather or not to show a warning before stopping the training proccess due to a halting
    /// condition.
    pub(crate) halting_condition_warning: bool,
}

impl Default for TrainingVerbosity {
    fn default() -> Self {
        TrainingVerbosity {
            show_current_epoch: true,
            show_epoch_progress: true,
            show_epoch_elapsed: true,
            print_loss: true,
            print_accuracy: false,
            halting_condition_warning: false,
        }
    }
}

#[derive(Debug)]
/// The condition for the training of a Model to stop before the amount predetermined epochs is
/// reached.
pub enum HaltingCondition {
    /// Will stop the training process if a certain loss is reached or surpassed.
    ///
    /// To use this you need to set `compute_loss` to true or at least set `print_loss` to true in
    /// the `verbosity` field.
    MinLossReached(f32),

    /// Will stop the training process if a certain accuracy is reached or surpassed.
    ///
    /// To use this you need to set `compute_accuracy` to true.
    MinAccuracyReached(f32),
}

#[derive(Debug)]
/// A struct that defines the options for training a Model.
///
/// # Example
///
/// ```rust
/// use intricate::TrainingOptions; 
/// use intricate::loss_functions::MeanSquared;
/// use intricate::optimizers;
///
/// let mut mean_squared = MeanSquared::new();
/// let mut basic_optimizer = optimizers::Basic::new(0.01 /* learning rate */);
///
/// let my_options = TrainingOptions::new(
///     &mut mean_squared, // this can be any struct that implements the Loss Function trait
///     &mut basic_optimizer // this can also be any struct that implements the Optimizer trait
/// ).set_batch_size(32)
///  .set_epochs(10)
///  .should_compute_loss(true).unwrap();
/// ```
pub struct TrainingOptions<'a> {
    /// The loss function that will be used for calculating how **wrong** the Model
    /// was after some prediction over many samples.
    pub(crate) loss_fn: &'a mut dyn LossFunction<'a>,

    /// The size of the batch given at once to the Model for training.
    /// This is here because a Model will always run on mini batches, if you wish to do `Batch
    /// Gradient Descent` you will need to just set this to the amount of training samples you
    /// have and for `Stochastic Gradient Descent` you just need to set this to one.
    pub(crate) batch_size: usize,

    /// The optimizer that will both optimize parameters before calculating gradients as well as
    /// optimize gradients and compute update vectors that are going to be actually used when
    /// applying the gradients
    pub(crate) optimizer: &'a mut dyn Optimizer<'a>, // this is mut because we need to init the optimizer
                                              // before using it
                                             
    /// Some verbosity options to determine what should appear when training a Model or not.
    pub(crate) verbosity: TrainingVerbosity,

    /// The extra conditions for stopping the Model's training before the amount of predetermined
    /// epochs is reached.
    pub(crate) halting_condition: Option<HaltingCondition>,

    /// Weather or not at the end of each training step the Model should compute its own loss and
    /// store it to then return a Vec containing all of them.
    ///
    /// This will be necessarily true if verbosity's `print_loss` is set to **true**
    /// and will also be true if there is a `Halting Condition` for the loss.
    pub(crate) compute_loss: bool,

    /// Weather or not to keep track of the accuracies after each epoch of training of the Model.
    pub(crate) compute_accuracy: bool,

    /// The amount of epochs that the Model should train for.
    pub(crate) epochs: usize,
}

#[derive(Debug)]
/// The error that will be used for setting some specific parameters in the TrainingOptions.
pub struct InvalidTrainingOptionError<T> {
    /// The value trying to be set by a builder pattern method in the TrainingOptions struct
    pub value_trying_to_be_set: T,
    /// The name of the parameter trying to be set.
    pub parameter_name: &'static str,
    /// The actual error message specifying what is wrong with setting this value
    pub error_message: String,
}

impl<'a> TrainingOptions<'a> {
    /// Creates new Training Options with some default parameters and a specified loss_fn and
    /// optimize.
    pub fn new(
        loss_fn: &'a mut dyn LossFunction<'a>,
        optimizer: &'a mut dyn Optimizer<'a>,
    ) -> Self {
        TrainingOptions { 
            loss_fn,
            batch_size: 256, 
            optimizer,
            verbosity: TrainingVerbosity::default(), 
            halting_condition: None, 
            compute_loss: true,
            compute_accuracy: false, 
            epochs: 10 
        }
    }

    /// Sets a new batch size into self and returns the mutated Training Options
    pub fn set_batch_size(mut self, new_batch_size: usize) -> Self {
        self.batch_size = new_batch_size;

        self
    }

    /// Sets the halting condition into self and returns the mutated Self
    pub fn set_halting_condition(
        mut self, 
        halting_condition: HaltingCondition
    ) -> Result<Self, InvalidTrainingOptionError<HaltingCondition>> {
        match halting_condition {
            HaltingCondition::MinLossReached(_) => {
                if !self.compute_loss {
                    return Err(InvalidTrainingOptionError { 
                        value_trying_to_be_set: halting_condition, 
                        parameter_name: "halting_condition", 
                        error_message: format!("Unable to set the halting condition to MinLossReached since the loss is not set to be calculated!")
                    });
                }
            },
            HaltingCondition::MinAccuracyReached(_) => {
                if !self.compute_accuracy {
                    return Err(InvalidTrainingOptionError { 
                        value_trying_to_be_set: halting_condition, 
                        parameter_name: "halting_condition", 
                        error_message: format!("Unable to set the halting condition to MinAccuracyReached since the accuracy is not set to be calculated!")
                    });
                }
            }
        };

        self.halting_condition = Some(halting_condition);

        Ok(self)
    }

    /// Sets weather or not the loss should be computed after each training step for graphing,
    /// printing or other purposes. (the losess are returned by the `fit` method)
    pub fn should_compute_loss(
        mut self, 
        should: bool
    ) -> Result<Self, InvalidTrainingOptionError<bool>> {
        if self.verbosity.print_loss && !should {
            return Err(InvalidTrainingOptionError {
                value_trying_to_be_set: should,
                parameter_name: "compute_loss",
                error_message: format!("Could not set 'compute_loss' = false since there will be a need for the loss because of print_loss being enabled in the TrainingVerbosity!"),
            });
        }

        self.compute_loss = should;

        Ok(self)
    }

    /// Sets weather or not the accuracy should be computed after each training step for graphing,
    /// printing or other purposes. (the accuracies are returned by the `fit` method)
    pub fn should_compute_accuracy(
        mut self, 
        should: bool
    ) -> Result<Self, InvalidTrainingOptionError<bool>> {
        if self.verbosity.print_loss && !should {
            return Err(InvalidTrainingOptionError {
                value_trying_to_be_set: should,
                parameter_name: "compute_loss",
                error_message: format!("Could not set 'compute_loss' = false since there will be a need for the loss because of print_loss being enabled in the TrainingVerbosity!"),
            });
        }

        self.compute_accuracy = should;

        Ok(self)
    }

    /// Sets weather or not the loss should be printed after the epoch
    pub fn should_print_loss(mut self, should: bool) -> Result<Self, InvalidTrainingOptionError<bool>> {
        if should && !self.compute_loss {
            return Err(InvalidTrainingOptionError { 
                value_trying_to_be_set: should, 
                parameter_name: "print_loss", 
                error_message: format!("Cannot print the loss without having first calculated it! Please use should_compute_loss first with 'true' as parameter!")
            });
        }

        self.verbosity.print_loss = should;

        Ok(self)
    }

    /// Sets weather or not the accuracy should be printed after the epoch
    pub fn should_print_accuracy(mut self, should: bool) -> Result<Self, InvalidTrainingOptionError<bool>> {
        if should && !self.compute_loss {
            return Err(InvalidTrainingOptionError { 
                value_trying_to_be_set: should, 
                parameter_name: "print_accuracy ", 
                error_message: format!("Cannot print the accuracy without having first calculated it! Please use should_compute_accuracy first with 'true' as parameter!")
            });
        }

        self.verbosity.print_accuracy = should;

        Ok(self)
    }

    /// Sets weather or not the progress of the current epoch is shown using an `indicatif`
    /// progress bar.
    pub fn should_show_epoch_progress(mut self, should: bool) -> Self {
        self.verbosity.show_epoch_progress = should;

        self
    }

    /// Sets weather or not there should appear a message showing the current epoch before starting
    /// the training steps.
    pub fn should_show_current_epoch_message(mut self, should: bool) -> Self {
        self.verbosity.show_current_epoch = should;

        self
    }

    /// Sets weather or not there should a warning printed if the training process is forcefully
    /// stopped by a halting condition
    pub fn should_show_halting_condition_warning(mut self, should: bool) -> Result<Self, InvalidTrainingOptionError<bool>> {
        if should && self.halting_condition.is_none() {
            return Err(InvalidTrainingOptionError { 
                value_trying_to_be_set: should, 
                parameter_name: "halting_condition_warning", 
                error_message: format!("Cannot have a halting condition warning without a Halting Condition being defined!"),
            });
        }

        self.verbosity.halting_condition_warning = should;

        Ok(self)
    }

    /// Sets the amount of epochs the training process will go through
    pub fn set_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;

        self
    }
}

#[derive(Debug)]
/// Just a struct that contains that history of metrics during a `fit` method of a Model
pub struct TrainingResults {
    /// The history of the losses after each one of the training steps
    pub loss_per_training_steps: Vec<f32>,
    /// The history of the accuracies after each one of the training steps
    pub accuracy_per_training_steps: Vec<f32>,
}