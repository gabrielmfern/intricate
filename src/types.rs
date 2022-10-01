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
    pub show_current_epoch: bool,

    /// Weather or not to show a progress bar of an epoch with the current steps it has gon through
    /// and the missing steps as well as an elapsed time and the last step's loss
    pub show_epoch_progress: bool,

    /// Weather or not to show how much time was elapsed going through a whole epoch
    pub show_epoch_elapsed: bool,

    /// Weather or not the loss of the Model after a epoch should be printed
    pub print_loss: bool,

    /// Weather or not the loss of the Model after a epoch should be printed
    pub print_accuracy: bool,

    /// Weather or not to show a warning before stopping the training proccess due to a halting
    /// condition.
    pub halting_condition_warning: bool,
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

/// A struct that defines the options for training a Model.
pub struct TrainingOptions<'a> {
    /// The loss function that will be used for calculating how **wrong** the Model
    /// was after some prediction over many samples.
    pub loss_fn: &'a mut dyn LossFunction<'a>,

    /// The size of the batch given at once to the Model for training.
    /// This is here because a Model will always run on mini batches, if you wish to do `Batch
    /// Gradient Descent` you will need to just set this to the amount of training samples you
    /// have and for `Stochastic Gradient Descent` you just need to set this to one.
    pub batch_size: usize,

    /// The optimizer that will both optimize parameters before calculating gradients as well as
    /// optimize gradients and compute update vectors that are going to be actually used when
    /// applying the gradients
    pub optimizer: &'a mut dyn Optimizer<'a>, // this is mut because we need to init the optimizer
                                              // before using it
                                             
    /// Some verbosity options to determine what should appear when training a Model or not.
    pub verbosity: TrainingVerbosity,

    /// The extra conditions for stopping the Model's training before the amount of predetermined
    /// epochs is reached.
    pub halting_condition: Option<HaltingCondition>,

    /// Weather or not at the end of each training step the Model should compute its own loss and
    /// store it to then return a Vec containing all of them.
    ///
    /// This will be necessarily true if verbosity's `print_loss` is set to **true**
    /// and will also be true if there is a `Halting Condition` for the loss.
    pub compute_loss: bool,

    /// Weather or not to keep track of the accuracies after each epoch of training of the Model.
    pub compute_accuracy: bool,

    /// The amount of epochs that the Model should train for.
    pub epochs: usize,
}

#[derive(Debug)]
/// Just a struct that contains that history of metrics during a `fit` method of a Model
pub struct TrainingResults {
    /// The history of the losses after each one of the training steps
    pub loss_per_training_steps: Vec<f32>,
    /// The history of the accuracies after each one of the training steps
    pub accuracy_per_training_steps: Vec<f32>,
}