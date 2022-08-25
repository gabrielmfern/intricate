//! A module containing internal data types for Intricate

use opencl3::error_codes::ClError;
use savefile_derive::Savefile;

use intricate_macros::{EnumLayer, FromForAllUnnamedVariants};

use crate::{
    layers::{
        activations::{ReLU, Sigmoid, SoftMax, TanH},
        Dense,
    },
    loss_functions::LossFunction,
    optimizers::Optimizer,
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
    TanH(TanH<'a>),
    SoftMax(SoftMax<'a>),
    ReLU(ReLU<'a>),
    Sigmoid(Sigmoid<'a>),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum that contains all of the possible Gradient Descent algorithms.
pub enum GradientDescent {}

/// A struct that defines the options for training a Model.
pub struct TrainingOptions<'a> {
    /// The loss function that will be used for calculating how **wrong** the Model
    /// was after some prediction over many samples.
    pub loss_algorithm: &'a mut dyn LossFunction<'a>,
    /// The graadient descent implementation that should be used for doing gradient descent
    /// during fitting
    // pub gradient_descent_method: GradientDescent,
    /// The optimizer that will both optimize parameters before calculating gradients as well as
    /// optimize gradients and compute update vectors that are going to be actually used when
    /// applying the gradients
    pub optimizer: &'a mut dyn Optimizer<'a>, // this is mut because we need to init the optimizer
                                              // when using it
    /// Weather or not the training process should be verbose, as to print the current epoch,
    /// and the current loss after applying gradients.
    pub verbose: bool,
    /// Weather or not at the end of each backprop the Model should compute its own loss and
    /// return it.
    ///
    /// If this is **true**, at the end of the **fit** method there will be returned the loss after
    /// applying the gradients.
    ///
    /// This will be necessarily true if `verbose` is set to **true**.
    pub compute_loss: bool,
    /// The amount of epochs that the Model should train for.
    pub epochs: usize,
}