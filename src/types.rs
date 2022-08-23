//! A module containing internal data types for Intricate

use opencl3::error_codes::ClError;
use savefile_derive::Savefile;

use intricate_macros::{EnumLayer, LossFunctionEnum, FromForAllUnnamedVariants, OptimizerEnum};

use crate::{
    layers::{activations::{TanH, SoftMax, ReLU, Sigmoid}, Dense},
    loss_functions::{CategoricalCrossEntropy, MeanSquared},
    utils::{opencl::UnableToSetupOpenCLError, OpenCLState}, optimizers::Dummy,
};

#[derive(Debug)]
pub struct ProgramNotFoundError(pub String);

impl From<String> for ProgramNotFoundError {
    fn from(program: String) -> Self {
        ProgramNotFoundError(program)
    }
}

#[derive(Debug)]
pub struct KernelNotFoundError(pub String);

impl From<String> for KernelNotFoundError {
    fn from(kernel: String) -> Self {
        KernelNotFoundError(kernel)
    }
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// A simple type for initialization errors, since they can be either a straight up ClError
/// or a compilation error for some kernel which yields a type of stacktrace.
pub enum CompilationOrOpenCLError {
    /// An error that happens when compilling a OpenCL program.
    CompilationError(String),
    /// An error that happens when doing some OpenCL procedure that fails.
    OpenCLError(ClError),
    /// An error that will happen when trying to setup OpenCL
    UnableToSetupOpenCLError,
}

impl From<UnableToSetupOpenCLError> for CompilationOrOpenCLError {
    fn from(_err: UnableToSetupOpenCLError) -> Self {
        Self::UnableToSetupOpenCLError
    }
}

#[derive(Debug, LossFunctionEnum, FromForAllUnnamedVariants)]
/// All of the loss functions implemented in Intricate that a usual sequential Model can use.
#[allow(missing_docs)]
pub enum ModelLossFunction<'a> {
    MeanSquared(MeanSquared<'a>),
    CategoricalCrossEntropy(CategoricalCrossEntropy<'a>),
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
pub enum GradientDescent {}

#[derive(Debug, OptimizerEnum, FromForAllUnnamedVariants)]
pub enum PossibleOptimizer<'a> {
    Dummy(Dummy<'a>),
}

/// A struct that defines the options for training a Model.
pub struct TrainingOptions<'a> {
    /// The amount at which the gradients should be multiplied as to have a
/// gradual learning experience for the Model.
    pub loss_algorithm: ModelLossFunction<'a>,
    /// The loss function that will be used for calculating how **wrong** the Model 
    /// was after some prediction over many samples.
    pub initial_learning_rate: f32,
    pub gradient_descent_method: GradientDescent,
    pub optimizer: PossibleOptimizer<'a>,
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