//! A module containing internal data types for Intricate

use opencl3::error_codes::ClError;
use savefile_derive::Savefile;

use intricate_macros::{EnumLayer, LossFunctionEnum, ErrorsEnum};

use crate::{
    layers::{activations::{TanH, SoftMax, ReLU, Sigmoid}, Dense},
    loss_functions::{CategoricalCrossEntropy, MeanSquared},
    utils::{opencl::UnableToSetupOpenCLError, OpenCLState},
};

#[derive(Debug, ErrorsEnum)]
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

#[derive(Debug, LossFunctionEnum)]
/// All of the loss functions implemented in Intricate that a usual sequential Model can use.
#[allow(missing_docs)]
pub enum ModelLossFunction<'a> {
    MeanSquared(MeanSquared<'a>),
    CategoricalCrossEntropy(CategoricalCrossEntropy<'a>),
}

#[derive(Debug, Savefile, EnumLayer)]
/// All of the possible layers that a usual Sequential Model can have.
#[allow(missing_docs)]
pub enum ModelLayer<'a> {
    Dense(Dense<'a>),
    TanH(TanH<'a>),
    SoftMax(SoftMax<'a>),
    ReLU(ReLU<'a>),
    Sigmoid(Sigmoid<'a>),
}

/// A struct that defines the options for training a Model.
pub struct TrainingOptions<'a> {
    /// The amount at which the gradients should be multiplied as to have a
/// gradual learning experience for the Model.
    pub loss_algorithm: ModelLossFunction<'a>,
    // TODO: implement optimizers
    /// The loss function that will be used for calculating how **wrong** the Model 
    /// was after some prediction over many samples.
    pub learning_rate: f32,
    /// Weather or not the training process should be verbose, as to print the current epoch, 
    /// and the current loss after applying gradients.
    pub should_print_information: bool,
    /// The amount of epochs that the Model should train for.
    pub epochs: usize,
}