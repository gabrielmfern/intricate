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
    CompilationError(String),
    OpenCLError(ClError),
    UnableToSetupOpenCLError,
}

impl From<UnableToSetupOpenCLError> for CompilationOrOpenCLError {
    fn from(_err: UnableToSetupOpenCLError) -> Self {
        Self::UnableToSetupOpenCLError
    }
}

#[derive(Debug, LossFunctionEnum)]
/// All of the loss functions implemented in Intricate that a usual sequential Model can use.
pub enum ModelLossFunction<'a> {
    MeanSquared(MeanSquared<'a>),
    CategoricalCrossEntropy(CategoricalCrossEntropy<'a>),
}

#[derive(Debug, Savefile, EnumLayer)]
/// All of the possible layers that a usual Sequential Model can have.
pub enum ModelLayer<'a> {
    Dense(Dense<'a>),
    TanH(TanH<'a>),
    SoftMax(SoftMax<'a>),
    ReLU(ReLU<'a>),
    Sigmoid(Sigmoid<'a>),
}

pub struct TrainingOptions<'a> {
    pub loss_algorithm: ModelLossFunction<'a>,
    // TODO: implement optimizers
    pub learning_rate: f32,
    pub should_print_information: bool,
    pub epochs: usize,
}