//! A module containing internal data types for Intricate

use opencl3::{
    device::cl_float, error_codes::ClError,
    memory::Buffer,
};
use savefile_derive::Savefile;

use intricate_macros::EnumLayer;

use crate::{
    layers::{activations::{TanH, SoftMax, ReLU, Sigmoid}, Dense},
    loss_functions::{CategoricalCrossEntropy, LossFunction, MeanSquared},
    utils::{opencl::UnableToSetupOpenCLError, OpenCLState},
};

/// A simple type for initialization errors, since they can be either a straight up ClError
/// or a compilation error for some kernel which yields a type of stacktrace.
#[derive(Debug)]
pub enum CompilationOrOpenCLError {
    CompilationError(String),
    OpenCLError(ClError),
    UnableToSetupOpenCLError,
}

impl From<ClError> for CompilationOrOpenCLError {
    fn from(err: ClError) -> Self {
        CompilationOrOpenCLError::OpenCLError(err)
    }
}

impl From<String> for CompilationOrOpenCLError {
    fn from(err: String) -> Self {
        CompilationOrOpenCLError::CompilationError(err)
    }
}

impl From<UnableToSetupOpenCLError> for CompilationOrOpenCLError {
    fn from(_err: UnableToSetupOpenCLError) -> Self {
        Self::UnableToSetupOpenCLError
    }
}


#[derive(Debug)]
pub enum ModelLossFunction<'a> {
    MeanSquared(MeanSquared<'a>),
    CategoricalCrossEntropy(CategoricalCrossEntropy<'a>),
}


impl<'a> From<MeanSquared<'a>> for ModelLossFunction<'a> {
    fn from(loss: MeanSquared<'a>) -> Self {
        ModelLossFunction::MeanSquared(loss)
    }
}

impl<'a> From<CategoricalCrossEntropy<'a>> for ModelLossFunction<'a> {
    fn from(loss: CategoricalCrossEntropy<'a>) -> Self {
        ModelLossFunction::CategoricalCrossEntropy(loss)
    }
}

pub struct TrainingOptions<'a> {
    pub loss_algorithm: ModelLossFunction<'a>,
    // TODO: implement optimizers
    pub learning_rate: f32,
    pub should_print_information: bool,
    pub epochs: usize,
}

impl<'a> LossFunction<'a> for ModelLossFunction<'a> {
    fn compute_loss(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<f32, ClError> {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => {
                lossfn.compute_loss(output_samples, expected_outputs, samples_amount)
            }
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => {
                lossfn.compute_loss(output_samples, expected_outputs, samples_amount)
            }
        }
    }

    fn init(
        &mut self,
        opencl_state: &'a mut OpenCLState,
    ) -> Result<(), CompilationOrOpenCLError> {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => lossfn.init(opencl_state),
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => lossfn.init(opencl_state),
        }
    }

    fn compute_loss_derivative_with_respect_to_output_samples(
        &self,
        output_samples: &Buffer<cl_float>,
        expected_outputs: &Buffer<cl_float>,
        samples_amount: usize,
    ) -> Result<Buffer<cl_float>, ClError> {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => lossfn
                .compute_loss_derivative_with_respect_to_output_samples(
                    output_samples,
                    expected_outputs,
                    samples_amount,
                ),
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => lossfn
                .compute_loss_derivative_with_respect_to_output_samples(
                    output_samples,
                    expected_outputs,
                    samples_amount,
                ),
        }
    }
}

#[derive(Debug, Savefile, EnumLayer)]
pub enum ModelLayer<'a> {
    Dense(Dense<'a>),
    TanH(TanH<'a>),
    SoftMax(SoftMax<'a>),
    ReLU(ReLU<'a>),
    Sigmoid(Sigmoid<'a>),
}