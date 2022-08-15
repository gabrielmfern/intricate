//! A module containing internal data types for Intricate

use opencl3::{
    command_queue::CommandQueue, context::Context, device::cl_float, error_codes::ClError,
    memory::Buffer,
};
use savefile_derive::Savefile;

use intricate_macros::EnumLayer;

use crate::{
    layers::{activations::TanH, Dense, Layer},
    loss_functions::{CategoricalCrossEntropy, LossFunction, MeanSquared},
    utils::opencl::UnableToSetupOpenCLError,
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

#[derive(Debug, Savefile, EnumLayer)]
pub enum ModelLayer<'a> {
    Dense(Dense<'a>),
    TanH(TanH<'a>),
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
        context: &'a Context,
        queue: &'a CommandQueue,
    ) -> Result<(), CompilationOrOpenCLError> {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => lossfn.init(context, queue),
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => lossfn.init(context, queue),
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

impl<'a> Layer<'a> for ModelLayer<'a> {
    fn get_last_inputs(&self) -> Option<&Buffer<cl_float>> {
        match self {
            ModelLayer::Dense(layer) => layer.get_last_inputs(),
            ModelLayer::TanH(layer) => layer.get_last_inputs(),
        }
    }

    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>> {
        match self {
            ModelLayer::Dense(layer) => layer.get_last_outputs(),
            ModelLayer::TanH(layer) => layer.get_last_outputs(),
        }
    }

    fn get_inputs_amount(&self) -> usize {
        match self {
            ModelLayer::Dense(layer) => layer.get_inputs_amount(),
            ModelLayer::TanH(layer) => layer.get_inputs_amount(),
        }
    }

    fn get_outputs_amount(&self) -> usize {
        match self {
            ModelLayer::Dense(layer) => layer.get_outputs_amount(),
            ModelLayer::TanH(layer) => layer.get_outputs_amount(),
        }
    }

    fn init(
        &mut self,
        queue: &'a CommandQueue,
        context: &'a Context,
    ) -> Result<(), CompilationOrOpenCLError> {
        match self {
            ModelLayer::Dense(layer) => layer.init(queue, context),
            ModelLayer::TanH(layer) => layer.init(queue, context),
        }
    }

    fn clean_up_gpu_state(&mut self) -> () {
        match self {
            ModelLayer::Dense(layer) => layer.clean_up_gpu_state(),
            ModelLayer::TanH(layer) => layer.clean_up_gpu_state(),
        }
    }

    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError> {
        match self {
            ModelLayer::Dense(layer) => layer.sync_data_from_gpu_with_cpu(),
            ModelLayer::TanH(layer) => layer.sync_data_from_gpu_with_cpu(),
        }
    }

    fn propagate(&mut self, inputs: &Buffer<cl_float>) -> Result<&Buffer<cl_float>, ClError> {
        match self {
            ModelLayer::Dense(layer) => layer.propagate(inputs),
            ModelLayer::TanH(layer) => layer.propagate(inputs),
        }
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Buffer<cl_float>,
        learning_rate: cl_float,
    ) -> Result<Option<Buffer<cl_float>>, ClError> {
        match self {
            ModelLayer::Dense(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            ModelLayer::TanH(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
        }
    }
}

impl<'a> From<Dense<'a>> for ModelLayer<'a> {
    fn from(layer: Dense<'a>) -> Self {
        ModelLayer::Dense(layer)
    }
}

impl<'a> From<TanH<'a>> for ModelLayer<'a> {
    fn from(layer: TanH<'a>) -> Self {
        ModelLayer::TanH(layer)
    }
}