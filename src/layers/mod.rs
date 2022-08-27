//! The module that contains all layers that Intricate currently implements.
//! As of v0.3.0, Intricate has only the Dense type of layer, but has the activation functions
//! which are used as layers in Intricate.

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    device::cl_float,
    error_codes::ClError,
    memory::{Buffer, CL_MEM_READ_ONLY},
};

use crate::{
    optimizers::{OptimizationError, Optimizer},
    utils::{opencl::{EnsureKernelsAndProgramError, BufferOperationError, BufferConversionError}, OpenCLState, BufferOperations}, types::{KernelNotFoundError, ProgramNotFoundError, SyncDataError},
};

pub mod activations;
pub mod dense;

pub use dense::Dense;

use self::{activations::compile_activations, dense::compile_dense};

pub(crate) fn compile_layers(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    compile_dense(opencl_state)?;
    compile_activations(opencl_state)?;

    Ok(())
}

#[derive(Debug)]
/// A simple struct that contains the gradients for a certain parameter and weather or not these
/// gradients should be optimized.
pub struct Gradient {
    /// The name of the parameter to keep track of what is updated in the optimizer
    pub parameter_id: String,

    /// The actual gradients of the parameter.
    pub value: Buffer<cl_float>,

    /// Weather or not the gradients should be optimized when computing the update vectors.
    pub optimizable: bool,
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum that contains all errors that can happen while trying to compute update vectors.
pub enum UpdateVectorsComputationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),
    /// Happens when the computation of the update vector made by the optimizer goes wrong.
    Optimizer(OptimizationError),
    /// Happens when a buffer operation goes wrong.
    BufferOperation(BufferOperationError),
}

pub(crate) fn compute_update_vectors<'a>(
    optimizer: &mut dyn Optimizer<'a>,
    all_gradients: &[Gradient],
    layer_index: usize,
    state: &OpenCLState,
) -> Result<Vec<Buffer<cl_float>>, UpdateVectorsComputationError> {
    let mut update_vectors: Vec<Buffer<cl_float>> = Vec::with_capacity(all_gradients.len());

    for gradients in all_gradients.iter() {
        if gradients.optimizable {
            update_vectors.push(optimizer.compute_update_vectors(
                &gradients.value,
                gradients.parameter_id.to_string(),
                layer_index,
            )?);
        } else {
            update_vectors.push(gradients.value.clone(CL_MEM_READ_ONLY, state)?);
        }
    }

    Ok(update_vectors)
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying to propagate a layer.
pub enum LayerPropagationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when a program could not be found inside of the OpenCLState.
    ProgramNotFound(ProgramNotFoundError),
    /// Happens when a kernel could not be found inside of the program.
    KernelNotFound(KernelNotFoundError),
    /// Happens when a buffer operation goes wrong.
    BufferOperation(BufferOperationError),

    /// Happens if the amounts of inputs per sample is not equivalent to the amount of actual
    /// inputs
    InputsDontMatchExpectedShape,

    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueueFound,
    /// Happens when there is no device in the OpenCLState.
    NoDeviceFound,

    /// Happens when the layer being propagate was not initialized before propagating.
    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying to compute gradients for a
/// layer.
pub enum LayerGradientComputationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when a program could not be found inside of the OpenCLState.
    ProgramNotFound(ProgramNotFoundError),
    /// Happens when a kernel could not be found inside of the program.
    KernelNotFound(KernelNotFoundError),

    /// Happens when the derivatives do not match the expected shape based on the input_amount and
    /// outputs_amount.
    DerivativesDontMatchExpectedShape,

    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueueFound,
    /// Happens when there is no device in the OpenCLState.
    NoDeviceFound,

    /// Happens when the layer being propagate was not initialized before propagating.
    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying to apply some calculated
/// gradients to a layer.
pub enum LayerGradientApplicationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when a program could not be found inside of the OpenCLState.
    ProgramNotFound(ProgramNotFoundError),
    /// Happens when a kernel could not be found inside of the program.
    KernelNotFound(KernelNotFoundError),

    /// Happens when a buffer operation goes wrong.
    BufferOperation(BufferOperationError),
    /// Happens when something goes wrong while trying to compute update vectors for each gradient.
    UpdateVectorsComputation(UpdateVectorsComputationError),

    /// Happens when the gradients given to the gradient application method do not match the
    /// expected amount of gradients
    GradientsDontMatchExpectedShape,

    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueueFound,
    /// Happens when there is no device in the OpenCLState.
    NoDeviceFound,

    /// Happens when the layer being propagate was not initialized before propagating.
    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying compute the derivatives of the
/// loss with respect to the inputs of a layer.
pub enum LayerLossToInputDifferentiationError {
    /// Happens when something goes wrong with OpenCL.
    OpenCL(ClError),

    /// Happens when a program could not be found inside of the OpenCLState.
    ProgramNotFound(ProgramNotFoundError),
    /// Happens when a kernel could not be found inside of the program.
    KernelNotFound(KernelNotFoundError),

    /// Happens when the derivatives do not match the expected shape based on the input_amount and
    /// outputs_amount.
    DerivativesDontMatchExpectedShape,
    /// Happens when the layer has not propagated before calculating the derivatives if the outputs
    /// are necessary.
    HasNotPropagatedBeforeCalculation,

    /// Happens when there is no command queue in the OpenCLState.
    NoCommandQueueFound,
    /// Happens when the layer being propagate was not initialized before propagating.
    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying to optimize the parameters of
/// a layer using the `optimizer_parameters` function of a Optimizer.
pub enum ParametersOptimizationError {
    /// Happens when something goes wrong in optimization.
    Optimization(OptimizationError),
    /// Happens when an optimizable parameter is empty.
    EmptyParameter(String),
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An enum containing all of the errors that can happen when trying to initialize a Layer.
pub enum LayerInitializationError {
    /// Happens when something goes wrong trying to convert the Layer's parameters into OpenCL
    /// Buffers.
    BufferConversion(BufferConversionError),
    /// Happens when a parameter that is going to be converted into a OpenCL Buffer is empty. Such
    /// as an empty Vec.
    EmptyParameter(String),
    /// Happens when there is no OpenCL Command Queue when needed.
    NoCommandQueue,
}

/// A trait implemented by Intricate that is implemented in every struct that represents a Model
/// Layer.
/// A layer in Intricate can be defined basically as a function that can take some inputs and gives
/// outputs however it sees fit, but, that also backpropagates using derivatives of the outputs to
/// the loss of the whole Model, and returning derivatives of the loss with respect to the inputs
/// of the layer.
pub trait Layer<'a> {
    /// Gets the last input samples that were used in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// inputs on propagate
    ///
    /// It is optional because the data of the Layer may not be stored currently in the GPU,
    /// perhaps after loading the layer from a file.
    fn get_last_inputs(&self) -> Option<&Buffer<cl_float>>;

    /// Gets the last output samples that were the result in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// ouputs on propagate
    ///
    /// It is optional because the data of the Layer may not be stored currently in the GPU,
    /// perhaps after loading the layer from a file.
    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>>;

    /// Gets the amount of inputs this layer is expected to receive. 
    ///
    /// Some layers may have just have an arbitrary value for this, like activation layers,
    /// but layers like the Dense layer just have a specific amount for the
    /// inputs_amount and the outputs_amount because of its architechture
    fn get_inputs_amount(&self) -> usize;

    /// Gets the amount of outpust this layer is expected to result in on
    /// propagation. 
    ///
    /// Some layers may have just have an arbitrary value for this,
    /// like activation layers, that have their outputs_amount = inputs_amount
    /// but layers like the Dense layer just have a specific amount for the
    /// inputs_amount and the outputs_amount because of its architechture.
    fn get_outputs_amount(&self) -> usize;

    /// Cleans up all of the buffers saved up in the Device
    /// for this layer
    fn clean_up_gpu_state(&mut self) -> ();

    /// Allocates the data from the OpenCL's device that contains the buffers of the data for the
    /// layer into the host with Vec's of the values.
    ///
    /// does not allocate the last_inputs nor the last_outputs
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong while triying to read the data
    /// from the buffers with OpenCL.
    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), SyncDataError>;

    /// Sends the important information of the current layer to the GPU
    /// as to be used in the propagation and back propagation
    ///
    /// mostly used after loading the layer using load_file and then
    /// there is a need to resend the data to the GPU since Savefile doesn't
    /// load the data into the GPU by itself
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong while 
    /// allocating buffers into the device of the queue.
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), LayerInitializationError>;

    /// Should calculate the outputs of the layer based on the inputs
    ///
    /// take care with the buffer you pass into the propagate
    /// because the buffer needs to be from the Context passed in
    /// and from when the Dense was initiated, so strictly associated with
    /// the same device everywhere here
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong while executing the layer's
    /// kernels.
    fn propagate(&mut self, inputs: &Buffer<cl_float>) -> Result<&Buffer<cl_float>, LayerPropagationError>;

    /// Computes the gradients that will be used to calculate the update vectors that will then be
    /// applied to the
    ///
    /// # Params
    ///
    /// - **layer_output_to_error_derivative**: The reference to the the buffer in the device
    /// containing the derivatives of the loss with respect to the outputs of the layer.
    ///
    /// take care with the buffer you pass into the **layer_output_to_error_derivative**
    /// because the buffer needs to be from the Context passed in
    /// and from when the Dense was initiated, so strictly associated with
    /// the same device everywhere here
    fn compute_gradients(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Vec<Gradient>, LayerGradientComputationError>;

    /// Tweaks all of the parameters of the Layer based on the optimizer's choices.
    ///
    /// # Errors
    ///
    /// This function will return an error if the Optimizer is unable to do it's calculations or if
    /// a parameter that is going to be optimized has no value.
    fn optimize_parameters(
        &mut self,
        optimizer: &dyn Optimizer<'a>,
        layer_index: usize,
    ) -> Result<(), ParametersOptimizationError>;

    /// Applies all of the gradients given by **compute_gradients** of the current layer using a
    /// certain optimizer.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Something goes wrong with OpenCL;
    /// - Something goes wrong while computing update vectors;
    /// - Something goes wrong inside a buffer operation;
    /// - A required program was not found;
    /// - A required kernel was not found;
    /// - There is no command queue;
    /// - There is no device;
    /// - The layer was not initialized.
    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: &[Gradient],
        optimizer: &mut dyn Optimizer<'a>,
        layer_model_index: usize,
    ) -> Result<(), LayerGradientApplicationError>;

    /// Computes the derivatives of the Model's loss with respect to all of the inputs in each
    /// sample of the batch.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Something goes wrong in OpenCL.
    /// - A required program was not found.
    /// - A required kernel was not found in a program.
    /// - The layer has not been propagated before this method was called.
    /// - The layer was not initialized.
    /// - There are no drivers for OpenCL.
    fn compute_loss_to_input_derivatives(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, LayerLossToInputDifferentiationError>;
}