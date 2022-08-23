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
    utils::{opencl::{EnsureKernelsAndProgramError, BufferOperationError}, OpenCLState, BufferOperations}, types::{KernelNotFoundError, ProgramNotFoundError, PossibleOptimizer},
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
pub struct Gradient {
    pub value: Buffer<cl_float>,
    pub optimizable: bool,
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum UpdateVectorsComputationError {
    OpenCL(ClError),
    GradientOptimzation(OptimizationError),
    BufferOperation(BufferOperationError),
    NoCommandQueueFound,
}

pub fn compute_update_vectors(
    optimizer: &PossibleOptimizer,
    all_gradients: &[Gradient],
    state: &OpenCLState,
) -> Result<Vec<Buffer<cl_float>>, UpdateVectorsComputationError> {
    if let Some(queue) = state.queues.first() {
        let mut update_vectors: Vec<Buffer<cl_float>> = Vec::with_capacity(all_gradients.len());

        let context = &state.context;

        for (i, gradients) in all_gradients.iter().enumerate() {
            if gradients.optimizable {
                update_vectors[i] = optimizer.compute_update_vectors(&gradients.value)?;
            } else {
                update_vectors[i] = gradients.value.clone(CL_MEM_READ_ONLY, state)?;
            }
        }

        Ok(update_vectors)
    } else {
        Err(UpdateVectorsComputationError::NoCommandQueueFound)
    }
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum LayerPropagationError {
    OpenCL(ClError),

    ProgramNotFound(ProgramNotFoundError),
    KernelNotFound(KernelNotFoundError),
    BufferOperation(BufferOperationError),

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum LayerGradientComputationError {
    OpenCL(ClError),

    ProgramNotFound(ProgramNotFoundError),
    KernelNotFound(KernelNotFoundError),

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum LayerGradientApplicationError {
    OpenCL(ClError),

    ComputeUpdateVectors(LayerGradientComputationError),
    BufferOperation(BufferOperationError),
    UpdateVectorsComputation(UpdateVectorsComputationError),

    ProgramNotFound(ProgramNotFoundError),
    KernelNotFound(KernelNotFoundError),

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum LayerSyncDataError {
    OpenCL(ClError),
    LayerNotInitialized,
    NotAllocatedInDevice {
        field_name: String
    },
    NoCommandQueue,
}

#[derive(Debug, FromForAllUnnamedVariants)]
pub enum LayerLossToInputDifferentiationError {
    OpenCL(ClError),
    LayerNotInitialized,
    NoCommandQueueFound,
    HasNotPropagatedBeforeCalculation,
    ProgramNotFound(ProgramNotFoundError),
    KernelNotFound(KernelNotFoundError),
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
    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), LayerSyncDataError>;

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
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError>;

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

    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: &[Gradient],
        optimizer: &PossibleOptimizer,
    ) -> Result<(), LayerGradientApplicationError>;

    fn compute_loss_to_input_derivatives(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, LayerLossToInputDifferentiationError>;
}