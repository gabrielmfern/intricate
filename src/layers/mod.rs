//! The module that contains all layers that Intricate currently implements.
//! As of v0.3.0, Intricate has only the Dense type of layer, but has the activation functions
//! which are used as layers in Intricate.

use intricate_macros::ErrorsEnum;
use opencl3::{
    command_queue::CommandQueue,
    device::cl_float,
    error_codes::ClError,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY},
};

use crate::{
    optimizers::{OptimizationError, Optimizer},
    utils::{opencl::EnsureKernelsAndProgramError, OpenCLState, BufferOperations},
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

#[derive(Debug, ErrorsEnum)]
pub enum GradientComputationError {
    OpenCL(ClError),
}

#[derive(Debug)]
pub struct Gradient {
    pub value: Buffer<cl_float>,
    pub optimizable: bool,
}

#[derive(Debug, ErrorsEnum)]
pub enum ComputeVectorComputationError {
    OpenCL(ClError),
    GradientOptimzationError(OptimizationError),
    UninitializedState,
    NoCommandQueueFound,
}

pub trait Gradients<'a> {
    fn get_gradients(&self) -> &[Gradient];

    fn get_opencl_state(&self) -> Option<&'a OpenCLState>;

    fn compute_update_vectors(
        &self,
        optimizer: dyn Optimizer,
    ) -> Result<Vec<Buffer<cl_float>>, ComputeVectorComputationError> {
        if let Some(state) = self.get_opencl_state() {
            if let Some(queue) = state.queues.first() {
                let all_gradients = self.get_gradients();
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
                Err(ComputeVectorComputationError::NoCommandQueueFound)
            }
        } else {
            Err(ComputeVectorComputationError::UninitializedState)
        }
    }
}

#[derive(Debug, ErrorsEnum)]
pub enum LayerPropagationError {
    OpenCL(ClError),

    ProgramNotFound,
    KernelNotFound,

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

#[derive(Debug, ErrorsEnum)]
pub enum LayerGradientComputationError {
    OpenCL(ClError),

    ProgramNotFound,
    KernelNotFound,

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

#[derive(Debug, ErrorsEnum)]
pub enum LayerGradientApplicationError {
    OpenCL(ClError),

    ProgramNotFound,
    KernelNotFound,

    NoCommandQueueFound,
    NoDeviceFound,

    LayerNotInitialized
}

/// A trait implemented by Intricate that is implemented in every struct that represents a Model
/// Layer.
/// A layer in Intricate can be defined basically as a function that can take some inputs and gives
/// outputs however it sees fit, but, that also backpropagates using derivatives of the outputs to
/// the loss of the whole Model, and returning derivatives of the loss with respect to the inputs
/// of the layer.
pub trait Layer<'a, LayerGradients>
where
    LayerGradients: Gradients<'a>,
{
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
    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), ClError>;

    /// Sends the important information of the current layer to the GPU
    /// as to be used in the propagation and back propagation
    ///
    /// mostly used after loading the layer using load_file and then
    /// there is a need to resend the data to the GPU since Savefile doesn't
    /// load the data into the GPU by itself
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong while trying to compile and
    /// build the OpenCL programs or while allocating buffers into the device of the queue.
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
    ) -> Result<LayerGradients, LayerGradientComputationError>;

    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: LayerGradients,
    ) -> Result<(), LayerGradientApplicationError>;
}
