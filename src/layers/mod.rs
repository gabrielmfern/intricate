use opencl3::{
    command_queue::CommandQueue, context::Context, device::cl_float, error_codes::ClError,
    memory::Buffer,
};

use crate::types::CompilationOrOpenCLError;

pub mod activations;
pub mod dense;

pub use dense::Dense;

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

    /// Gets the amount of inputs this layer is expected to receive, some layers
    /// may have just have an arbitrary value for this, like activation layers,
    /// but layers like the Dense layer just have a specific amount for the
    /// inputs_amount and the outputs_amount because of its architechture
    fn get_inputs_amount(&self) -> usize;

    /// Gets the amount of outpust this layer is expected to result in on
    /// propagation, some layers may have just have an arbitrary value for this,
    /// like activation layers, that have their outputs_amount = inputs_amount
    /// but layers like the Dense layer just have a specific amount for the
    /// inputs_amount and the outputs_amount because of its architechture
    fn get_outputs_amount(&self) -> usize;

    /// Cleans up all of the buffers saved up in the GPU
    /// for this layer
    fn clean_up_gpu_state(&mut self) -> ();

    /// Allocates the data from the GPU into the CPU
    ///
    /// does not allocate the last_inputs nor the last_outputs
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong while triying to read the data
    /// from the buffers with OpenCL.
    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError>;

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
    fn init(
        &mut self,
        queue: &'a CommandQueue,
        context: &'a Context,
    ) -> Result<(), CompilationOrOpenCLError>;

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
    fn propagate(&mut self, inputs: &Buffer<cl_float>) -> Result<&Buffer<cl_float>, ClError>;

    /// Should calculate and apply the gradients,
    /// receiving the derivatives of outputs to the loss
    /// and then return the derivatives of inputs to the loss.
    ///
    /// dE/dI <- back_propagate <- dE/dO
    ///
    /// the returning part can be disabled in case of
    /// wanting to save some computing time where
    /// there is no other layer before it.
    ///
    /// the queue and the context here is used for
    /// making OpenCL calls to run kernels
    /// (opencl functions that run in the GPU) for computations
    /// but of curse, this may run just on Rust in the CPU,
    /// so it is an Option
    ///
    /// take care with the buffer you pass into the layer_output_to_error_derivative
    /// because the buffer needs to be from the Context passed in
    /// and from when the Dense was initiated, so strictly associated with
    /// the same device everywhere here
    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Buffer<cl_float>,
        learning_rate: cl_float,
    ) -> Result<Option<Buffer<cl_float>>, ClError>;
}
