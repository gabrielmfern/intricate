use opencl3::{device::cl_float, memory::Buffer, error_codes::ClError, command_queue::CommandQueue, context::Context};

pub mod activations;
pub mod dense;
pub mod dense_gpu;

/// A layer can be defined basically as function receiving some input
/// and giving an output, something can be called a 'Layer' if it does that
/// but the OpenCLLayer is expected to use OpenCL and the GPU to accelerate
/// the computations in the layer
pub trait OpenCLLayer<'a> {
    /// Gets the last input samples that were used in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// inputs on propagate
    fn get_last_inputs(&self) -> Option<&'a Buffer<cl_float>>;

    /// Gets the last output samples that were the result in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// ouputs on propagate
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
    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError>;

    /// Sends the important information of the current layer to the GPU
    /// as to be used in the propagation and back propagation
    ///
    /// mostly used after loading the layer using load_file and then
    /// there is a need to resend the data to the GPU since Savefile doesn't
    /// load the data into the GPU by itself
    fn send_to_gpu(
        &mut self,
        queue: &'a CommandQueue,
        context: &'a Context,
    ) -> Result<(), ClError>;

    /// Should calculate the outputs of the layer based on the inputs
    ///
    /// is asynchronous so that communication between the gpu and the cpu
    /// can happen normally on this function if needed in the layer
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
    fn propagate(
        &mut self,
        inputs: &'a Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, ClError>;

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

/// A layer can be defined basically as function receiving some input
/// and giving an output, something can be called a 'Layer' if it does that
pub trait Layer {
    /// Gets the last input samples that were used in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// inputs on propagate
    fn get_last_inputs(&self) -> &Vec<Vec<f32>>;

    /// Gets the last output samples that were the result in the 'propagate' method,
    /// having this getter forces a struct that implements Layer to save its
    /// ouputs on propagate
    fn get_last_outputs(&self) -> &Vec<Vec<f32>>;

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

    /// Should calculate the outputs of the layer based on the inputs
    ///
    /// is asynchronous so that communication between the gpu and the cpu
    /// can happen normally on this function if needed in the layer
    ///
    /// the queue and the context here is used for
    /// making OpenCL calls to run kernels
    /// (opencl functions that run in the GPU) for computations
    /// but of curse, this may run just on Rust in the CPU,
    /// so it is an Option
    fn propagate(&mut self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>>;

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
    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> Option<Vec<Vec<f32>>>;
}