pub mod dense;
pub mod activations;
pub mod dense_gpu;

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
    fn propagate(
        &mut self, 
        inputs: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>>;

    /// Should calculate and apply the gradients,
    /// receiving the derivatives of outputs to the loss
    /// and then return the derivatives of inputs to the loss.
    ///
    /// dE/dI <- back_propagate <- dE/dO
    ///
    /// the returning part can be disabled in case of 
    /// wanting to save some computing time where 
    /// the layer is not used.
    /// 
    /// is asynchronous so that communication between the gpu and the cpu
    /// can happen normally on this function if needed in the layer
    fn back_propagate(
        &mut self, 
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> Option<Vec<Vec<f32>>>;
}