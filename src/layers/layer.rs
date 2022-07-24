use async_trait::async_trait;

/// T here is a number type
/// so that we can have multiple numbers types to save RAM in simple Neural Networks
/// but still have precision on neural networks that use an activation such as
/// softmax that has such steep values
#[async_trait]
pub trait Layer<T> {
    fn get_last_inputs(&self) -> Vec<Vec<T>>;
    fn get_last_outputs(&self) -> Vec<Vec<T>>;

    fn get_inputs_amount(&self) -> usize;
    fn get_outputs_amount(&self) -> usize;
    
    /// Should calculate the outputs of the layer based on the inputs
    /// 
    /// is asynchronous so that communication between the gpu and the cpu
    /// can happen normally on this function if needed in the layer
    async fn propagate(
        &mut self, 
        inputs: &Vec<Vec<T>>,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Vec<Vec<T>>;

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
    async fn back_propagate(
        &mut self, 
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<T>>,
        learning_rate: f64,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<T>>>;
}

