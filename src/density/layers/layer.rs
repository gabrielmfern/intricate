/// Should not be used, T here is a number type
/// so that we can have multiple numbers types to save RAM in simple Neural Networks
/// but still have precision on neural networks that use an activation such as
/// softmax that has such steep values
pub trait InternalLayer<T> {
    fn get_last_inputs(&self) -> Vec<Vec<T>>;
    fn get_last_outputs(&self) -> Vec<Vec<T>>;
    
    fn propagate(&self, inputs: &Vec<Vec<T>>) -> Vec<Vec<T>>;

    fn back_propagate(&self, output_gradients: &Vec<Vec<T>>) -> Vec<Vec<T>>;
}

pub trait F32Layer: InternalLayer<f32> { }

pub trait F64Layer: InternalLayer<f64> { }

