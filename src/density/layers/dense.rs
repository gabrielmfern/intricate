use crate::density::layers::layer::{InternalLayer, F64Layer};

pub struct Dense { 
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,

    last_inputs: Vec<Vec<f64>>,
    last_outputs: Vec<Vec<f64>>,
}

impl InternalLayer<f64> for Dense {
    fn get_last_inputs(&self) -> Vec<Vec<f64>> { self.last_inputs.to_vec() }
    fn get_last_outputs(&self) -> Vec<Vec<f64>> { self.last_outputs.to_vec() }
    
    fn propagate(&self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        Vec::new()
    }

    fn back_propagate(&self, output_gradients: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        Vec::new()
    }
}

impl F64Layer for Dense { }

