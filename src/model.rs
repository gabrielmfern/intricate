use crate::layers::layer::Layer;

pub struct ModelF64 {
    layers: Vec<Box<dyn Layer<f64>>>,
}

impl ModelF64 {
    fn new(layers: Vec<Box<dyn Layer<f64>>>) -> ModelF64 {
        ModelF64 { layers }
    }
}

