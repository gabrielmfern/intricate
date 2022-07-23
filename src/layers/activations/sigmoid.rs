use std::f64::consts::E;

use async_trait::async_trait;

use crate::layers::activations::activation::ActivationLayerF64;
use crate::layers::layer::Layer;

#[derive(Debug, Clone)]
pub struct SigmoidF64 {
    last_inputs: Vec<Vec<f64>>,
    last_outputs: Vec<Vec<f64>>,
}

impl SigmoidF64 {
    #[allow(dead_code)]

    pub fn new() -> SigmoidF64 {
        SigmoidF64 {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayerF64 for SigmoidF64 {
    fn function(inputs: &Vec<f64>) -> Vec<f64> {
        inputs
            .iter()
            .map(|input| 1.0 / (1.0 - E.powf(-input)))
            .collect::<Vec<f64>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        _: usize,
    ) -> f64 {
        self.last_outputs[sample_index][input_index]
            * (1.0 - self.last_outputs[sample_index][input_index])
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f64>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f64>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

#[async_trait]
impl Layer<f64> for SigmoidF64 {
    fn get_last_inputs(&self) -> Vec<Vec<f64>> {
        self.last_inputs.to_vec()
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) -> Option<Vec<Vec<f64>>> {
        self.base_back_propagate(
            should_calculate_input_to_error_derivative,
            layer_output_to_error_derivative,
            learning_rate,
        )
    }

    async fn propagate(&mut self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.base_propagate(inputs)
    }

    fn get_last_outputs(&self) -> Vec<Vec<f64>> {
        self.last_outputs.to_vec()
    }

    fn get_inputs_amount(&self) -> usize {
        if self.last_inputs.is_empty() {
            0
        } else {
            self.last_inputs[0].len()
        }
    }

    fn get_outputs_amount(&self) -> usize {
        if self.last_outputs.is_empty() {
            0
        } else {
            self.last_outputs[0].len()
        }
    }
}
