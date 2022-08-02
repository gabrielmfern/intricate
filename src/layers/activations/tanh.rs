use savefile_derive::Savefile;

use crate::layers::activations::ActivationLayer;
use crate::layers::Layer;

#[derive(Debug, Clone, Savefile)]
/// The hyperbolic tangent as an activation function,
/// it squashed the outputs of the last layer between -1.0 and 1.0
/// similar to Sigmoid but not the same
pub struct TanH {
    last_inputs: Vec<Vec<f32>>,
    last_outputs: Vec<Vec<f32>>,
}

impl TanH {
    #[allow(dead_code)]

    pub fn new() -> TanH {
        TanH {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayer for TanH {
    fn function(inputs: &Vec<f32>) -> Vec<f32> {
        inputs
            .iter()
            .map(|input| input.tanh())
            .collect::<Vec<f32>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        _: usize,
    ) -> f32 {
        1.0_f32 - self.last_outputs[sample_index][input_index].powf(2.0_f32)
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f32>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f32>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

impl Layer for TanH {
    fn get_last_inputs(&self) -> &Vec<Vec<f32>> {
        &self.last_inputs
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f32>> {
        &self.last_outputs
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> Option<Vec<Vec<f32>>> {
        self.base_back_propagate(
            should_calculate_input_to_error_derivative,
            layer_output_to_error_derivative,
            learning_rate,
        )
    }

    fn propagate(
        &mut self, 
        inputs: &Vec<Vec<f32>>, 
    ) -> Vec<Vec<f32>> {
        self.base_propagate(inputs)
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