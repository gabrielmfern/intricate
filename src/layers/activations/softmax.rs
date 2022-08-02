use std::f64::consts::E;

use savefile_derive::Savefile;

use crate::layers::activations::ActivationLayer;
use crate::layers::Layer;
use crate::utils::vector_operations::VectorOperations;

#[derive(Debug, Clone, Savefile)]
/// The SoftMax function, a good function for solving categorical problems
/// because it's returning value will be very close to 1 where the value
/// is very close to the largest of the outputs in the sample and very close
/// to 0 if it is just a bit far from the max among them
pub struct SoftMax {
    last_inputs: Vec<Vec<f32>>,
    last_outputs: Vec<Vec<f32>>,
}

impl SoftMax {
    #[allow(dead_code)]

    pub fn new() -> SoftMax {
        SoftMax {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayer for SoftMax {
    fn function(inputs: &Vec<f32>) -> Vec<f32> {
        let max_input = inputs.iter().copied().fold(f32::NAN, f32::max);
        let exponentials: &Vec<f32> = &inputs.subtract_number(max_input).from_powf(E);
        let total = &exponentials.iter().sum::<f32>();

        inputs
            .iter()
            .enumerate()
            .map(|(i, _)| exponentials[i] / total)
            .collect::<Vec<f32>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        output_index: usize,
    ) -> f32 {
        if input_index == output_index {
            self.last_outputs[sample_index][input_index]
                * (1.0_f32 - self.last_outputs[sample_index][output_index])
        } else {
            -self.last_outputs[sample_index][input_index]
                * self.last_outputs[sample_index][output_index]
        }
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f32>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f32>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

impl Layer for SoftMax {
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