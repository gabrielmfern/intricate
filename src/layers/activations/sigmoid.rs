use std::f32::consts::E;

use async_trait::async_trait;
use savefile_derive::Savefile;

use crate::layers::activations::ActivationLayer;
use crate::layers::Layer;

#[derive(Debug, Clone, Savefile)]
pub struct Sigmoid {
    last_inputs: Vec<Vec<f32>>,
    last_outputs: Vec<Vec<f32>>,
}

impl Sigmoid {
    #[allow(dead_code)]
    pub fn new() -> Sigmoid {
        Sigmoid {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayer for Sigmoid {
    fn function(inputs: &Vec<f32>) -> Vec<f32> {
        inputs
            .iter()
            .map(|input| 1.0_f32 / (1.0_f32 - E.powf(-input)))
            .collect::<Vec<f32>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        _: usize,
    ) -> f32 {
        self.last_outputs[sample_index][input_index]
            * (1.0 - self.last_outputs[sample_index][input_index])
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f32>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f32>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

#[async_trait]
impl Layer for Sigmoid {
    fn get_last_inputs(&self) -> &Vec<Vec<f32>> {
        &self.last_inputs
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f32>> {
        &self.last_outputs
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32,
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<f32>>> {
        self.base_back_propagate(
            should_calculate_input_to_error_derivative,
            layer_output_to_error_derivative,
            learning_rate,
        )
    }

    async fn propagate(
        &mut self, 
        inputs: &Vec<Vec<f32>>, 
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
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
