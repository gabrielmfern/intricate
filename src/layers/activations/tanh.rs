use async_trait::async_trait;
use savefile::SavefileError;
use savefile_derive::Savefile;

use crate::layers::activations::activation::{ActivationLayerF64, ActivationLayerF32};
use crate::layers::layer::Layer;

#[derive(Debug, Clone, Savefile)]
pub struct TanHF64 {
    last_inputs: Vec<Vec<f64>>,
    last_outputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Savefile)]
pub struct TanHF32 {
    last_inputs: Vec<Vec<f32>>,
    last_outputs: Vec<Vec<f32>>,
}

impl TanHF64 {
    #[allow(dead_code)]

    pub fn new() -> TanHF64 {
        TanHF64 {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayerF64 for TanHF64 {
    fn function(inputs: &Vec<f64>) -> Vec<f64> {
        inputs
            .iter()
            .map(|input| input.tanh())
            .collect::<Vec<f64>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        _: usize,
    ) -> f64 {
        1.0 - self.last_outputs[sample_index][input_index].powf(2.0)
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f64>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f64>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

#[async_trait]
impl Layer<f64> for TanHF64 {
    fn get_last_inputs(&self) -> &Vec<Vec<f64>> {
        &self.last_inputs
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f64>> {
        &self.last_outputs
    }
    
    fn save(&self, _: &str, _: u32) -> Result<(), SavefileError> {
        Ok(())
    }

    fn load(&mut self, _: &str, _: u32) -> Result<(), SavefileError> {
        Ok(())
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f64>>,
        learning_rate: f64,
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<f64>>> {
        self.base_back_propagate(
            should_calculate_input_to_error_derivative,
            layer_output_to_error_derivative,
            learning_rate,
        )
    }

    async fn propagate(
        &mut self, 
        inputs: &Vec<Vec<f64>>, 
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f64>> {
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

impl TanHF32 {
    #[allow(dead_code)]

    pub fn new() -> TanHF32 {
        TanHF32 {
            last_inputs: Vec::new(),
            last_outputs: Vec::new(),
        }
    }
}

impl ActivationLayerF32 for TanHF32 {
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
        1.0_f32 - self.last_outputs[sample_index][input_index].tanh().powf(2.0_f32)
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f32>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f32>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

#[async_trait]
impl Layer<f32> for TanHF32 {
    fn get_last_inputs(&self) -> &Vec<Vec<f32>> {
        &self.last_inputs
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f32>> {
        &self.last_outputs
    }
    
    fn save(&self, _: &str, _: u32) -> Result<(), SavefileError> {
        Ok(())
    }

    fn load(&mut self, _: &str, _: u32) -> Result<(), SavefileError> {
        Ok(())
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f64,
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