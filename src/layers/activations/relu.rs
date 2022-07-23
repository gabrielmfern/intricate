
use crate::layers::activations::activation::ActivationLayer;
use crate::layers::layer::Layer;

#[derive(Debug, Clone)]
pub struct ReLU {
    last_inputs: Vec<Vec<f64>>,
    last_outputs: Vec<Vec<f64>>,
}

impl ReLU {
    fn new() -> ReLU {
        ReLU {
            last_outputs: Vec::new(),
            last_inputs: Vec::new()
        }
    }
}

#[test]
fn should_be_0_when_x_is_negative() {
    let x: Vec<f64> = Vec::from([-30.0, -40.0, -1.0, -0.3, -0.99]);
    
    let mut activation_layer = ReLU::new();
    let actual_outputs = activation_layer.propagate(&Vec::from([x]));

    let expected_outputs = Vec::from([Vec::from([0.0,0.0,0.0,0.0,0.0])]);

    assert_eq!(actual_outputs, expected_outputs);
}

#[test]
fn should_be_x_when_x_is_larger_than_negative_one() {
    let x: Vec<f64> = Vec::from([-30.0, 40.0, 21.0, -0.3, -0.99]);
    
    let mut activation_layer = ReLU::new();
    let actual_outputs = activation_layer.propagate(&Vec::from([x]));

    let expected_outputs = Vec::from([Vec::from([0.0,40.0,21.0,0.0,0.0])]);

    assert_eq!(actual_outputs, expected_outputs);
}

#[test]
fn differential_should_return_correct_value() {
    let x: Vec<f64> = Vec::from([-30.0, 40.0, 21.0, -0.3, -0.99]);

    let mut activation_layer = ReLU::new();
    activation_layer.propagate(&Vec::from([x.to_vec()]));

    let output_index = 3;

    let expected_derivatives = Vec::from([
        0.0, 1.0, 1.0, 0.0, 0.0
    ]);

    for (i, _) in (&x).iter().enumerate() {
        let actual_differential = activation_layer.differential_of_output_with_respect_to_input(
            0, 
            i, 
            output_index
        );
        
        assert_eq!(actual_differential, expected_derivatives[i]);
    }
}

impl ActivationLayer for ReLU {
    fn function(inputs: &Vec<f64>) -> Vec<f64> {
        inputs.iter().map(|input| input.max(0.0)).collect::<Vec<f64>>()
    }

    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        _: usize,
    ) -> f64 {
        let activated_value = self.last_outputs[sample_index][input_index];

        if activated_value == 0.0_f64 {
            0.0
        } else {
            1.0
        }
    }

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f64>>) {
        self.last_inputs = input_samples.to_vec();
    }

    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f64>>) {
        self.last_outputs = output_samples.to_vec();
    }
}

impl Layer<f64> for ReLU {
    fn get_last_inputs(&self) -> Vec<Vec<f64>> {
        self.last_inputs.to_vec()
    }

    fn back_propagate(
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

    fn propagate(&mut self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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
