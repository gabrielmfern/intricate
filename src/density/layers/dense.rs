use crate::density::layers::layer::Layer;
use crate::density::utils::matrix_operations::MatrixOperations;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

pub struct Dense {
    inputs_amount: usize,
    outputs_amount: usize,

    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,

    last_inputs: Vec<Vec<f64>>,
    last_outputs: Vec<Vec<f64>>,
}

impl Layer<f64> for Dense {
    fn get_last_inputs(&self) -> Vec<Vec<f64>> {
        self.last_inputs.to_vec()
    }
    fn get_last_outputs(&self) -> Vec<Vec<f64>> {
        self.last_outputs.to_vec()
    }

    fn get_inputs_amount(&self) -> usize {
        self.inputs_amount
    }
    fn get_outputs_amount(&self) -> usize {
        self.outputs_amount
    }

    fn propagate(&mut self, inputs_samples: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.last_inputs = inputs_samples.to_vec();
        self.last_outputs = vec![self.biases.to_vec(); inputs_samples.len()];
        for (sample_index, inputs) in inputs_samples.iter().enumerate() {
            self.last_outputs[sample_index] = self.weights.dot_product_with_vector(inputs);
        }
        self.last_outputs.to_vec()
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) -> Option<Vec<Vec<f64>>> {
        assert!(!self.last_inputs.is_empty());
        let samples_amount = layer_output_to_error_derivative.len();

        // apply the gradients averaging the calculations between the samples
        // but becomes extremely hard to calculate on very large neural networks
        // with a large amount of samples to train on
        //
        // TODO: implement this on compute shaders using WGPU or any equivalent
        self.weights = (0..self.inputs_amount)
            .into_par_iter()
            .map(|l| {
                (0..self.outputs_amount)
                    .into_iter()
                    .map(|j| {
                        -learning_rate
                            * layer_output_to_error_derivative
                                .iter()
                                .enumerate()
                                .map(|(sample_index, sample_output_derivatives)| {
                                    sample_output_derivatives[j] * self.last_inputs[sample_index][l]
                                })
                                .sum::<f64>()
                            / samples_amount as f64
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        if should_calculate_input_to_error_derivative {
            let layer_input_to_error_derivatives = layer_output_to_error_derivative
                .par_iter()
                .map(|sample_output_derivatives| {
                    self.weights
                        .iter()
                        .map(|input_to_outputs| {
                            input_to_outputs
                                .iter()
                                .enumerate()
                                .map(|(j, weight)| weight * sample_output_derivatives[j])
                                .sum::<f64>()
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            Some(layer_input_to_error_derivatives)
        } else {
            None
        }
    }
}
