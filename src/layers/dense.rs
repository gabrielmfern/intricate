use async_trait::async_trait;
use rand::Rng;
use savefile_derive::Savefile;

use crate::utils::matrix_operations::MatrixOperations;
use crate::{layers::Layer, utils::vector_operations::VectorOperations};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(Debug, Clone, Savefile)]
/// A densely connected layer, this layer consists of some inputs
/// and the weights that connect each input to all outputs,
/// its propagation results in a dot product between these weights
/// and the inputs received in the propagation method
pub struct Dense {
    pub inputs_amount: usize,
    pub outputs_amount: usize,

    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,

    pub last_inputs: Vec<Vec<f32>>,
    pub last_outputs: Vec<Vec<f32>>,
}

impl Dense {
    #[allow(dead_code)]

    pub fn new(inputs_amount: usize, outputs_amount: usize) -> Dense {
        let mut rng = rand::thread_rng();
        Dense {
            inputs_amount,
            outputs_amount,
            weights: (0..inputs_amount)
                .into_iter()
                .map(|_| {
                    (0..outputs_amount)
                        .into_iter()
                        .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>(),
            biases: (0..outputs_amount)
                .into_iter()
                .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
                .collect::<Vec<f32>>(),
            last_outputs: Vec::new(),
            last_inputs: Vec::new(),
        }
    }

    /// Creates an empty Dense layer without any weights,
    /// inputs_amount or outputs_amount
    pub fn dummy() -> Dense {
        Dense {
            inputs_amount: 0,
            outputs_amount: 0,
            weights: Vec::new(),
            biases: Vec::new(),
            last_outputs: Vec::new(),
            last_inputs: Vec::new()
        }
    }
}

#[async_trait]
impl Layer for Dense {
    fn get_last_inputs(&self) -> &Vec<Vec<f32>> {
        &self.last_inputs
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f32>> {
        &self.last_outputs
    }

    fn get_inputs_amount(&self) -> usize {
        self.inputs_amount
    }

    fn get_outputs_amount(&self) -> usize {
        self.outputs_amount
    }

    async fn propagate(
        &mut self, 
        inputs_samples: &Vec<Vec<f32>>, 
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f32>> {
        self.last_inputs = inputs_samples.to_vec();
        self.last_outputs = inputs_samples
            .par_iter()
            .map(|inputs| self.biases.add(&self.weights.dot_product(inputs)))
            .collect::<Vec<Vec<f32>>>();
        self.last_outputs.to_vec()
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32, 
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<f32>>> {
        assert!(!self.last_inputs.is_empty());
        let samples_amount = layer_output_to_error_derivative.len();
        let float_samples_amount = samples_amount as f32;

        // apply the gradients averaging the calculations between the samples
        // but becomes extremely hard to calculate on very large neural networks
        // with a large amount of samples to train on
        self.weights = (0..self.inputs_amount)
            .into_par_iter()
            .map(|l| {
                (0..self.outputs_amount)
                    .into_iter()
                    .map(|j| {
                        self.weights[l][j]
                            + learning_rate
                                * layer_output_to_error_derivative
                                    .iter()
                                    .enumerate()
                                    .map(|(sample_index, sample_output_derivatives)| {
                                        sample_output_derivatives[j]
                                            * self.last_inputs[sample_index][l]
                                    })
                                    .sum::<f32>()
                                / float_samples_amount
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        self.biases = (0..self.outputs_amount)
            .into_par_iter()
            .map(|j| {
                self.biases[j]
                    + learning_rate
                        * layer_output_to_error_derivative
                            .iter()
                            .map(|sample_output_derivatives| sample_output_derivatives[j])
                            .sum::<f32>()
                        / float_samples_amount
            })
            .collect::<Vec<f32>>();

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
                                .sum::<f32>()
                        })
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();

            Some(layer_input_to_error_derivatives)
        } else {
            None
        }
    }
}