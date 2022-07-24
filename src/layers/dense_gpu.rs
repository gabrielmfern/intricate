use async_trait::async_trait;
use rand::Rng;

use crate::gpu::apply_gradients_to_dense_weights::apply_gradients_to_f64_dense_weights;
use crate::gpu::calculate_dense_input_to_error_derivatives::calculate_dense_input_to_error_derivatives;
use crate::utils::matrix_operations::MatrixOperations;
use crate::{layers::layer::Layer, utils::vector_operations::VectorOperations};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub struct DenseGpuF64 {
    pub inputs_amount: usize,
    pub outputs_amount: usize,

    // TODO: flatten everything on this struct
    // so that there is no need to flatten and unflatten these when back_propagating
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,

    pub last_inputs: Vec<Vec<f64>>,
    pub last_outputs: Vec<Vec<f64>>,
}

impl DenseGpuF64 {
    #[allow(dead_code)]

    pub fn new(inputs_amount: usize, outputs_amount: usize) -> DenseGpuF64 {
        let mut rng = rand::thread_rng();
        DenseGpuF64 {
            inputs_amount,
            outputs_amount,
            weights: (0..inputs_amount)
                .into_iter()
                .map(|_| {
                    (0..outputs_amount)
                        .into_iter()
                        .map(|_| rng.gen_range(0.0_f64..=1.0_f64))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>(),
            biases: (0..outputs_amount)
                .into_iter()
                .map(|_| rng.gen_range(0.0_f64..=1.0_f64))
                .collect::<Vec<f64>>(),
            last_outputs: Vec::new(),
            last_inputs: Vec::new(),
        }
    }
}

#[async_trait]
impl Layer<f64> for DenseGpuF64 {
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

    async fn propagate(
        &mut self,
        inputs_samples: &Vec<Vec<f64>>,
        _: &Option<wgpu::Device>,
        _: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f64>> {
        self.last_inputs = inputs_samples.to_vec();
        self.last_outputs = inputs_samples
            .par_iter()
            .map(|inputs| self.biases.add(&self.weights.dot_product(inputs)))
            .collect::<Vec<Vec<f64>>>();
        self.last_outputs.to_vec()
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f64>>,
        learning_rate: f64,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<f64>>> {
        if device.is_none() || queue.is_none() {
            panic!("Cannot use DenseGPUF64 without setting up for the GPU!");
        }

        assert!(!self.last_inputs.is_empty());
        let samples_amount = layer_output_to_error_derivative.len();

        apply_gradients_to_f64_dense_weights(
            self,
            device.as_ref().unwrap(),
            queue.as_ref().unwrap(),
            layer_output_to_error_derivative,
            learning_rate,
        )
        .await;

        self.biases = (0..self.outputs_amount)
            .into_par_iter()
            .map(|j| {
                self.biases[j]
                    + learning_rate
                        * layer_output_to_error_derivative
                            .iter()
                            .map(|sample_output_derivatives| sample_output_derivatives[j])
                            .sum::<f64>()
                        / samples_amount as f64
            })
            .collect::<Vec<f64>>();

        if should_calculate_input_to_error_derivative {
            let layer_input_to_error_derivatives = calculate_dense_input_to_error_derivatives(
                self, 
                device.as_ref().unwrap(), 
                queue.as_ref().unwrap(), 
                layer_output_to_error_derivative
            ).await.expect("Some error happenned computing the input to error derivatives on a DenseGPUF64 layer");

            // layer_output_to_error_derivative
            //     .par_iter()
            //     .map(|sample_output_derivatives| {
            //         self.weights
            //             .iter()
            //             .map(|input_to_outputs| {
            //                 input_to_outputs
            //                     .iter()
            //                     .enumerate()
            //                     .map(|(j, weight)| weight * sample_output_derivatives[j])
            //                     .sum::<f64>()
            //             })
            //             .collect::<Vec<f64>>()
            //     })
            //     .collect::<Vec<Vec<f64>>>();

            Some(layer_input_to_error_derivatives)
        } else {
            None
        }
    }
}
