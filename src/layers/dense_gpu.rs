use async_trait::async_trait;
use rand::Rng;
use savefile::{save_file, SavefileError, load_file};
use savefile_derive::Savefile;

use crate::gpu::apply_gradients_to_dense_weights::apply_gradients_to_f32_dense_weights;
use crate::gpu::calculate_dense_input_to_error_derivatives::calculate_dense_input_to_error_derivatives;
use crate::gpu::propagate_through_weights_and_biases::propagate_through_weights_and_biases;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::layer::Layer;

/// We use f32's here because I'm not quite sure 
/// how to use f64's in a shader with wgpu since it
/// always yields runtime errors while trying to use 
/// f64 or double or any other types that I tried
#[derive(Debug, Clone, Savefile)]
pub struct DenseGpuF32 {
    pub inputs_amount: usize,
    pub outputs_amount: usize,

    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,

    pub last_inputs: Vec<Vec<f32>>,
    pub last_outputs: Vec<Vec<f32>>,
}

impl DenseGpuF32 {
    /// can be used for just instantiating a Layer to then load 
    /// the weights and biases from some saved layer file
    pub fn dummy() -> DenseGpuF32 {
        DenseGpuF32 {
            inputs_amount: 0,
            outputs_amount: 0,
            weights: Vec::new(),
            biases: Vec::new(),
            last_outputs: Vec::new(),
            last_inputs: Vec::new()
        }
    }

    #[allow(dead_code)]

    pub fn new(inputs_amount: usize, outputs_amount: usize) -> DenseGpuF32 {
        let mut rng = rand::thread_rng();
        DenseGpuF32 {
            inputs_amount,
            outputs_amount,
            weights: (0..inputs_amount)
            .into_iter()
            .map(|_| {
                (0..outputs_amount)
                    .into_iter()
                    .map(|_| rng.gen_range(0.0_f32..=1.0_f32))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
            biases: (0..outputs_amount)
                .into_iter()
                .map(|_| rng.gen_range(0.0_f32..=1.0_f32))
                .collect::<Vec<f32>>(),
            last_outputs: Vec::new(),
            last_inputs: Vec::new(),
        }
    }
}

#[async_trait]
impl Layer<f32> for DenseGpuF32 {
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

    /// saves all the information of the current layer
    /// expect for the last_outputs and last_inputs since these don't
    /// really matter
    fn save(&self, path: &str, version: u32) -> Result<(), SavefileError> {
        let mut layer_to_save = self.clone();
        layer_to_save.last_outputs = Vec::new();
        layer_to_save.last_inputs = Vec::new();
        save_file(path, version, &layer_to_save)
    }

    /// loads all of the weights, biases, inputs_amount and ouputs_amount
    /// into the current layer from the file in the path with that version
    fn load(&mut self, path: &str, version: u32) -> Result<(), SavefileError> {
        let loaded_layer_result: Result<Self, SavefileError> = load_file(path, version);
        if loaded_layer_result.is_err() {
            Err(loaded_layer_result.err().unwrap())
        } else {
            let loaded_layer = loaded_layer_result.unwrap();

            self.weights = loaded_layer.weights;
            self.biases = loaded_layer.biases;

            self.outputs_amount = loaded_layer.outputs_amount;
            self.inputs_amount = loaded_layer.inputs_amount;

            Ok(())
        }
    }

    async fn propagate(
        &mut self,
        inputs_samples: &Vec<Vec<f32>>,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f32>> {
        if device.is_none() || queue.is_none() {
            panic!("Cannot use DenseGPUF32 without setting up for the GPU!");
        }

        self.last_inputs = inputs_samples.to_vec();

        self.last_outputs = propagate_through_weights_and_biases(
            self, 
            inputs_samples, 
            device.as_ref().unwrap(), 
            queue.as_ref().unwrap(),
        ).await.unwrap();
        
        self.last_outputs.to_vec()
    }

    async fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f64,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Option<Vec<Vec<f32>>> {
        if device.is_none() || queue.is_none() {
            panic!("Cannot use DenseGPUF32 without setting up for the GPU!");
        }

        assert!(!self.last_inputs.is_empty());
        let samples_amount = layer_output_to_error_derivative.len();

        apply_gradients_to_f32_dense_weights(
            self,
            device.as_ref().unwrap(),
            queue.as_ref().unwrap(),
            layer_output_to_error_derivative,
            learning_rate as f32,
        )
        .await;

        // very small calculation so I kept it here
        // since it doesn't impact performance that much being just
        // O(outputs_amount * samples_amount) or O(n^2)
        self.biases = (0..self.outputs_amount)
            .into_par_iter()
            .map(|j| {
                self.biases[j]
                    + learning_rate as f32
                        * layer_output_to_error_derivative
                            .iter()
                            .map(|sample_output_derivatives| sample_output_derivatives[j])
                            .sum::<f32>()
                        / samples_amount as f32
            })
            .collect::<Vec<f32>>();

        if should_calculate_input_to_error_derivative {
            let layer_input_to_error_derivatives = calculate_dense_input_to_error_derivatives(
                self, 
                device.as_ref().unwrap(), 
                queue.as_ref().unwrap(), 
                layer_output_to_error_derivative
            ).await.expect("Some error happenned computing the input to error derivatives on a DenseGPUF32 layer");

            Some(layer_input_to_error_derivatives)
        } else {
            None
        }
    }
}