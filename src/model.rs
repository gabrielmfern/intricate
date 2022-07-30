use std::time::Instant;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{layers::layer::Layer, loss_functions::loss_function::{LossFunctionF64, LossFunctionF32}, gpu};

// TODO: implement a macro for creating the f32 and f64 versions of everything

#[derive(Debug)]
pub struct TrainingOptionsF64 {
    pub loss_algorithm: Box<dyn LossFunctionF64>,
    // TODO: implement optimizers
    pub learning_rate: f64,
    pub should_print_information: bool,
    pub instantiate_gpu: bool,
    pub epochs: usize,
}

pub struct TrainingOptionsF32 {
    pub loss_algorithm: Box<dyn LossFunctionF32>,
    // TODO: implement optimizers
    pub learning_rate: f64,
    pub should_print_information: bool,
    pub instantiate_gpu: bool,
    pub epochs: usize,
}

#[allow(dead_code)]
pub struct ModelF64 {
    layers: Vec<Box<dyn Layer<f64>>>,
}

#[allow(dead_code)]
pub struct ModelF32 {
    layers: Vec<Box<dyn Layer<f32>>>,
}

#[allow(dead_code)]
impl ModelF64 {
    pub fn new(layers: Vec<Box<dyn Layer<f64>>>) -> ModelF64 {
        ModelF64 { layers }
    }

    pub async fn predict(
        &mut self,
        input_samples: &Vec<Vec<f64>>,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f64>> {
        let mut current_values = input_samples.to_vec();
        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(&current_values, device, queue).await;
        }
        current_values
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss
    pub async fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f64>>,
        training_expected_output_samples: &Vec<Vec<f64>>,
        training_options: TrainingOptionsF64,
    ) -> () {
        for epoch_index in 0..training_options.epochs {
            if training_options.should_print_information {
                println!("epoch #{}", epoch_index + 1);
            }

            let mut device = None;
            let mut queue = None;
            
            if training_options.instantiate_gpu {
                let (temp_device, temp_queue) = gpu::setup_device_and_queue().await;

                device = Some(temp_device);
                queue = Some(temp_queue);
            }

            self.back_propagate(
                training_input_samples, 
                training_expected_output_samples, 
                &training_options, 
                device, 
                queue
            ).await;
        }
    }

    /// This method is made to work with both
    /// GPU and CPU so it needs to receive the wgpu Device
    /// and the wgpu Queue to run the shaders on the GPU,
    /// but of curse it is an Option, so if you don't want to use
    /// the GPU just pass in None, DenseGPU will panic
    /// if there is no Device or Queue
    pub async fn back_propagate(
        &mut self,
        training_input_samples: &Vec<Vec<f64>>,
        training_expected_output_samples: &Vec<Vec<f64>>,
        training_options: &TrainingOptionsF64,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> f64 {
        assert_eq!(
            training_input_samples.len(),
            training_expected_output_samples.len()
        );

        assert!(training_input_samples.len() > 0);

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict(training_input_samples, &device, &queue).await;

        let outputs_amount = training_expected_output_samples[0].len();

        // Not sure if this can be implemented on the GPU because of the
        // computation of the loss bellow being done on dyn LossFunction
        let mut lost_to_outputs_derivatives = training_expected_output_samples
            .par_iter()
            .zip(training_actual_outputs)
            .map(|(expected_outputs, actual_outputs)| {
                expected_outputs
                    .iter()
                    .zip(actual_outputs)
                    .map(|(expected_output, actual_output)| {
                        (&training_options)
                            .loss_algorithm
                            .compute_loss_derivative_with_respect_to_output(
                                outputs_amount,
                                actual_output,
                                *expected_output,
                            )
                    })
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        for (layer_index, layer) in self.layers.iter_mut().enumerate().rev() {
            if layer_index > 0 {
                // always Some
                lost_to_outputs_derivatives = layer
                    .back_propagate(
                        true,
                        &lost_to_outputs_derivatives,
                        training_options.learning_rate,
                        &device,
                        &queue,
                    )
                    .await
                    .unwrap();
            } else {
                layer
                    .back_propagate(
                        // always None
                        false,
                        &lost_to_outputs_derivatives,
                        training_options.learning_rate,
                        &device,
                        &queue,
                    )
                    .await;
            }
        }

        let actual_sample_outputs = &self.predict(training_input_samples, &device, &queue).await;

        let new_loss = training_options
            .loss_algorithm
            .average_loss_for_samples(actual_sample_outputs, training_expected_output_samples);

        // let new_loss = self.compute_loss(
        //     training_input_samples,
        //     training_expected_output_samples,
        //     training_options.loss_algorithm
        // ).await;

        if training_options.should_print_information {
            println!(
                "{}s elapsed, now has loss of {}",
                start_instant.elapsed().as_secs_f32(),
                new_loss
            );
        }

        new_loss
    }
}

impl ModelF32 {
    pub fn new(layers: Vec<Box<dyn Layer<f32>>>) -> ModelF32 {
        ModelF32 { layers }
    }

    pub async fn predict(
        &mut self,
        input_samples: &Vec<Vec<f32>>,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> Vec<Vec<f32>> {
        let mut current_values = input_samples.to_vec();
        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(&current_values, device, queue).await;
        }
        current_values
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss
    pub async fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: TrainingOptionsF32,
    ) -> () {
        let mut device = None;
        let mut queue = None;
        
        if training_options.instantiate_gpu {
            let (temp_device, temp_queue) = gpu::setup_device_and_queue().await;

            device = Some(temp_device);
            queue = Some(temp_queue);
        }

        for epoch_index in 0..training_options.epochs {
            if training_options.should_print_information {
                println!("epoch #{}", epoch_index + 1);
            }

            self.back_propagate(
                training_input_samples, 
                training_expected_output_samples, 
                &training_options, 
                &device, 
                &queue
            ).await;
        }
    }

    /// This method is made to work with both
    /// GPU and CPU so it needs to receive the wgpu Device
    /// and the wgpu Queue to run the shaders on the GPU,
    /// but of curse it is an Option, so if you don't want to use
    /// the GPU just pass in None, DenseGPU will panic
    /// if there is no Device or Queue
    pub async fn back_propagate(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: &TrainingOptionsF32,
        device: &Option<wgpu::Device>,
        queue: &Option<wgpu::Queue>,
    ) -> f32 {
        assert_eq!(
            training_input_samples.len(),
            training_expected_output_samples.len()
        );

        assert!(training_input_samples.len() > 0);

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict(training_input_samples, &device, &queue).await;

        let outputs_amount = training_expected_output_samples[0].len();

        // Not sure if this can be implemented on the GPU because of the
        // computation of the loss bellow being done on dyn LossFunction
        let mut lost_to_outputs_derivatives = training_expected_output_samples
            .par_iter()
            .zip(training_actual_outputs)
            .map(|(expected_outputs, actual_outputs)| {
                expected_outputs
                    .iter()
                    .zip(actual_outputs)
                    .map(|(expected_output, actual_output)| {
                        (&training_options)
                            .loss_algorithm
                            .compute_loss_derivative_with_respect_to_output(
                                outputs_amount,
                                actual_output,
                                *expected_output,
                            )
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        for (layer_index, layer) in self.layers.iter_mut().enumerate().rev() {
            if layer_index > 0 {
                // always Some
                lost_to_outputs_derivatives = layer
                    .back_propagate(
                        true,
                        &lost_to_outputs_derivatives,
                        training_options.learning_rate,
                        &device,
                        &queue,
                    )
                    .await
                    .unwrap();
            } else {
                layer
                    .back_propagate(
                        // always None
                        false,
                        &lost_to_outputs_derivatives,
                        training_options.learning_rate,
                        &device,
                        &queue,
                    )
                    .await;
            }
        }

        let actual_sample_outputs = &self.predict(training_input_samples, &device, &queue).await;

        let new_loss = training_options
            .loss_algorithm
            .average_loss_for_samples(actual_sample_outputs, training_expected_output_samples);

        if training_options.should_print_information {
            println!(
                "{}s elapsed, now has loss of {}",
                start_instant.elapsed().as_secs_f32(),
                new_loss
            );
        }

        new_loss
    }
}