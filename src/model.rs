use std::time::Instant;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{layers::layer::Layer, loss_functions::loss_function::LossFunctionF64};

pub struct TrainingOptionsF64 {
    pub loss_algorithm: Box<dyn LossFunctionF64>,
    // TODO: implement optimizers
    pub learning_rate: f64,
    pub should_print_information: bool
}

#[allow(dead_code)]
pub struct ModelF64 {
    layers: Vec<Box<dyn Layer<f64>>>,
}

#[allow(dead_code)]
impl ModelF64 {
    pub fn new(layers: Vec<Box<dyn Layer<f64>>>) -> ModelF64 {
        ModelF64 { layers }
    }

    pub fn predict(&mut self, input_samples: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut current_values = input_samples.to_vec();
        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(&current_values);
        }
        current_values
    }

    pub fn compute_loss(
        &mut self, 
        training_input_samples: &Vec<Vec<f64>>, 
        training_expected_output_samples: &Vec<Vec<f64>>,
        loss_algorithm: Box<dyn LossFunctionF64>
    ) -> f64 {
        let actual_sample_outputs = self.predict(training_input_samples);

        loss_algorithm.average_loss_for_samples(
            &actual_sample_outputs, training_expected_output_samples
        )
    }

    pub fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f64>>,
        training_expected_output_samples: &Vec<Vec<f64>>,
        training_options: TrainingOptionsF64,
    ) -> f64 {
        assert_eq!(
            training_input_samples.len(),
            training_expected_output_samples.len()
        );

        assert!(training_input_samples.len() > 0);

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict(training_input_samples);

        let outputs_amount = training_expected_output_samples[0].len();
        
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
                lost_to_outputs_derivatives = layer.back_propagate(
                    true,
                    &lost_to_outputs_derivatives, 
                    training_options.learning_rate
                ).unwrap();
            } else {
                layer.back_propagate( // always None
                    false,
                    &lost_to_outputs_derivatives, 
                    training_options.learning_rate
                );
            }
        }

        let new_loss = self.compute_loss(
            training_input_samples, 
            training_expected_output_samples, 
            training_options.loss_algorithm
        );

        if training_options.should_print_information {
            println!("{}s elapsed, now has loss of {}", start_instant.elapsed().as_secs_f32(), new_loss);
        }

        new_loss
    }
}
