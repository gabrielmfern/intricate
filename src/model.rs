use std::time::Instant;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use savefile_derive::Savefile;

use crate::{
    layers::{
        activations::{relu::ReLU, sigmoid::Sigmoid, softmax::SoftMax, tanh::TanH},
        dense::Dense,
        Layer,
    },
    loss_functions::{
        categorical_cross_entropy::CategoricalCrossEntropy, mean_squared::MeanSquared, LossFunction,
    },
};

#[derive(Debug, Clone, Savefile)]
pub enum ModelLayer {
    Dense(Dense),
    TanH(TanH),
    Sigmoid(Sigmoid),
    SoftMax(SoftMax),
    ReLU(ReLU),
}

#[derive(Debug)]
pub enum ModelLossFunction {
    CategoricalCrossEntropy(CategoricalCrossEntropy),
    MeanSquared(MeanSquared),
}

impl LossFunction for ModelLossFunction {
    fn compute_loss(&self, outputs: &Vec<f32>, expected_outputs: &Vec<f32>) -> f32 {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => {
                lossfn.compute_loss(outputs, expected_outputs)
            }
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => {
                lossfn.compute_loss(outputs, expected_outputs)
            }
        }
    }

    fn average_loss_for_samples(
        &self,
        sample_outputs: &Vec<Vec<f32>>,
        sample_expected_outputs: &Vec<Vec<f32>>,
    ) -> f32 {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => {
                lossfn.average_loss_for_samples(sample_outputs, sample_expected_outputs)
            }
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => {
                lossfn.average_loss_for_samples(sample_outputs, sample_expected_outputs)
            }
        }
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f32,
        expected_output: f32,
    ) -> f32 {
        match self {
            ModelLossFunction::MeanSquared(lossfn) => lossfn
                .compute_loss_derivative_with_respect_to_output(
                    ouputs_amount,
                    output,
                    expected_output,
                ),
            ModelLossFunction::CategoricalCrossEntropy(lossfn) => lossfn
                .compute_loss_derivative_with_respect_to_output(
                    ouputs_amount,
                    output,
                    expected_output,
                ),
        }
    }
}

impl Layer for ModelLayer {
    fn get_last_inputs(&self) -> &Vec<Vec<f32>> {
        match self {
            ModelLayer::Dense(layer) => layer.get_last_inputs(),
            ModelLayer::TanH(layer) => layer.get_last_inputs(),
            ModelLayer::Sigmoid(layer) => layer.get_last_inputs(),
            ModelLayer::SoftMax(layer) => layer.get_last_inputs(),
            ModelLayer::ReLU(layer) => layer.get_last_inputs(),
        }
    }

    fn get_last_outputs(&self) -> &Vec<Vec<f32>> {
        match self {
            ModelLayer::Dense(layer) => layer.get_last_outputs(),
            ModelLayer::TanH(layer) => layer.get_last_outputs(),
            ModelLayer::Sigmoid(layer) => layer.get_last_outputs(),
            ModelLayer::SoftMax(layer) => layer.get_last_outputs(),
            ModelLayer::ReLU(layer) => layer.get_last_outputs(),
        }
    }

    fn get_inputs_amount(&self) -> usize {
        match self {
            ModelLayer::Dense(layer) => layer.get_inputs_amount(),
            ModelLayer::TanH(layer) => layer.get_inputs_amount(),
            ModelLayer::Sigmoid(layer) => layer.get_inputs_amount(),
            ModelLayer::SoftMax(layer) => layer.get_inputs_amount(),
            ModelLayer::ReLU(layer) => layer.get_inputs_amount(),
        }
    }

    fn get_outputs_amount(&self) -> usize {
        match self {
            ModelLayer::Dense(layer) => layer.get_outputs_amount(),
            ModelLayer::TanH(layer) => layer.get_outputs_amount(),
            ModelLayer::Sigmoid(layer) => layer.get_outputs_amount(),
            ModelLayer::SoftMax(layer) => layer.get_outputs_amount(),
            ModelLayer::ReLU(layer) => layer.get_outputs_amount(),
        }
    }

    fn propagate(
        &mut self,
        inputs: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        match self {
            ModelLayer::Dense(layer) => layer.propagate(inputs),
            ModelLayer::TanH(layer) => layer.propagate(inputs),
            ModelLayer::Sigmoid(layer) => layer.propagate(inputs),
            ModelLayer::SoftMax(layer) => layer.propagate(inputs),
            ModelLayer::ReLU(layer) => layer.propagate(inputs),
        }
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) -> Option<Vec<Vec<f32>>> {
        match self {
            ModelLayer::Dense(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            ModelLayer::TanH(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            ModelLayer::Sigmoid(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            ModelLayer::SoftMax(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
            ModelLayer::ReLU(layer) => layer.back_propagate(
                should_calculate_input_to_error_derivative,
                layer_output_to_error_derivative,
                learning_rate,
            ),
        }
    }
}

pub struct TrainingOptions {
    pub loss_algorithm: ModelLossFunction,
    // TODO: implement optimizers
    pub learning_rate: f32,
    pub should_print_information: bool,
    pub instantiate_gpu: bool,
    pub epochs: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Savefile)]
/// An Intricate Model which can be defined as just an ordering
/// of some layers with their inputs and outputs, the Model receives
/// the inputs for the first layer and results in the outputs of the last layer,
/// it also back_propagates returning the new loss for the Model based on the
/// defined Loss Function and calls the back_propagate method on each layer
/// going from the last to the first layer
pub struct Model {
    pub layers: Vec<ModelLayer>,
}

impl Model {
    pub fn new(layers: Vec<ModelLayer>) -> Model {
        Model { layers }
    }

    pub fn predict(
        &mut self,
        input_samples: &Vec<Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let mut current_values = input_samples.to_vec();
        for layer in self.layers.iter_mut() {
            current_values = layer.propagate(&current_values);
        }
        current_values
    }

    /// fits the Model to best suit the training data
    /// using the back_propagate method of every layer
    /// and prints the loss
    pub fn fit(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: TrainingOptions,
    ) -> () {
        for epoch_index in 0..training_options.epochs {
            if training_options.should_print_information {
                println!("epoch #{}", epoch_index + 1);
            }

            self.back_propagate(
                training_input_samples,
                training_expected_output_samples,
                &training_options,
            );
        }
    }

    /// This method is made to work with both
    /// GPU and CPU so it needs to receive the wgpu Device
    /// and the wgpu Queue to run the shaders on the GPU,
    /// but of curse it is an Option, so if you don't want to use
    /// the GPU just pass in None, DenseGPU will panic
    /// if there is no Device or Queue
    pub fn back_propagate(
        &mut self,
        training_input_samples: &Vec<Vec<f32>>,
        training_expected_output_samples: &Vec<Vec<f32>>,
        training_options: &TrainingOptions,
    ) -> f32 {
        assert_eq!(
            training_input_samples.len(),
            training_expected_output_samples.len()
        );

        assert!(training_input_samples.len() > 0);

        let start_instant = Instant::now();

        let training_actual_outputs = self.predict(training_input_samples);

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
                    )
                    .unwrap();
            } else {
                layer.back_propagate(
                    // always None
                    false,
                    &lost_to_outputs_derivatives,
                    training_options.learning_rate,
                );
            }
        }

        let actual_sample_outputs = &self.predict(training_input_samples);

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