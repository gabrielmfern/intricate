use intricate::layers::activations::tanh::TanHF32;
use intricate::layers::dense::DenseF32;
use intricate::layers::layer::Layer;

use intricate::loss_functions::mean_squared::MeanSquared;
use intricate::model::{ModelF32, TrainingOptionsF32};

async fn run() {
    // Defining the training data
    let training_inputs: Vec<Vec<f32>> = Vec::from([
        Vec::from([0.0, 0.0]),
        Vec::from([0.0, 1.0]),
        Vec::from([1.0, 0.0]),
        Vec::from([1.0, 1.0]),
    ]);
    let expected_outputs: Vec<Vec<f32>> = Vec::from([
        Vec::from([0.0]),
        Vec::from([1.0]),
        Vec::from([1.0]),
        Vec::from([0.0]),
    ]);

    // Defining the layers for our XoR Model
    let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();

    layers.push(Box::new(DenseF32::new(2, 3)));
    // The tanh activation function
    layers.push(Box::new(TanHF32::new()));
    layers.push(Box::new(DenseF32::new(3, 1)));
    layers.push(Box::new(TanHF32::new()));

    // Actually instantiate the Model with the layers
    let mut xor_model = ModelF32::new(layers);

    // Fit the model however many times we want
    xor_model.fit(
        &training_inputs, 
        &expected_outputs, 
        TrainingOptionsF32 {
            learning_rate: 0.1,
            loss_algorithm: Box::new(MeanSquared), // The Mean Squared loss function
            should_print_information: true, // Should be verbose
            instantiate_gpu: false, // Should not initialize WGPU Device and Queue for GPU layers since there are no GPU layers here
            epochs: 10000,
        },
    ).await;
    // we await here because for a GPU computation type of layer
    // the responses from the GPU must be awaited on the CPU
    // and since the model technically does not know what type of layers there are
    // it cannot automatically initialize or not wgpu Deivce and Queue
    // the dense gpu layers will panic if use_gpu is false
}

fn main() {
    pollster::block_on(run());
}