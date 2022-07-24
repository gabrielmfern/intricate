use intricate::layers::activations::tanh::TanHF64;
use intricate::layers::dense::DenseF64;
use intricate::layers::dense_gpu::DenseGpuF64;
use intricate::layers::layer::Layer;

use intricate::loss_functions::mean_squared::MeanSquared;
use intricate::model::{ModelF64, TrainingOptionsF64};

async fn run() {
    // Defining the training data
    let training_inputs = Vec::from([
        Vec::from([0.0, 0.0]),
        Vec::from([0.0, 1.0]),
        Vec::from([1.0, 0.0]),
        Vec::from([1.0, 1.0]),
    ]);
    let expected_outputs = Vec::from([
        Vec::from([0.0]),
        Vec::from([1.0]),
        Vec::from([1.0]),
        Vec::from([0.0]),
    ]);

    // Defining the layers for our XoR Model
    let mut layers: Vec<Box<dyn Layer<f64>>> = Vec::new();

    layers.push(Box::new(DenseF64::new(2, 3)));
    // The tanh activation function
    layers.push(Box::new(TanHF64::new()));
    layers.push(Box::new(DenseF64::new(3, 1)));
    layers.push(Box::new(TanHF64::new()));

    // Actually instantiate the Model with the layers
    let mut xor_model = ModelF64::new(layers);

    let epoch_amount = 100;

    // Fit the model however many times we want
    xor_model.fit(
        &training_inputs, 
        &expected_outputs, 
        TrainingOptionsF64 {
            learning_rate: 0.1,
            loss_algorithm: Box::new(MeanSquared), // The Mean Squared loss function
            should_print_information: true, // Should be verbose
            use_gpu: false // Should initialize WGPU Device and Queue for GPU layers
        },
        10000 // Epochs
    ).await;
}

fn main() {
    pollster::block_on(run());
}
