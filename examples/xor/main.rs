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
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            TrainingOptionsF32 {
                learning_rate: 0.1,
                loss_algorithm: Box::new(MeanSquared), // The Mean Squared loss function
                should_print_information: true,        // Should be verbose
                instantiate_gpu: false, // Should not initialize WGPU Device and Queue for GPU layers since there are no GPU layers here
                epochs: 10000,
            },
        )
        .await;
    // we await here because for a GPU computation type of layer
    // the responses from the GPU must be awaited on the CPU
    // and since the model technically does not know what type of layers there are
    // it cannot automatically initialize or not wgpu Deivce and Queue
    // the dense gpu layers will panic if use_gpu is false

    // for saving Intricate uses the 'savefile' crate
    // that simply needs to call the 'save_file' function to the path you want
    // for the layers in the model and then load the layers and instiate the model again
    // the reason we do this is because the model can't really be easily Sized by the compiler
    // because the model can have any type of layer
    // just call the function bellow
    xor_model.layers[0]
        .save("xor-model-first-dense.bin", 0)
        .unwrap();
    xor_model.layers[2]
        .save("xor-model-second-dense.bin", 0)
        .unwrap();

    // as for loading we can just call the 'load_file' function
    // on each of the layers like this:
    let mut first_dense: Box<DenseF32> = Box::new(DenseF32::dummy());
    first_dense.load("xor-model-first-dense.bin", 0).unwrap();
    let mut second_dense: Box<DenseF32> = Box::new(DenseF32::dummy());
    second_dense.load("xor-model-second-dense.bin", 0).unwrap();

    let mut new_layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
    new_layers.push(first_dense);
    new_layers.push(Box::new(TanHF32::new()));
    new_layers.push(second_dense);
    new_layers.push(Box::new(TanHF32::new()));

    let mut loaded_xor_model = ModelF32::new(new_layers);

    let loaded_model_prediction = loaded_xor_model.predict(&training_inputs, &None, &None).await;
    let model_prediction = xor_model.predict(&training_inputs, &None, &None).await;

    assert_eq!(loaded_model_prediction, model_prediction);
}

fn main() {
    // just wait for the everything to run before stopping
    pollster::block_on(run());
}
