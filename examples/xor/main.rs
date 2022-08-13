use intricate::layers::activations::tanh::TanH;
use intricate::layers::dense::Dense;

use intricate::loss_functions::mean_squared::MeanSquared;
use intricate::model::{Model, ModelLayer, TrainingOptions, ModelLossFunction};
use savefile::{load_file, save_file};

fn main() {
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
    let mut layers: Vec<ModelLayer> = Vec::new();

    layers.push(ModelLayer::Dense(Dense::new(2, 3)));
    // The tanh activation function
    layers.push(ModelLayer::TanH(TanH::new()));
    layers.push(ModelLayer::Dense(Dense::new(3, 1)));
    layers.push(ModelLayer::TanH(TanH::new()));

    // Actually instantiate the Model with the layers
    let mut xor_model = Model::new(layers);

    // Fit the model however many times we want
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            TrainingOptions {
                learning_rate: 0.1,
                loss_algorithm: ModelLossFunction::MeanSquared(MeanSquared), // The Mean Squared loss function
                should_print_information: true,        // Should be verbose
                epochs: 5000,
            },
        );

    // for saving Intricate uses the 'savefile' crate
    // that simply needs to call the 'save_file' function to the path you want
    // for the Model as follows
    save_file("xor-model.bin", 0, &xor_model).unwrap();

    // as for loading we can just call the 'load_file' function
    // on the path we saved to before
    let mut loaded_xor_model: Model = load_file("xor-model.bin", 0).unwrap();

    let loaded_model_prediction = loaded_xor_model
        .predict(&training_inputs);
    let model_prediction = xor_model.predict(&training_inputs);

    assert_eq!(loaded_model_prediction, model_prediction);

}