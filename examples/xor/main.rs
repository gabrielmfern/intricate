use intricate::layers::activations::TanH;
use intricate::layers::Dense;

use intricate::loss_functions::MeanSquared;
use intricate::Model;
use intricate::types::{ModelLayer, TrainingOptions};
use intricate::utils::setup_opencl;
use savefile::{load_file, save_file};

fn main() -> () {
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
    let layers: Vec<ModelLayer> = vec![
        Dense::new(2, 3),
        TanH::new (3),
        Dense::new(3, 1),
        TanH::new (1),
    ];

    // Actually instantiate the Model with the layers
    let mut xor_model = Model::new(layers);
    let opencl_state = setup_opencl().unwrap();
    xor_model.init(&opencl_state).unwrap();

    // Fit the model however many times we want
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            &mut TrainingOptions {
                learning_rate: 0.1,
                loss_algorithm: MeanSquared::new(), // The Mean Squared loss function
                should_print_information: true,        // Should be verbose
                epochs: 10000,
            },
        ).unwrap();

    // for saving Intricate uses the 'savefile' crate
    // that simply needs to call the 'save_file' function to the path you want
    // for the Model as follows
    save_file("xor-model.bin", 0, &xor_model).unwrap();

    // as for loading we can just call the 'load_file' function
    // on the path we saved to before
    let mut loaded_xor_model: Model = load_file("xor-model.bin", 0).unwrap();
    loaded_xor_model.init(&opencl_state).unwrap();

    loaded_xor_model
        .predict(&training_inputs).unwrap();
    xor_model.predict(&training_inputs).unwrap();

    let model_prediction = xor_model.get_last_prediction().unwrap();
    let loaded_model_prediction = loaded_xor_model.get_last_prediction().unwrap();

    assert_eq!(loaded_model_prediction, model_prediction);
}