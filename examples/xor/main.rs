use intricate::layers::activations::TanH;
use intricate::layers::Dense;

use intricate::loss_functions::MeanSquared;
use intricate::optimizers;
use intricate::types::{ModelLayer, TrainingOptions, HaltingCondition};
use intricate::utils::opencl::DeviceType;
use intricate::utils::setup_opencl;
use intricate::Model;

use intricate::utils::savefile::save_file;
use intricate::utils::savefile::load_file;

fn main() -> () {
    // Defining the training data
    let training_inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let expected_outputs: Vec<Vec<f32>> = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Defining the layers for our XoR Model
    let layers: Vec<ModelLayer> = vec![
        Dense::new(2, 3),
        TanH::new(3),
        Dense::new(3, 1),
        TanH::new(1),
    ];

    // Actually instantiate the Model with the layers
    let mut xor_model = Model::new(layers);
    //            you can change this to DeviceType::GPU if you want
    let opencl_state = setup_opencl(DeviceType::CPU).unwrap();
    xor_model.init(&opencl_state).unwrap();

    let mut loss = MeanSquared::new();
    let mut optimizer = optimizers::Basic::new(0.1);

    // Fit the model however many times we want
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            &mut TrainingOptions::new(&mut loss, &mut optimizer)
                .set_epochs(10000)
                .set_batch_size(4)
                .should_compute_accuracy(true).unwrap()
                .should_print_accuracy(true).unwrap()
                .set_halting_condition(HaltingCondition::MinAccuracyReached(0.95)).unwrap()
                .should_show_halting_condition_warning(true).unwrap(),
        )
        .unwrap();

    // for saving Intricate uses the 'savefile' crate
    // that simply needs to call the 'save_file' function to the path you want
    // for the Model as follows
    xor_model.sync_data_from_buffers_to_host().unwrap();
    save_file("xor-model.bin", 0, &xor_model).unwrap();

    // as for loading we can just call the 'load_file' function
    // on the path we saved to before
    let mut loaded_xor_model: Model = load_file("xor-model.bin", 0).unwrap();
    loaded_xor_model.init(&opencl_state).unwrap();

    loaded_xor_model.predict(&training_inputs).unwrap();
    xor_model.predict(&training_inputs).unwrap();

    let model_prediction = xor_model.get_last_prediction().unwrap();
    let loaded_model_prediction = loaded_xor_model.get_last_prediction().unwrap();

    assert_eq!(loaded_model_prediction, model_prediction);
}