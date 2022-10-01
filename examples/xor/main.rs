use intricate::layers::activations::TanH;
use intricate::layers::Dense;

use intricate::loss_functions::MeanSquared;
use intricate::optimizers::BasicOptimizer;
use intricate::types::{ModelLayer, TrainingOptions, TrainingVerbosity, HaltingCondition};
use intricate::utils::opencl::DeviceType;
use intricate::utils::setup_opencl;
use intricate::Model;

use savefile::{load_file, save_file};

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
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();
    xor_model.init(&opencl_state).unwrap();

    let mut loss = MeanSquared::new();
    let mut optimizer = BasicOptimizer::new(0.1);

    // Fit the model however many times we want
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            &mut TrainingOptions {
                loss_fn: &mut loss, // the type of loss function that should be used for Intricate
                                    // to determine how bad the Model is
                verbosity: TrainingVerbosity {
                    show_current_epoch: true, // show a message for each epoch like `epoch #5`
                    show_epoch_progress: false, // show a progress bar of the training steps in a
                                                // epoch
                    show_epoch_elapsed: true, // show elapsed time in calculations for one epoch
                    print_accuracy: true, // should print the accuracy after each epoch
                    print_loss: true, // should print the loss after each epoch
                    halting_condition_warning: true,
                },
                //                 a condition for stopping the training if a min accuracy is reached
                halting_condition: Some(HaltingCondition::MinAccuracyReached(0.95)),
                compute_accuracy: false, // if Intricate should compute the accuracy after each
                                         // training step
                compute_loss: true, // if Intricate should compute the loss after each training
                                    // step
                optimizer: &mut optimizer,
                batch_size: 4, // the size of the mini-batch being used in Intricate's Mini-batch
                               // Gradient Descent
                epochs: 10000,
            },
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