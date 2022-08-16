use intricate::layers::activations::TanH;
use intricate::layers::Dense;

use intricate::loss_functions::MeanSquared;
use intricate::types::{ModelLayer, TrainingOptions};
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
    let opencl_state = setup_opencl(DeviceType::CPU).unwrap();
    xor_model.init(&opencl_state).unwrap();

    // Fit the model however many times we want
    xor_model
        .fit(
            &training_inputs,
            &expected_outputs,
            &mut TrainingOptions {
                learning_rate: 0.1,
                loss_algorithm: MeanSquared::new(), // The Mean Squared loss function
                should_print_information: true,     // Should be verbose
                epochs: 5000,
            },
        )
        .await
        .unwrap();

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
=======
            &mut TrainingOptions {
                learning_rate: 0.1,
                loss_algorithm: MeanSquared::new(), // The Mean Squared loss function
                should_print_information: true,     // Should be verbose
                epochs: 5000,
            },
        )
        .unwrap();

    // for saving Intricate uses the 'savefile' crate
    // that simply needs to call the 'save_file' function to the path you want
    // for the Model as follows
    xor_model.sync_gpu_data_with_cpu().unwrap();
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