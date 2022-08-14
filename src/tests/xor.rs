#[allow(unused_imports)]
use opencl3::error_codes::ClError;
#[allow(unused_imports)]
use crate::{
    types::CompilationOrOpenCLError,
    utils::opencl::DeviceType
};

#[allow(unused_imports)]
use crate::{
    layers::activations::TanH,
    layers::Dense,
    loss_functions::MeanSquared,
    loss_functions::LossFunction,
    model::Model,
    types::{ModelLayer, ModelLossFunction, TrainingOptions},
    utils::{setup_opencl, OpenCLState},
};

#[test]
fn should_decrease_error() -> () {
    let layers: Vec<ModelLayer> = vec![
        Dense::new(2, 3),
        TanH::new (3),
        Dense::new(3, 1),
        TanH::new (1),
    ];

    let mut model = Model::new(layers);
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();
    model.init(&opencl_state).unwrap();

    let training_input_samples = vec![
        vec![0.0_f32, 0.0_f32],
        vec![1.0_f32, 0.0_f32],
        vec![0.0_f32, 1.0_f32],
        vec![1.0_f32, 1.0_f32],
    ];

    let training_output_samples = vec![
        vec![0.0_f32],
        vec![1.0_f32],
        vec![1.0_f32],
        vec![0.0_f32],
    ];


    let last_loss = model
        .fit(
            &training_input_samples,
            &training_output_samples,
            &mut TrainingOptions {
                loss_algorithm: MeanSquared::new(), 
                learning_rate: 0.1,
                should_print_information: true,
                epochs: 5000,
            },
        ).unwrap()
        .unwrap();

    let max_loss = 0.1;

    assert!(last_loss <= max_loss);
}