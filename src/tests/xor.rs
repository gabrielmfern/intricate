#[allow(unused_imports)]
use opencl3::error_codes::ClError;
#[allow(unused_imports)]
use crate::utils::opencl::DeviceType;

#[allow(unused_imports)]
use crate::{
    layers::activations::TanH,
    layers::Dense,
    optimizers::BasicOptimizer,
    loss_functions::MeanSquared,
    loss_functions::LossFunction,
    model::Model,
    types::{ModelLayer, TrainingOptions},
    utils::{setup_opencl, OpenCLState},
};

// Seems to fail every time I try to run this with DeviceType::CPU
// ending up with a bunch of NaN, no idea why, perhaps something with my drivers
// are wrong
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


    let mut loss = MeanSquared::new();
    let mut optimizer = BasicOptimizer::new(0.1);

    // Fit the model however many times we want
    let losses = model
        .fit(
            &training_input_samples,
            &training_output_samples,
            &mut TrainingOptions {
                loss_fn: &mut loss,
                verbose: true,     // Should be verbose
                compute_loss: true,
                batch_size: 4,
                optimizer: &mut optimizer,
                epochs: 10000,
            },
        )
        .unwrap();

    let max_loss = 0.1;
    let last_loss = losses.last().unwrap();

    assert!(last_loss <= &max_loss);
}