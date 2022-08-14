#[allow(unused_imports)]
use opencl3::error_codes::ClError;
#[allow(unused_imports)]
use crate::types::CompilationOrOpenCLError;
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
fn should_decrease_error() -> Result<(), CompilationOrOpenCLError> {
    let mut layers: Vec<ModelLayer> = Vec::new();
    layers.push(ModelLayer::Dense(Dense::new(2, 3)));
    layers.push(ModelLayer::TanH(TanH::new(3)));
    layers.push(ModelLayer::Dense(Dense::new(3, 1)));
    layers.push(ModelLayer::TanH(TanH::new(1)));

    let mut model = Model::new(layers);
    let opencl_state = setup_opencl()?;
    model.init(&opencl_state)?;

    let training_input_samples = Vec::from([
        Vec::from([0.0_f32, 0.0_f32]),
        Vec::from([1.0_f32, 0.0_f32]),
        Vec::from([0.0_f32, 1.0_f32]),
        Vec::from([1.0_f32, 1.0_f32]),
    ]);

    let training_output_samples = Vec::from([
        Vec::from([0.0_f32]),
        Vec::from([1.0_f32]),
        Vec::from([1.0_f32]),
        Vec::from([0.0_f32]),
    ]);


    let last_loss = model
        .fit(
            &training_input_samples,
            &training_output_samples,
            &mut TrainingOptions {
                loss_algorithm: ModelLossFunction::MeanSquared(MeanSquared::new()),
                learning_rate: 0.1,
                should_print_information: true,
                epochs: 5000,
            },
        )?
        .unwrap();

    let max_loss = 0.1;

    assert!(last_loss <= max_loss);

    Ok(())
}