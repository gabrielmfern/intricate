#[allow(unused_imports)]
use opencl3::error_codes::ClError;

#[allow(unused_imports)]
use crate::{
    layers::activations::tanh_gpu::TanHGPU,
    layers::dense_gpu::DenseGPU,
    loss_functions::mean_squared::OpenCLMeanSquared,
    loss_functions::OpenCLLossFunction,
    utils::{OpenCLState, setup_opencl},
    model_gpu::{GPUModel, GPUModelLayer, GPUModelLossFunction, GPUTrainingOptions}
};

#[test]
fn should_decrease_error() -> Result<(), ClError> {
    let mut layers: Vec<GPUModelLayer> = Vec::new();
    layers.push(GPUModelLayer::Dense(DenseGPU::new(2, 3)));
    layers.push(GPUModelLayer::TanH(TanHGPU::new(3)));
    layers.push(GPUModelLayer::Dense(DenseGPU::new(3, 1)));
    layers.push(GPUModelLayer::TanH(TanHGPU::new(1)));

    let mut model = GPUModel::new(layers);
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

    let last_loss = model.fit(
        &training_input_samples,
        &training_output_samples,
        &mut GPUTrainingOptions {
            loss_algorithm: GPUModelLossFunction::MeanSquared(OpenCLMeanSquared::new()),
            learning_rate: 0.1,
            should_print_information: true,
            epochs: 10000,
        },
    )?.unwrap();

    let max_loss = 0.1;

    assert!(last_loss <= max_loss);

    Ok(())
}