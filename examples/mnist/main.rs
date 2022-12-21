use intricate::{
    datasets::mnist,
    layers::{
        activations::{ReLU, SoftMax, Sigmoid},
        Conv2D, Dense, Layer,
    },
    loss_functions::CategoricalCrossEntropy,
    optimizers,
    types::TrainingOptions,
    utils::{opencl::DeviceType, setup_opencl},
    Model,
};
use savefile::save_file;

const MODEL_PATH: &str = "mnist-model.bin";

fn main() -> () {
    // don't really recommend using CPU for this, but it is possible as long as you have drivers
    let state = setup_opencl(DeviceType::GPU).expect("unable to setup OpenCL");

    let mut mnist_model: Model = Model::new(vec![
        Conv2D::new((28, 28), (3, 3)),
        ReLU::new(26 * 26),
        // Dense::new(28 * 28, 10 * 10),
        // ReLU::new(10 * 10),

        Dense::new(26 * 26, 10),
        // Dense::new(10 * 10, 10),
        // ReLU::new(10),
        SoftMax::new(10),
    ]);

    mnist_model
        .init(&state)
        .expect("unable to initialize Mnist model");

    let mut loss_fn = CategoricalCrossEntropy::new();
    let mut optimizer = optimizers::Adam::new(0.001, 0.9, 0.999, 0.0000001);

    let training_inputs = mnist::get_training_inputs();
    let training_outputs = mnist::get_training_outputs();


    mnist_model
        .fit(
            &training_inputs,
            &training_outputs,
            &mut TrainingOptions::new(&mut loss_fn, &mut optimizer)
                .set_batch_size(512)
                .set_epochs(100)
                .should_compute_loss(true).expect("unable to define that the loss should be computed")
                .should_print_loss(true).expect("unable to define that the loss should be printed")
        )
        .expect("unable to fit Mnist model");

    mnist_model
        .sync_data_from_buffers_to_host()
        .expect("unable to sync weights from the GPU");

    // dbg!(mnist_model.layers[0].get_flattened_parameter_data("weights"));
    // dbg!(mnist_model.layers[0].get_flattened_parameter_data("biases"));

    save_file(MODEL_PATH, 0, &mnist_model).expect("unable to save Mnist model");
}