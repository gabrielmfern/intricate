use std::path::Path;

use intricate::{
    datasets::mnist,
    layers::{
        activations::{SoftMax, TanH},
        Dense,
    },
    loss_functions::CategoricalCrossEntropy,
    optimizers::AdagradOptimizer,
    types::{TrainingOptions, TrainingVerbosity},
    utils::{opencl::DeviceType, setup_opencl},
    Model,
};
use savefile::{save_file, load_file};

const MODEL_PATH: &str = "mnist-model.bin";

fn main() -> () {
    // don't really recommend using CPU for this, but it is possible as long as you have drivers
    let state = setup_opencl(DeviceType::GPU).expect("unable to setup OpenCL");

    let training_inputs = mnist::get_training_inputs();
    let training_outputs = mnist::get_training_outputs();

    let mut mnist_model: Model;
    if Path::new(MODEL_PATH).exists() {
        mnist_model = load_file(MODEL_PATH, 0).expect("unable to load the model");
    } else {
        // this model does work, but it will not get very far without Conv layers (not yet
        // implemented)
        mnist_model = Model::new(vec![
            Dense::new(28 * 28, 500),
            TanH::new(500),
            Dense::new(500, 450),
            TanH::new(450),
            Dense::new(450, 250),
            TanH::new(250),
            Dense::new(250, 100),
            TanH::new(100),
            Dense::new(100, 10),
            SoftMax::new(10),
        ]);
    }

    mnist_model
        .init(&state)
        .expect("unable to initialize Mnist model");

    let mut loss_fn = CategoricalCrossEntropy::new();
    let mut optimizer = AdagradOptimizer::new(0.002, 0.00000001);

    mnist_model
        .fit(
            &training_inputs,
            &training_outputs,
            &mut TrainingOptions {
                loss_fn: &mut loss_fn,
                optimizer: &mut optimizer,
                batch_size: 64,
                verbosity: TrainingVerbosity {
                    show_current_epoch: true,
                    show_epoch_progress: true,
                    show_epoch_elapsed: true,
                    print_loss: true,
                    print_accuracy: false,
                    halting_condition_warning: false,
                },
                halting_condition: None,
                compute_loss: true,
                compute_accuracy: false,
                epochs: 100,
            },
        )
        .expect("unable to fit Mnist model");

    mnist_model
        .sync_data_from_buffers_to_host()
        .expect("unable to sync weights from the GPU");

    save_file(MODEL_PATH, 0, &mnist_model).expect("unable to save Mnist model");
}