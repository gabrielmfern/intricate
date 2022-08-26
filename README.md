# Intricate

[![Crates.io](https://img.shields.io/crates/v/intricate.svg?label=intricate)](https://crates.io/crates/intricate)
[![Crates.io](https://img.shields.io/crates/dv/intricate)](https://crates.io/crates/intricate)
![github.com](https://img.shields.io/github/license/gabrielmfern/intricate)
![github.com](https://img.shields.io/github/commit-activity/m/gabrielmfern/intricate)

A GPU accelerated library that creates/trains/runs neural networks in safe Rust code.

---

### Table of contents

* [Architechture overview](#architechture-overview)
    * [Models](#models)
    * [Layers](#layers)
    * [Optimizers](#optimizers)
    * [Loss Functions](#loss-functions)
* [XoR using Intricate](#xor-using-intricate)
    * [Setting up the training data](#setting-up-the-training-data)
    * [Setting up the layers](#setting-up-the-layers)
    * [Setting up OpenCL](#setting-up-opencls-state)
    * [Fitting our Model](#fitting-our-model)
* [How to save and load models](#how-to-save-and-load-models)
    * [Saving the Model](#saving-the-model)
    * [Loading the Model](#loading-the-model)
* [Things to be done still](#things-to-be-done-still)

---

## Architechture overview

Intricate has a layout very similar to popular libraries out there such as Keras.

It consists at the surface of a [Model](#models), which consists then 
of [Layers](#layers) which can be adjusted using a [Loss Function](#loss-functions)
that is also helped by a [Optimizer](#optimizers).

### Models

As said before, similar to Keras, Intricate defines Models as basically
a list of [Layers](#layers).

A model does not have much logic in it, mostly it delegates most of the work to the layers,
all that it does is orchestrate how the layers should work together and how the data goes from
a layer to another.

### Layers

Every layer receives **inputs** and returns **outputs** following some rule that they must define. 

They must also implement four methods that together constitute backpropagation:

- `optimize_parameters`
- `compute_gradients`
- `apply_gradients`
- `compute_loss_to_input_derivatives`

Mostly the optimize_parameters will rely on an [Optimizer](#optimizers) that will try to improve
the parameters that the Layer allows it to optimize.

These methods together will be called sequentially to do backpropagation in the Model and
using the results from the `compute_loss_to_input_derivatives` we will then do the same for
the last layer and so on.

These layers can be really any type of transformation on the inputs and outputs.
An example of this is the activation functions in Intricate which are actual 
layers instead of being one with other layers
which does simplify calculations tremendously and works like a charm.

### Optimizers

Optimizers the do just what you might think, they optimize.

Specifically they optimize both the parameters a Layer allows them to optimize, as well
as the [Layer](#layers)'s gradients so that the Layer can use them to apply the optimized gradients on itself.

This is useful because anyone using Intricate can develop and perhaps debug a Optimizer to see how well it does
for certain use cases which is very good for where I want Intricate to go. All you have to do is create some struct
that implements the `Optimizer` trait.

Intricate currently only does have one optimizer since it is still on heavy development and changing
architecture constantly so writing many implementations would be really annoying to change later.

### Loss Functions

Loss Functions are just basically some implementations of a certain trait that are used
to determine how bad a Model is. 

Loss Functions are **NOT** used in a layer, they are used
for the Model itself. Even though a Layer will use derivatives with respect 
to the loss they don't really communicate with the Loss Function directly.

---

## XoR using Intricate

If you look at the `examples/` in the repository 
you will find XoR implemented using Intricate. 
The following is basically just that example with some separate explanation.

### Setting up the training data

```rust
let training_inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];

let expected_outputs = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0],
];
```

### Setting up the layers

```rust
use intricate::layers::{
    activations::TanH,
    Dense
};
let mut layers: Vec<ModelLayer> = vec![
    Dense::new(2, 3), // inputs amount, outputs amount
    TanH::new (3),
    Dense::new(3, 1),
    TanH::new (1),
];
```

### Creating the model with the layers

```rust
use intricate::Model;
// Instantiate our model using the layers
let mut xor_model = Model::new(layers);
```

We make the model `mut` because we will call `fit` for training our model
which will tune each of the layers when necessary.

### Setting up OpenCL's state

Since Intricate does use OpenCL under the hood for doing calculations,
we do need to initialize a `OpenCLState` which is just a struct
containing some necessary OpenCL stuff:

```rust
use intricate::utils::{
    setup_opencl,
    DeviceType
}
//              you can change this device type to GPU if you want
let opencl_state = setup_opencl(DeviceType::CPU).unwrap();
```

For our Model to be able to actually do computations, we need to pass the OpenCL state 
into the `init` method inside of the Model as follows:

```rust
xor_model.init(&opencl_state).unwrap();
```

### Fitting our model

For training our Model we just need to call the `fit`
method and pass in some parameters as follows:

```rust
use intricate::{
    loss_functions::MeanSquared,
    optimizers::BasicOptimizer,
    types::{TrainingOptions, TrainingVerbosity},
};

let mut loss = MeanSquared::new();
let mut optimizer = BasicOptimizer::new(0.1);

// Fit the model however many times we want
xor_model
    .fit(
        &training_inputs,
        &expected_outputs,
        &mut TrainingOptions {
            loss_fn: &mut loss,
            verbosity: TrainingVerbosity {
                show_current_epoch: true, // Show a current epoch message such as `epoch #5`

                show_epoch_progress: true, // Show the training steps process for each epoch in 
                                           // a indicatif progress bar

                show_epoch_elapsed: true, // Show the time elapsed in the epoch

                print_loss: true, // Show the loss after an epoch of training
            },
            compute_loss: true,
            optimizer: &mut optimizer,
            batch_size: 4, // Intricate will always use Mini-batch Gradient Descent under the hood
                           // since with it you can have all other variants of Gradient Descent.
                           // So this is basically the size of the batch being used in gradient descent.
            epochs: 500,
        },
    )
    .unwrap();
```

As you can see it is extremely easy creating these models, and blazingly fast as well.

---

## How to save and load models

For saving and loading models Intricate uses the [savefile](https://github.com/avl/savefile) crate which makes it very simple and fast to save models.

### Saving the model

As an example let's try saving and loading our XoR model.
For doing that we will first need to sync all of the relevant layer information
of the Model with OpenCL's `host`, (or just with the CPU), and then we will need
to call the `save_file` method as follows:

```rust
xor_model.sync_data_from_buffers_to_host().unwrap(); // sends the weights and biases from 
                                                     // OpenCL buffers to Rust Vec's
save_file("xor-model.bin", 0, &xor_model).unwrap();
```

### Loading the model

As for loading our XoR model, we just need to call the 
counterpart of the save_file method: `load_file`.

```rust
let mut loaded_xor_model: Model = load_file("xor-model.bin", 0).unwrap();
```

Now of curse, the savefile crate cannot load in the data to the GPU, so if you want
to use the Model after loading it, you **must** call the `init` method in the `loaded_xor_model`
(done in examples/xor.rs).

## Things to be done still

- separate Intricate into more than one crate as to make development more lightweight with rust-analyzer
- implement convolutional layers and perhaps even solve some image classification problems in a example
- have some feature of Intricate, should be optional, that would contain preloaded datasets, such as MNIST and others
- I saw on another crate, (don't remember the name), a pretty cool feature, which was a halting condition that would determine when for the training to stop, perhaps adding that would be nice
- add a way to send into the training process a callback closure that would be called everytime a epoch finished or even a step too with some cool info
- make an example after doing the thing above ^, that uses that same function to plot the loss realtime using a crate like `textplots`
