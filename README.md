# Intricate

[![Crates.io](https://img.shields.io/crates/v/intricate.svg?label=intricate)](https://crates.io/crates/intricate)
[![Crates.io](https://img.shields.io/crates/dv/intricate)](https://crates.io/crates/intricate)
![github.com](https://img.shields.io/github/license/gabrielmfern/intricate)
![github.com](https://img.shields.io/github/commit-activity/m/gabrielmfern/intricate)

A GPU accelerated library that creates/trains/runs neural networks in pure safe Rust code.

## Architechture overview

Intricate has a layout very similar to popular libraries out there such as Keras.

### Models

As said before, similar to Keras from Tensorflow, Intricate defines Models as basically
a list of `Layers` and the definition for "layer" is as follows.

### Layers

Every layer receives **inputs** and returns **outputs**, 
they must also implement a `back_propagate` method that 
will mutate the layer if needed and then return the derivatives
of the loss function with respected to the inputs, 
written with **I** as the inputs of the layer, 
**E** as the loss and **O** as the outputs of the layer:

```
dE/dI <- Model <- dE/dO
```

These layers can be anything you want and just propagates the previous inputs
to the next inputs for the next layer or for the outputs of the whole Model.

There are a few activations already implemented, but still many to be implemented.

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
use intricate::utils::setup_opencl;
let opencl_state = setup_opencl().unwrap();
```

After setting it up we need to initialize the Model using this state
for it to save the reference to the state so that it can do the
things it needs with it as such:

```rust
xor_model.init(&opencl_state).unwrap();
```

Beware that as v0.3.0 of Intricate any method called before `init`
will panic because they do not have the OpenCL state to do calculations.

### Fitting our model

For training our Model we just need to call the `fit`
method and pass in some parameters as follows:

```rust
xor_model.fit(
    &training_inputs, 
    &expected_outputs, 
    TrainingOptions {
        learning_rate: 0.1,
        loss_algorithm: MeanSquared::new(), // The Mean Squared loss function
        should_print_information: true, // Should or not be verbose
        epochs: 10000,
    },
).unwrap(); // Will return an Option containing the last loss after training
```

As you can see it is extremely easy creating these models, and blazingly fast as well.

## How to save and load models

For saving and loading models Intricate uses the [savefile](https://github.com/avl/savefile) crate which makes it very simple and fast to save models.

### Saving the model

To load and save data, as an example, say for the XoR model
we trained above, we can just call the `save_file` function as such:

```rs
save_file("xor-model.bin", 0, &xor_model).unwrap();
```

Which will save all of the configuration of the XoR Model including its
activation functions and Dense layer's weights and biases and of the other types of layers
information.

### Loading the model

As for the loading of the data we must create some dummy dense layers and tell
them to load their data from the paths created above

```rs
let mut loaded_xor_model: Model = load_file("xor-model.bin", 0).unwrap();
```

Now of curse, **savefile** cannot load in the GPU state so if you want
to use the loaded Model you **must** call the `setup_opencl` again
and initialize the Model with it.

## Things to be done still

- separate Intricate into more than one crate as to make development more lightweight with rust-analyzer
- implement convolutional layers and perhaps even solve some image classification problems in a example