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

```rs
let training_inputs = Vec::from([
    Vec::from([0.0, 0.0]),
    Vec::from([0.0, 1.0]),
    Vec::from([1.0, 0.0]),
    Vec::from([1.0, 1.0]),
]);

let expected_outputs = Vec::from([
    Vec::from([0.0]),
    Vec::from([1.0]),
    Vec::from([1.0]),
    Vec::from([0.0]),
]);
```

### Setting up the layers

```rs
let mut layers: Vec<ModelLayer> = Vec::new();

//                      inputs_amount|outputs_amount
layers.push(ModelLayer::Dense(Dense::new(2, 3)));
layers.push(ModelLayer::TanH(TanH::new())); // activation functions are layers
layers.push(ModelLayer::Dense(Dense::new(3, 1)));
layers.push(Modellayer::TanH(TanH::new()));
```

### Creating the model with the layers

```rs
// Instantiate our model using the layers
let mut xor_model = ModelF64::new(layers);
```

We make the model mut because we will call `fit` for training our model
which will tune each of the layers when necessary.

### Fitting our model

For training our Model we just need to call the `fit`
method and pass in some parameters as follows:

```rs
xor_model.fit(
    &training_inputs, 
    &expected_outputs, 
    TrainingOptions {
        learning_rate: 0.1,
        loss_algorithm: ModelLossFunction::MeanSquared(MeanSquared), // The Mean Squared loss function
        should_print_information: true, // Should or not be verbose
        instantiate_gpu: false, // Should or not not initialize WGPU Device and Queue for GPU layers
        epochs: 10000,
    },
).await;
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

## Things to be done still

- improve the GPU computations perhaps using Vulkano instead of wgpu, and separating GPU computations in another crate with the 'intricate' prefix
- separate Intricate into more than one crate as to make development more lightweight with rust-analyzer
- create GPU accelerated activations and loss functions as to make everything GPU accelerated.
- perhaps write some shader to calculate the Model **loss** to **output** gradient (derivatives).
- implement convolutional layers and perhaps even solve some image classification problems in a example
- add a example that uses GPU acceleration