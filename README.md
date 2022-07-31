# Intricate

[![Crates.io](https://img.shields.io/crates/v/intricate.svg?label=intricate)](https://crates.io/crates/intricate)

A GPU accelerated library that creates/trains/runs neural networks in pure safe Rust code.

## Architechture overview

Intricate has a layout very similar to popular libraries out there such as Keras.

### Models

As said before, similar to Tensorflow, Intricate defines Models as basically
a list of `Layers` that are explained down bellow.

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
let mut layers: Vec<Box<dyn Layer<f64>>> = Vec::new();

//                      inputs_amount|outputs_amount
layers.push(Box::new(DenseF64::new(2, 3)));
layers.push(Box::new(TanHF64::new())); // activation functions are layers
layers.push(Box::new(DenseF64::new(3, 1)));
layers.push(Box::new(TanHF64::new()));
```

### Creating the model with the layers

```rs
// Instantiate our model using the layers
let mut xor_model = ModelF64::new(layers);
// mutable because the 'fit' method lets the layers tweak themselves
```

### Fitting our model

```rs
xor_model.fit(
    &training_inputs, 
    &expected_outputs, 
    TrainingOptionsF64 {
        learning_rate: 0.1,
        loss_algorithm: Box::new(MeanSquared), // The Mean Squared loss function
        should_print_information: true, // Should be verbose
        instantiate_gpu: false, // Should initialize WGPU Device and Queue for GPU layers
        epochs: 10000,
    },
).await;
```

As you can see it is extremely easy creating these models, and blazingly fast as well.

Although if you wish to do (just like in the actual XoR example) you 
could write this using the F32 version of numbers which is 30% faster 
overall and uses half the RAM but at the price of less precision.

## How to save and load models

Intricate implements a few functions for each layer that saves and loads the necessary
layer information to some file using the [savefile](https://github.com/avl/savefile) crate.

But a layer can save and load the data anyway it sees fit, as long as it does what the trait Layer requires. 

### Saving the model

To load and save data, as an example, say for the XoR model
we trained above,  we can just call the `save` function as such:

```rs
xor_model.layers[0].save("xor-model-first-dense.bin", 0).unwrap();
xor_model.layers[2].save("xor-model-second-dense.bin", 0).unwrap();
```

And we save only the Dense layers here because the Activation layers don't really
hold any valuable information, only the Dense layers do.

### Loading the model

As for the loading of the data we must create some dummy dense layers and tell
them to load their data from the paths created above

```rs
let mut first_dense: Box<DenseF32> = Box::new(DenseF32::dummy());
first_dense.load("xor-model-first-dense.bin", 0).unwrap();
let mut second_dense: Box<DenseF32> = Box::new(DenseF32::dummy()); 
second_dense.load("xor-model-second-dense.bin", 0).unwrap();

let mut new_layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
new_layers.push(first_dense);
new_layers.push(Box::new(TanHF32::new()));
new_layers.push(second_dense);
new_layers.push(Box::new(TanHF32::new()));

let loaded_xor_model = ModelF32::new(new_layers);
```

## Things to be done still

- writing some kind of macro to generate the code for f32 and f64 versions of certain structs and traits to not have duplicated code.
- improve the GPU shaders, perhaps finding a way to send the full unflattened matrices to the GPU instead of sending just a flattened array.
- create GPU accelerated activations and loss functions as to make everything GPU accelerated.
- perhaps write some shader to calculate the Model **loss** to **output** gradient (derivatives).
- implement convolutional layers and perhaps even solve some image classification problems in a example
- add a example that uses GPU acceleration
