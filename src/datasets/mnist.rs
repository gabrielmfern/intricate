use rayon::prelude::*;

use super::{read_3d_ubyte_file, read_1d_ubyte_file};

const IMAGES_SOURCE: &[u8] = include_bytes!("mnist_data/mnist-train-images.idx3-ubyte");
const LABELS_SOURCE: &[u8] = include_bytes!("mnist_data/mnist-train-labels.idx1-ubyte");

const TEST_IMAGES_SOURCE: &[u8] = include_bytes!("mnist_data/mnist-test-images.idx3-ubyte");
const TEST_IMAGES_LABELS: &[u8] = include_bytes!("mnist_data/mnist-test-labels.idx1-ubyte");

/// Gets the training images with the pixel in the image flattened into just one vector per
/// training sample.
///
/// Will also normalize the colors from `0 to 1` by dividing by **255*8.
pub fn get_training_inputs() -> Vec<Vec<f32>> {
    println!("reading the MNIST digit database images");
    read_3d_ubyte_file(IMAGES_SOURCE, |byte| byte as f32 / 255.0)
        .par_iter()
        .map(|image| image.par_iter().flatten().map(|x| *x).collect::<Vec<f32>>())
        .collect()
}

/// Gets the training labels of the MNIST dataset ready to be given as input to a Intricate model.
pub fn get_training_outputs() -> Vec<Vec<f32>> {
    println!("reading the MNIST digit database labels");
    read_1d_ubyte_file(LABELS_SOURCE)
        .par_iter()
        .map(|digit_index| {
            let mut output = vec![0.0; 10];
            output[*digit_index as usize] = 1.0;
            output
        })
        .collect()
}

/// Gets the inputs for testing the Model that should be associated with their respective outputs
pub fn get_test_inputs() -> Vec<Vec<f32>> {
    println!("reading the MNIST digit testing database images");
    read_3d_ubyte_file(TEST_IMAGES_SOURCE, |byte| byte as f32 / 255.0)
        .par_iter()
        .map(|image| image.par_iter().flatten().map(|x| *x).collect::<Vec<f32>>())
        .collect()
}

/// Gets the outputs for testing the Model that should be associated with their respective inputs
pub fn get_test_outputs() -> Vec<Vec<f32>> {
    println!("reading the MNIST digit testing database labels");
    read_1d_ubyte_file(TEST_IMAGES_LABELS)
        .par_iter()
        .map(|digit_index| {
            let mut output = vec![0.0; 10];
            output[*digit_index as usize] = 1.0;
            output
        })
        .collect()
}
