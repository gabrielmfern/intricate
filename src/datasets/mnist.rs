use std::io::Read;

use lazy_static::lazy_static;
use rayon::prelude::*;

use reqwest::blocking::get;
use flate2::read::GzDecoder;

use super::{read_3d_ubyte_file, read_1d_ubyte_file};

const IMAGES_SOURCE_URL: &str = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
const LABELS_SOURCE_URL: &str = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";

const TEST_IMAGES_SOURCE_URL: &str = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
const TEST_LABELS_SOURCE_URL: &str = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

/// Gets the training images with the pixel in the image flattened into just one vector per
/// training sample.
///
/// Will also normalize the colors from `0 to 1` by dividing by **255*8.
pub fn get_training_inputs() -> Vec<Vec<f32>> {
    lazy_static! {
        static ref IMAGES_SOURCE: Vec<u8> = GzDecoder::new(
            get(IMAGES_SOURCE_URL)
                .expect("Unable to fetch MNIST training images from source")
        ).bytes()
        .collect::<Result<Vec<u8>, std::io::Error>>()
        .expect("unable to decompress MNIST training images");
    }

    println!("reading the MNIST digit database images");
    read_3d_ubyte_file(IMAGES_SOURCE.as_slice(), |byte| byte as f32 / 255.0)
        .par_iter()
        .map(|image| image.par_iter().flatten().map(|x| *x).collect::<Vec<f32>>())
        .collect()
}

#[test]
fn should_get_training_inputs_correctly() {
    let training_inputs = get_training_inputs();
    let training_outputs = get_training_outputs();
    const IMAGE_WIDTH: usize = 28;
    const IMAGE_HEIGHT: usize = 28;

    let mut first_digit = [[0.0; IMAGE_WIDTH]; IMAGE_HEIGHT];

    training_inputs[1].iter().enumerate().for_each(|(i, p)| {
        let y = (i as f32 / IMAGE_WIDTH as f32).floor() as usize;
        let x = i % IMAGE_WIDTH;

        first_digit[y][x] = *p;
    });

    first_digit.iter().for_each(|row| {
        row.iter().for_each(|pixel| {
            print!("{:.1}", pixel);
        });
        print!("\n");
    });

    assert_eq!(training_outputs[1], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

/// Gets the training labels of the MNIST dataset ready to be given as input to a Intricate model.
pub fn get_training_outputs() -> Vec<Vec<f32>> {
    lazy_static! {
        static ref IMAGES_LABELS: Vec<u8> = GzDecoder::new(
            get(LABELS_SOURCE_URL)
                .expect("Unable to fetch MNIST training labels from source")
        ).bytes()
        .collect::<Result<Vec<u8>, std::io::Error>>()
        .expect("unable to decompress MNIST training labels");
    }

    println!("reading the MNIST digit database labels");
    read_1d_ubyte_file(IMAGES_LABELS.as_slice())
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
    lazy_static! {
        static ref TEST_IMAGES: Vec<u8> = GzDecoder::new(
            get(TEST_IMAGES_SOURCE_URL)
                .expect("Unable to fetch MNIST testing images from source")
        ).bytes()
        .collect::<Result<Vec<u8>, std::io::Error>>()
        .expect("unable to decompress MNIST testing images");
    }

    println!("reading the MNIST digit testing database images");
    read_3d_ubyte_file(TEST_IMAGES.as_slice(), |byte| byte as f32 / 255.0)
        .par_iter()
        .map(|image| image.par_iter().flatten().map(|x| *x).collect::<Vec<f32>>())
        .collect()
}

/// Gets the outputs for testing the Model that should be associated with their respective inputs
pub fn get_test_outputs() -> Vec<Vec<f32>> {
    lazy_static! {
        static ref TEST_LABELS: Vec<u8> = GzDecoder::new(
            get(TEST_LABELS_SOURCE_URL)
                .expect("Unable to fetch MNIST testing labels from source")
        ).bytes()
        .collect::<Result<Vec<u8>, std::io::Error>>()
        .expect("unable to decompress MNIST testing labels");
    }

    println!("reading the MNIST digit testing database labels");
    read_1d_ubyte_file(TEST_LABELS.as_slice())
        .par_iter()
        .map(|digit_index| {
            let mut output = vec![0.0; 10];
            output[*digit_index as usize] = 1.0;
            output
        })
        .collect()
}