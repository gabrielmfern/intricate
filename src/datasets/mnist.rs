const IMAGES_SOURCE: &[u8] = include_bytes!("mnist-train-images.idx3-ubyte");
const LABELS_SOURCE: &[u8] = include_bytes!("mnist-train-labels.idx1-ubyte");

const TEST_IMAGES_SOURCE: &[u8] = include_bytes!("mnist-test-images.idx3-ubyte");
const TEST_IMAGES_LABELS: &[u8] = include_bytes!("mnist-test-labels.idx1-ubyte");

/// Gets the training images with the pixel in the image flattened into just one vector per
/// training sample.
///
/// Will also normalize the colors from `0 to 1` by dividing by **255*8.
pub fn get_training_inputs() -> Vec<Vec<f32>> {
    let mut sample_inputs: Vec<Vec<f32>> = Vec::with_capacity(60_000);

    let image_size = 28 * 28;
    for (i, byte) in IMAGES_SOURCE.iter().skip(16).enumerate() {
        let sample_index = (i as f64 / image_size as f64).floor() as usize;

        if sample_inputs.len() == sample_index {
            if sample_index > 0 {
                assert_eq!(sample_inputs[sample_index - 1].len(), image_size);
            }

            sample_inputs.push(Vec::with_capacity(image_size));
        }

        sample_inputs[sample_index].push(*byte as f32 / 255.0);
    }

    assert_eq!(sample_inputs.len(), 60_000);

    sample_inputs
}

/// Gets the training labels of the MNIST dataset ready to be given as input to a Intricate model.
pub fn get_training_outputs() -> Vec<Vec<f32>> {
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(60_000);

    for byte in LABELS_SOURCE.iter().skip(8) {
        let mut output = vec![0.0; 10];
        output[*byte as usize] = 1.0;

        samples.push(output);
    }

    assert_eq!(samples.len(), 60_000);

    samples
}

/// Gets the inputs for testing the Model that should be associated with their respective outputs
pub fn get_test_inputs() -> Vec<Vec<f32>> {
    let mut sample_inputs: Vec<Vec<f32>> = Vec::with_capacity(10_000);

    let image_size = 28 * 28;
    for (i, byte) in TEST_IMAGES_SOURCE.iter().skip(16).enumerate() {
        let sample_index = (i as f64 / image_size as f64).floor() as usize;

        if sample_inputs.len() == sample_index {
            if sample_index > 0 {
                assert_eq!(sample_inputs[sample_index - 1].len(), image_size);
            }

            sample_inputs.push(Vec::with_capacity(image_size));
        }

        sample_inputs[sample_index].push(*byte as f32 / 255.0);
    }

    assert_eq!(sample_inputs.len(), 10_000);

    sample_inputs
}

/// Gets the outputs for testing the Model that should be associated with their respective inputs
pub fn get_test_outputs() -> Vec<Vec<f32>> {
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(10_000);

    for byte in TEST_IMAGES_LABELS.iter().skip(8) {
        let mut output = vec![0.0; 10];
        output[*byte as usize] = 1.0;

        samples.push(output);
    }

    assert_eq!(samples.len(), 10_000);

    samples
}