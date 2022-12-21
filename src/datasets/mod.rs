//! The module for the datasets feature of Intricate.
//!
//! `DISCLAIMER`: All of the datasets will be downloaded, decompressed if necessary and then read into the proper
//! Rust types.
//!
//! Currently contains the following datasets.
//! - MNIST

use indicatif::ProgressIterator;

/// The module containing the MNIST dataset
pub mod mnist;

#[allow(dead_code)]
fn get_dimensions_of_ubyte_dataset(source: &[u8], dimensions_amount: usize) -> Vec<usize> {
    let mut dimensions = Vec::with_capacity(dimensions_amount);

    for dimension in 0..dimensions_amount {
        let bytes = source
            .iter()
            .skip(4 * (dimension + 1))
            .take(4 * (dimension + 2))
            .map(|b| *b)
            .collect::<Vec<u8>>();

        dimensions.push(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize);
    }

    dimensions
}

#[allow(dead_code)]
fn read_1d_ubyte_file(source: &[u8]) -> Vec<f32> {
    source
        .iter()
        .skip(8) // skip the magic number
        .progress()
        .map(|byte| *byte as f32)
        .collect::<Vec<f32>>()
}

#[allow(dead_code)]
fn read_3d_ubyte_file<TransformationFunc>(
    source: &[u8],
    transformation: TransformationFunc,
) -> Vec<Vec<Vec<f32>>>
where
    TransformationFunc: Fn(u8) -> f32,
{
    let dimensions = get_dimensions_of_ubyte_dataset(source, 3);
    let samples_amount = dimensions[0];
    let width = dimensions[1];
    let height = dimensions[2];

    let mut data: Vec<Vec<Vec<f32>>> = Vec::with_capacity(samples_amount);

    for (i, byte) in source.iter().skip(16).enumerate().progress() {
        let sample_index = (i as f64 / (width * height) as f64).floor() as usize;
        let row_index = ((i % (width * height)) as f64 / width as f64).floor() as usize;

        if data.len() == sample_index {
            if sample_index > 0 {
                assert_eq!(data[sample_index - 1].len(), height);
            }

            data.push(Vec::with_capacity(height));
        }

        if data[sample_index].len() == row_index {
            if row_index > 0 {
                assert_eq!(data[sample_index][row_index - 1].len(), width);
            }

            data[sample_index].push(Vec::with_capacity(width));
        }

        data[sample_index][row_index].push(transformation(*byte));
    }

    data
}