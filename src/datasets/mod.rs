//! The module for the datasets feature of Intricate.
//!
//! Currently contains the following datasets.
//! - MNIST

/// The module containing the MNIST dataset
#[cfg(feature = "mnist")]
pub mod mnist;

fn get_dimensions_of_ubyte_dataset(source: &[u8], dimensions_amount: usize) -> Vec<usize> {
    let mut dimensions = Vec::with_capacity(dimensions_amount);

    for dimension in 0..dimensions_amount {
        let bytes = source
            .iter()
            .skip(4 * dimension)
            .take(4 * (dimension + 1))
            .map(|b| *b)
            .collect::<Vec<u8>>();

        dimensions.push(i32::from_be_bytes([
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3],
        ]) as usize);
    }

    dimensions
}

fn read_1d_ubyte_file(source: &[u8]) -> Vec<f32> {
    let data = Vec::new();

    let samples_amount = get_dimensions_of_ubyte_dataset(source, 1)[0];

    todo!();

    data
}
