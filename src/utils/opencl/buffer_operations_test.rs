use rand::{thread_rng, Rng};
use rayon::prelude::*;

use super::{
    opencl_state::{setup_opencl, DeviceType},
    BufferLike, BufferOperations,
};

#[test]
fn should_add_buffers_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 5123;

    let vec1: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let vec2: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a + b).collect();

    let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
    let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

    let actual = Vec::<f32>::from_buffer(
        &buff1.add(&buff2, &opencl_state).unwrap(),
        true,
        &opencl_state,
    )
    .unwrap();

    expected.iter().zip(actual).for_each(|(expected, actual)| {
        assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
    });
}

#[test]
fn should_subtract_buffers_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 5123;

    let vec1: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let vec2: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a - b).collect();

    let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
    let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

    let actual = Vec::<f32>::from_buffer(
        &buff1.subtract(&buff2, &opencl_state).unwrap(),
        true,
        &opencl_state,
    )
    .unwrap();

    expected.iter().zip(actual).for_each(|(expected, actual)| {
        assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
    });
}

#[test]
fn should_multiply_buffers_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 5123;

    let vec1: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-153_f32..141_f32) })
        .collect();
    let vec2: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-151_f32..121_f32) })
        .collect();
    let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a * b).collect();

    let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
    let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

    let actual = Vec::<f32>::from_buffer(
        &buff1.multiply(&buff2, &opencl_state).unwrap(),
        true,
        &opencl_state,
    )
    .unwrap();

    expected.iter().zip(actual).for_each(|(expected, actual)| {
        assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
    });
}

#[test]
fn should_divide_buffers_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 5123;

    let vec1: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let vec2: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();
    let expected: Vec<f32> = vec1.iter().zip(&vec2).map(|(a, b)| a / b).collect();

    let buff1 = vec1.to_buffer(true, &opencl_state).unwrap();
    let buff2 = vec2.to_buffer(true, &opencl_state).unwrap();

    let actual = Vec::<f32>::from_buffer(
        &buff1.divide(&buff2, &opencl_state).unwrap(),
        true,
        &opencl_state,
    )
    .unwrap();

    expected.iter().zip(actual).for_each(|(expected, actual)| {
        assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
    });
}

#[test]
fn should_scale_buffers_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 5123;

    let vec1: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-1513_f32..12341_f32) })
        .collect();

    let scaler = 0.123;
    let expected: Vec<f32> = vec1.iter().map(|a| a * scaler).collect();

    let buff = vec1.to_buffer(true, &opencl_state).unwrap();

    let actual = Vec::<f32>::from_buffer(
        &buff.scale(scaler, &opencl_state).unwrap(),
        true,
        &opencl_state,
    )
    .unwrap();

    expected.iter().zip(actual).for_each(|(expected, actual)| {
        assert!((expected - actual).abs() / expected.max(actual) <= 0.0001);
    });
}

#[test]
fn should_sum_buffer_to_correct_value() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let numbers_amount = 256;
    let test_vec: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-123.31_f32..3193.31_f32) })
        .collect();
    let expected_sum: f32 = test_vec.par_iter().sum();

    let buff = test_vec.to_buffer(true, &opencl_state).unwrap();

    let actual_result = buff.sum(&opencl_state).unwrap();

    println!("{} - {}", actual_result, expected_sum);
    assert!(((actual_result - expected_sum) / (actual_result.max(expected_sum))).abs() <= 0.0001);
}
