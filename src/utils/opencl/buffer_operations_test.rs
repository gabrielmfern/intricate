use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::utils::approx_eq::assert_approx_equal;

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
    let numbers_amount = 1001;
    let test_vec: Vec<f32> = (0..numbers_amount)
        .map(|_| -> f32 { rng.gen_range(-123.31_f32..3193.31_f32) })
        .collect();
    let expected_sum: f32 = test_vec.par_iter().sum();

    let buff = test_vec.to_buffer(true, &opencl_state).unwrap();

    let actual_result = buff.sum(&opencl_state).unwrap();

    println!("{} - {}", actual_result, expected_sum);
    assert!(((actual_result - expected_sum) / (actual_result.max(expected_sum))).abs() <= 0.0001);
}

#[test]
fn should_sum_buffers_width_wise_with_very_divisble_widths_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let width = 6;
    let test_vec = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let expected_results = vec![
        21.0, 
        15.0, 
        10.0, 
        6.0,
        3.0,
        1.0,
        0.0
    ];

    let buff = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &opencl_state)
        .unwrap();

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width)
        .unwrap();
    let actual_results = Vec::from_buffer(&buf_actual_results, true, &opencl_state).unwrap();

    dbg!(&actual_results);
    dbg!(&expected_results);
    assert_approx_equal(&actual_results, &expected_results, 2);
}

#[test]
fn should_sum_buffers_width_wise_with_very_large_heights() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let width = 10;
    let test_vec = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; 5000];
    let expected_results = vec![55.0; 5000];

    let buff = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &opencl_state)
        .unwrap();

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width)
        .unwrap();
    let actual_results = Vec::from_buffer(&buf_actual_results, true, &opencl_state).unwrap();

    assert_approx_equal(&actual_results, &expected_results, 2);
}

#[test]
fn should_sum_random_buffers_per_row_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let width = rng.gen_range(100..1024);
    let height = rng.gen_range(27..3132);

    let test_vec: Vec<Vec<f32>> = (0..height)
        .map(|_| {
            (0..width).map(|_| {
                rng.gen_range(-1231f32..4123f32)
                // 1.0
            }).collect()
        })
        .collect();
    let expected_results: Vec<f32> = test_vec.par_iter()
        .map(|row| row.iter().sum::<f32>())
        .collect();

    let buf = test_vec.iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &state).unwrap();
    let actual_results_buf = buf.sum_2d_per_row(&state, width).unwrap();
    let actual_results = Vec::from_buffer(&actual_results_buf, false, &state).unwrap();

    assert_eq!(actual_results.len(), expected_results.len(), "The sizes of the results do not even match");
    actual_results.iter().zip(&expected_results).for_each(|(actual, expected)| {
        assert!((actual - expected) / expected <= 0.5, "The values of the results are not at most 1% apart");
    });
}

#[test]
fn should_sum_buffers_width_wise_with_prime_widths_correctly() {
    let opencl_state = setup_opencl(DeviceType::GPU).unwrap();

    let width = 7;
    let test_vec = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let expected_results = vec![
        28.0,
        21.0,
        15.0,
        10.0,
        6.0,
        3.0,
        1.0,
        0.0
    ];

    let buff = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &opencl_state)
        .unwrap();

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width)
        .unwrap();
    let actual_results = Vec::from_buffer(&buf_actual_results, true, &opencl_state).unwrap();

    dbg!(&actual_results);
    dbg!(&expected_results);
    assert_approx_equal(&actual_results, &expected_results, 1);
}