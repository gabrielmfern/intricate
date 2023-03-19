use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::utils::approx_eq::{assert_approx_equal, assert_approx_equal_distance};

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
    let expected_results = vec![21.0, 15.0, 10.0, 6.0, 3.0, 1.0, 0.0];

    let buff = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &opencl_state)
        .unwrap();

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width).unwrap();
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

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width).unwrap();
    let actual_results = Vec::from_buffer(&buf_actual_results, true, &opencl_state).unwrap();

    assert_approx_equal(&actual_results, &expected_results, 2);
}

#[test]
fn should_sum_random_buffers_per_row_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();

    let mut rng = thread_rng();
    let width = rng.gen_range(451..1324);
    let height = rng.gen_range(123..1412);

    let test_vec: Vec<Vec<f32>> = (0..height)
        .map(|_| (0..width).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let expected_results: Vec<f32> = test_vec.iter().map(|row| row.iter().sum::<f32>()).collect();

    let buf = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &state)
        .unwrap();
    let actual_results_buf = buf.sum_2d_per_row(&state, width).unwrap();
    let actual_results = Vec::from_buffer(&actual_results_buf, false, &state).unwrap();

    assert_eq!(
        actual_results.len(),
        expected_results.len(),
        "The sizes of the results do not even match"
    );
    assert_approx_equal_distance(&actual_results, &expected_results, 0.1);
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
    let expected_results = vec![28.0, 21.0, 15.0, 10.0, 6.0, 3.0, 1.0, 0.0];

    let buff = test_vec
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f32>>()
        .to_buffer(false, &opencl_state)
        .unwrap();

    let buf_actual_results = buff.sum_2d_per_row(&opencl_state, width).unwrap();
    let actual_results = Vec::from_buffer(&buf_actual_results, true, &opencl_state).unwrap();

    dbg!(&actual_results);
    dbg!(&expected_results);
    assert_approx_equal(&actual_results, &expected_results, 1);
}

#[test]
fn should_compute_fft_1d_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let input = vec![
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,

        0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0
    ]
    .to_buffer(false, &state)
    .unwrap();
    let expected_fft = vec![
        12.        ,0.        ,  0.        ,0.        ,
         0.        ,0.        ,  0.        ,0.        ,
        -4.        ,0.        ,  0.        ,0.        ,
         0.        ,0.        ,  0.        ,0.        ,

         6.        , 0.        , -0.70710678, -0.29289322,
        -1.        ,-1.        ,  0.70710678, 1.70710678,
        -4.        , 0.        ,  0.70710678,-1.70710678,
        -1.        , 1.        , -0.70710678, 0.29289322
    ];
    let actual_fft = Vec::from_buffer(&input.fft(&state, 2).unwrap(), false, &state).unwrap();

    assert_approx_equal_distance(&expected_fft, &actual_fft, 0.1);
}

#[test]
fn should_compute_ifft_1d_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let input = vec![
        12.        ,0.        ,  0.        ,0.        ,
         0.        ,0.        ,  0.        ,0.        ,
        -4.        ,0.        ,  0.        ,0.        ,
         0.        ,0.        ,  0.        ,0.        ,

         6.        , 0.        , -0.70710678, -0.29289322,
        -1.        ,-1.        ,  0.70710678, 1.70710678,
        -4.        , 0.        ,  0.70710678,-1.70710678,
        -1.        , 1.        , -0.70710678, 0.29289322
    ]
    .to_buffer(false, &state)
    .unwrap();
    let expected_ifft = vec![
        1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,

        0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0
    ];
    let actual_ifft = Vec::from_buffer(
        &input.ifft(&state, 2).unwrap()
            .real_part(&state).unwrap(),
        false, 
        &state
    ).unwrap();

    assert_approx_equal_distance(&expected_ifft, &dbg!(actual_ifft), 0.1);
}

#[test]
fn should_tranpose_images_correclty() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let input = vec![
        0.4, 0.0, 0.5, 2.0, 0.9, 0.0, 0.1, 0.0,
        0.6, 0.0, 0.1, 0.0, 4.9, 0.0, 1.1, 0.0,
        3.4, 0.0, 7.3, 0.0, 1.9, 0.0, 0.2, 0.0,

        1.3, 0.0, 2.2, 0.0, 6.6, 0.0, 1.2, 0.0,
        0.2, 0.0, 0.5, 0.0, 9.1, 0.0, 4.4, 0.0,
        3.8, 0.0, 4.4, 0.0, 2.3, 0.0, 3.1, 0.0
    ]
    .to_buffer(false, &state)
    .unwrap();
    let expected_transpose = vec![
        0.4, 0.0, 0.6, 0.0, 3.4, 0.0,
        0.5, 2.0, 0.1, 0.0, 7.3, 0.0,
        0.9, 0.0, 4.9, 0.0, 1.9, 0.0,
        0.1, 0.0, 1.1, 0.0, 0.2, 0.0,

        1.3, 0.0, 0.2, 0.0, 3.8, 0.0,
        2.2, 0.0, 0.5, 0.0, 4.4, 0.0,
        6.6, 0.0, 9.1, 0.0, 2.3, 0.0,
        1.2, 0.0, 4.4, 0.0, 3.1, 0.0,
    ];
    let actual_transpose = Vec::from_buffer(
        &input.complex_tranpose(&state, 4, 3).unwrap(), 
        false, 
        &state
    ).unwrap();

    assert_approx_equal_distance(&expected_transpose, &actual_transpose, 0.1);
}

#[test]
fn should_padd_2d_buffer_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let matrix = vec![
        2.0, 8.2, 3.0,
        5.0, 3.0, 1.0,
        3.0, 2.0, 2.4,

        1.0, 2.0, 3.0,
        7.0, 3.0, 4.0,
        9.0, 6.0, 5.0,
    ]
    .to_buffer(false, &state)
    .unwrap();

    let padded_matrix = Vec::from_buffer(
        &matrix.padd_2d(3, 3, 5, 5, &state).unwrap(),
        false,
        &state
    ).unwrap();

    let expected_padded_matrix = vec![
        2.0, 8.2, 3.0, 0.0, 0.0,
        5.0, 3.0, 1.0, 0.0, 0.0,
        3.0, 2.0, 2.4, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,

        1.0, 2.0, 3.0, 0.0, 0.0,
        7.0, 3.0, 4.0, 0.0, 0.0,
        9.0, 6.0, 5.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    assert_approx_equal_distance(&expected_padded_matrix, dbg!(&padded_matrix), 0.0);
}

#[test]
fn should_slice_2d_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let matrix = vec![
        2.0, 8.2, 3.0,
        5.0, 3.0, 1.0,
        3.0, 2.0, 2.4,

        1.0, 2.0, 3.0,
        7.0, 3.0, 4.0,
        9.0, 6.0, 5.0,
    ]
    .to_buffer(false, &state)
    .unwrap();

    let sliced_matrix = Vec::from_buffer(
        &matrix.slice_2d(1..2, 0..2, 3, 3, &state).unwrap(),
        false,
        &state
    ).unwrap();

    let expected_sliced_matrix = vec![
        8.2, 3.0,
        3.0, 1.0,
        2.0, 2.4,

        2.0, 3.0,
        3.0, 4.0,
        6.0, 5.0
    ];

    assert_approx_equal_distance(&expected_sliced_matrix, dbg!(&sliced_matrix), 0.0);
}

#[test]
fn should_compute_complex_multiplication_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let first_matrix = vec![
        0.5, 0.04,  0.8, 0.13,   0.9, 0.41,
        0.1, 0.21,  0.3, 0.41,  0.34, 0.93,

        0.2, 0.21,  0.3, 0.32,  0.51, 0.93,
        0.4, 0.83,  0.8, 0.85,  0.83, 0.12,
    ].to_buffer(false, &state).unwrap();
    let second_matrix = vec![
        0.5, 0.04,  0.8, 0.13,   0.9, 0.41,
        0.1, 0.21,  0.3, 0.41,  0.34, 0.93,
    ].to_buffer(false, &state).unwrap();
    let result = Vec::from_buffer(
        &first_matrix.complex_multiply(1, 2, &second_matrix, &state).unwrap(), 
        false, 
        &state
    ).unwrap();
    let expected_result = vec![
         0.2484, 0.04,    0.6231, 0.208,   0.6419, 0.738,
        -0.0341, 0.042,  -0.0781, 0.246,  -0.7493, 0.6324,

         0.0916, 0.113,   0.1984, 0.295,   0.0777, 1.0461,
        -0.1343, 0.167,  -0.1085, 0.583,   0.1706, 0.8127
    ];

    assert_approx_equal_distance(&expected_result, &dbg!(result), 0.01);
}