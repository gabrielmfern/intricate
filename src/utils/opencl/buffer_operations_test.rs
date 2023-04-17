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
    .unwrap()
    .to_complex_float2_buffer(&state)
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

    assert_approx_equal_distance(&expected_ifft, &dbg!(actual_ifft), 0.01);
}

#[test]
fn should_get_real_part() {
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
    let expected = vec![
        12.0, 0.0,
        0.0, 0.0,
        -4.0, 0.0,
        0.0, 0.0,

          6.0, -0.70710678,
         -1.0,  0.70710678,
         -4.0,  0.70710678,
         -1.0, -0.70710678
    ];
    let actual = Vec::from_buffer(
        &input.real_part(&state).unwrap(),
        false, 
        &state
    ).unwrap();

    assert_approx_equal_distance(&expected, &actual, 0.01);
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
        0.02201559, 0.08995634, 0.15729659, 0.94611638, 0.86306924, 0.48746292,
        0.80557588, 0.70359446, 0.56813448, 0.49895051, 0.18331707, 0.30356326,
        0.37590278, 0.92821253, 0.55003469, 0.94294651, 0.81038448, 0.71612575,

        0.62463144, 0.38389965, 0.76023669, 0.15724373, 0.63528774, 0.15250522,
        0.95931403, 0.0736696 , 0.14337976, 0.36089077, 0.70138251, 0.20881555,
        0.60740324, 0.68478843, 0.77183901, 0.0225153 , 0.28929563, 0.94464145
    ].to_buffer(false, &state).unwrap();
    let second_matrix = vec![
        0.01623191, 0.3336699 , 0.78501506, 0.85333468, 0.18285339, 0.22090835,
        0.3146594 , 0.58969164, 0.49100913, 0.61831977, 0.27888753, 0.13454833,
        0.46675664, 0.62190476, 0.8278547 , 0.27088854, 0.91279219, 0.23284111,

        0.55763434, 0.41348137, 0.63071945, 0.70323306, 0.05711824, 0.46770871,
        0.39529738, 0.4022816 , 0.80698926, 0.93042717, 0.32406534, 0.26143797,
        0.24111568, 0.7651777 , 0.44970323, 0.90303971, 0.59668867, 0.25291748,

        0.02764636, 0.20806934, 0.86502986, 0.50578618, 0.63498839, 0.0476984,
        0.64422009, 0.90617307, 0.52598484, 0.98243063, 0.64622625, 0.7719338,
        0.09444524, 0.73904437, 0.08984856, 0.39729492, 0.38062662, 0.21186757
    ].to_buffer(false, &state).unwrap();
    let result = Vec::from_buffer(
        &first_matrix.complex_multiply(3, 2, &second_matrix, &state).unwrap().dbg(6, &state).unwrap(), 
        false, 
        &state
    ).unwrap();
    let expected_result = vec![
        -0.02965837,0.0088061, -0.68387373,0.87694224, 0.05013051,0.27979345,
        -0.16142175,0.69643397, -0.02955174,0.59627804, 0.01028092,0.10932501,
        -0.40180468,0.66702509,  0.1999154 ,0.92962079, 0.57296911,0.84236482,

        -0.02491862,0.05926578, -0.5661303 ,0.70735017, -0.17869365,0.43150803,
        0.03539892,0.6021974 , -0.00575868,0.93125546, -0.01995625,0.14630037,
        -0.61961148,0.51143902, -0.60416576,0.92074925, 0.30242651,0.63226452,

        -0.01810851,0.00706773, -0.34246635,0.89797737, 0.52478774,0.35070031,
        -0.11861019,1.18326085, -0.19135414,0.82059312, -0.11586644,0.33767919,
        -0.65048802,0.36547409, -0.32520803,0.30324837, 0.15673008,0.44427071,


        -0.11795679,0.21465213,  0.46261573,0.77217503, 0.08247484,0.16822646,
        0.25841483,0.5888803, -0.15274513,0.2658552 , 0.16751105,0.1526059,
        -0.14236369,0.69737651,  0.63287141,0.22772174, 0.04411543,0.92962125,

        0.18958059,0.47234908,  0.36891708,0.63380026, -0.0350415 ,0.30584044,
        0.34957839,0.41503578, -0.22007665,0.4246394 , 0.17270145,0.2510379,
        -0.37753039,0.62988464,  0.32676628,0.70712648, -0.06629691,0.63682477,

        -0.06260896,0.14058008,  0.57809574,0.52053774, 0.39612608,0.12714125,
        0.55125196,0.91676398, -0.27913457,0.33068374, 0.29206001,0.67636296,
        -0.44872269,0.51357295,  0.06040341,0.30867069, -0.09002527,0.42084804
    ];

    assert_approx_equal_distance(&expected_result, &dbg!(result), 0.01);
}

#[test]
fn should_sample_complex_multiply_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let first_matrix = vec![
        0.02201559, 0.08995634, 0.15729659, 0.94611638, 0.86306924, 0.48746292,
        0.80557588, 0.70359446, 0.56813448, 0.49895051, 0.18331707, 0.30356326,
        0.37590278, 0.92821253, 0.55003469, 0.94294651, 0.81038448, 0.71612575,

        0.62463144, 0.38389965, 0.76023669, 0.15724373, 0.63528774, 0.15250522,
        0.95931403, 0.0736696 , 0.14337976, 0.36089077, 0.70138251, 0.20881555,
        0.60740324, 0.68478843, 0.77183901, 0.0225153 , 0.28929563, 0.94464145
    ].to_buffer(false, &state).unwrap();
    let second_matrix = vec![
        0.01623191, 0.3336699 , 0.78501506, 0.85333468, 0.18285339, 0.22090835,
        0.3146594 , 0.58969164, 0.49100913, 0.61831977, 0.27888753, 0.13454833,
        0.46675664, 0.62190476, 0.8278547 , 0.27088854, 0.91279219, 0.23284111,

        0.55763434, 0.41348137, 0.63071945, 0.70323306, 0.05711824, 0.46770871,
        0.39529738, 0.4022816 , 0.80698926, 0.93042717, 0.32406534, 0.26143797,
        0.24111568, 0.7651777 , 0.44970323, 0.90303971, 0.59668867, 0.25291748,

        0.02764636, 0.20806934, 0.86502986, 0.50578618, 0.63498839, 0.0476984,
        0.64422009, 0.90617307, 0.52598484, 0.98243063, 0.64622625, 0.7719338,
        0.09444524, 0.73904437, 0.08984856, 0.39729492, 0.38062662, 0.21186757,

        0.62463144, 0.38389965, 0.76023669, 0.15724373, 0.63528774, 0.15250522,
        0.95931403, 0.0736696 , 0.14337976, 0.36089077, 0.70138251, 0.20881555,
        0.60740324, 0.68478843, 0.77183901, 0.0225153 , 0.28929563, 0.94464145
    ].to_buffer(false, &state).unwrap();
    let result = Vec::from_buffer(
        &first_matrix.sampled_complex_pointwise_mutliply(&second_matrix, 3, 3, &state)
            .unwrap().dbg(6, &state).unwrap(), 
        false, 
        &state
    ).unwrap();
    let expected_result = vec![
        -0.02965837, 0.0088061,  -0.68387373, 0.87694224, 0.05013051, 0.27979345,
        -0.16142175, 0.69643397, -0.02955175, 0.59627804, 0.01028092, 0.10932501,
        -0.40180467, 0.66702509,  0.1999154 , 0.92962079, 0.57296911, 0.84236481,

        -0.02491862, 0.05926578, -0.5661303 , 0.70735017, -0.17869366, 0.43150802,
        0.03539893, 0.6021974,  -0.00575869, 0.93125546, -0.01995625, 0.14630037,
        -0.61961147, 0.51143902, -0.60416577, 0.92074926, 0.30242652, 0.63226452,

        -0.06260896, 0.14058008,  0.57809573, 0.52053773, 0.39612608, 0.12714125,
        0.55125196, 0.91676398, -0.27913457, 0.33068374, 0.29206001, 0.67636296,
        -0.44872269, 0.51357295,  0.06040341, 0.30867069, -0.09002527, 0.42084804,

        0.24278549, 0.47959158,  0.55323423, 0.23908491, 0.38033267, 0.19376939,
        0.9148562 , 0.14134456, -0.10968439, 0.10348886, 0.44833349, 0.29291915,
        -0.0999965 , 0.83188542,  0.59522852, 0.03475637, -0.80865551, 0.54656129,
    ];

    assert_approx_equal_distance(&expected_result, &result, 0.01);
}

#[test]
fn should_sampled_convolve_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let image = vec![
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,

        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    ].to_buffer(false, &state).unwrap();
    let filter_matrix = vec![
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,

        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,

        0.2, 0.2, 0.2,
        0.2, 0.2, 0.2,

        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
    ].to_buffer(false, &state).unwrap();
    let result = Vec::from_buffer(
        &image.sampled_convolve_2d(&state, &filter_matrix, 5, 4, 3, 2, (2..4, 1..3)).unwrap(), 
        false, 
        &state
    ).unwrap();
    let expected_result = vec![
        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,

        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,
        0.6, 0.6, 0.6,

        1.2, 1.2, 1.2,
        1.2, 1.2, 1.2,
        1.2, 1.2, 1.2,

        1.8, 1.8, 1.8,
        1.8, 1.8, 1.8,
        1.8, 1.8, 1.8,
    ];

    assert_approx_equal_distance(&expected_result, &dbg!(result), 0.01);
}

#[test]
fn should_flip_2d_correctly() {
    let state = setup_opencl(DeviceType::GPU).unwrap();
    let matrix = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
    ].to_buffer(false, &state).unwrap();
    let result = Vec::from_buffer(
        &matrix.flip_2d(5, 4, &state).unwrap(),
        false, 
        &state
    ).unwrap();
    let expected_result = vec![
        20.0, 19.0, 18.0, 17.0, 16.0,
        15.0, 14.0, 13.0, 12.0, 11.0,
        10.0, 9.0, 8.0, 7.0, 6.0,
        5.0, 4.0, 3.0, 2.0, 1.0
    ];

    assert_approx_equal_distance(&expected_result, &dbg!(result), 0.01);
}