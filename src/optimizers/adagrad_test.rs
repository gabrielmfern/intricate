use super::Adagrad;
use crate::{utils::opencl::*, optimizers::Optimizer};
use rand::prelude::*;

#[test]
fn should_compute_update_vectors_correctly() {
    let mut rng = thread_rng();

    let gradients = vec![rng.gen_range(0f32..1f32)];

    let episilon = 0.000_000_01;
    let learning_rate = 0.01;

    let expected_first_update_vector = vec![learning_rate * gradients[0]];
    let expected_second_update_vector =
        vec![learning_rate / (gradients[0] + episilon) * gradients[0]];
    let expected_third_update_vector = vec![
        learning_rate / ((gradients[0].powf(2.0) + gradients[0].powf(2.0)).sqrt() + episilon)
            * gradients[0],
    ];

    let state = setup_opencl(DeviceType::GPU).unwrap();

    let gradients_buf = gradients
        .to_buffer(false, &state)
        .unwrap();

    let mut optimizer = Adagrad::new(learning_rate, episilon);
    optimizer.init(&state).unwrap();

    let first_update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();
    let second_update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();
    let third_update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();

    let first_update_vector =
        Vec::from_buffer(&first_update_buf, false, &state).unwrap();
    let second_update_vector =
        Vec::from_buffer(&second_update_buf, false, &state).unwrap();
    let third_update_vector =
        Vec::from_buffer(&third_update_buf, false, &state).unwrap();

    assert!(
        (dbg!(first_update_vector[0]) - dbg!(expected_first_update_vector[0])).abs()
            / expected_first_update_vector[0]
            <= 0.001
    );
    assert!(
        (dbg!(second_update_vector[0]) - dbg!(expected_second_update_vector[0])).abs()
            / expected_second_update_vector[0]
            <= 0.001
    );
    assert!(
        (dbg!(third_update_vector[0]) - dbg!(expected_third_update_vector[0])).abs()
            / expected_third_update_vector[0]
            <= 0.001
    );
}