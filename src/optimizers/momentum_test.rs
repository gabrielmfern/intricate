use rand::prelude::*;

use crate::{
    optimizers::Optimizer,
    utils::{
        opencl::{BufferLike, DeviceType},
        setup_opencl,
    },
};

use super::Momentum;

#[test]
fn should_compute_update_vectors_correctly() {
    // theta = theta - v_t
    // v_t = gamma * v_(t-1) + learning_rate * gradients of theta with respect to the loss
    let mut rng = thread_rng();

    let gradients = vec![rng.gen_range(0f32..1f32)];

    let gamma = 0.9;
    let learning_rate = 0.01;

    let expected_inital_update_vector = vec![learning_rate * gradients[0]];
    let expected_second_update_vector =
        vec![gamma * expected_inital_update_vector[0] + learning_rate * gradients[0]];

    let state = setup_opencl(DeviceType::GPU).unwrap();

    let gradients_buf = gradients.to_buffer(false, &state).unwrap();

    let mut optimizer = Momentum::new(learning_rate, gamma);
    optimizer.init(&state).unwrap();

    let initial_update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();
    let secondary_update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();

    let initial_update_vector =
        Vec::from_buffer(&initial_update_buf, false, &state).unwrap();
    let secondary_update_vector =
        Vec::from_buffer(&secondary_update_buf, false, &state).unwrap();

    assert!(
        (dbg!(initial_update_vector[0]) - dbg!(expected_inital_update_vector[0])).abs()
            / expected_inital_update_vector[0]
            <= 0.001
    );
    assert!(
        (dbg!(secondary_update_vector[0]) - dbg!(expected_second_update_vector[0])).abs()
            / expected_second_update_vector[0]
            <= 0.001
    );
}