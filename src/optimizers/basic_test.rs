use rand::prelude::*;

use crate::{
    optimizers::Optimizer,
    utils::{
        opencl::{BufferLike, DeviceType},
        setup_opencl,
    },
};

use super::Basic;

#[test]
fn should_compute_update_vectors_correctly() {
    // theta = theta - v_t
    // v_t = learning_rate * gradients of theta with respect to the loss
    let mut rng = thread_rng();

    let gradients = vec![rng.gen_range(0f32..1f32)];

    let learning_rate = 0.01;

    let expected_update_vector = vec![learning_rate * gradients[0]];

    let state = setup_opencl(DeviceType::GPU).unwrap();

    let gradients_buf = gradients.to_buffer(false, &state).unwrap();

    let mut optimizer = Basic::new(learning_rate);
    optimizer.init(&state).unwrap();

    let update_buf = optimizer
        .compute_update_vectors(&gradients_buf, "parameter".to_string(), 0, 0)
        .unwrap();

    let update_vector = Vec::from_buffer(&update_buf, false, &state).unwrap();

    assert!(
        (dbg!(update_vector[0]) - dbg!(expected_update_vector[0])).abs()
            / expected_update_vector[0]
            <= 0.001
    );
}
