use crate::layers::activations::softmax::SoftMax;
use crate::layers::Layer;
#[allow(unused_imports)]
use std::f32::consts::E;

use crate::utils::approx_eq::assert_approx_equal_matrix;

#[allow(dead_code)]
async fn test() {
    let inputs = Vec::from([300.1, 20.0, 5.2, 213.3]);
    let total_sum = E.powf(300.1 - 300.1) + E.powf(20.0 - 300.1) + E.powf(5.2 - 300.1) + E.powf(213.3 - 300.1);
    let expected_outputs = Vec::from([
        E.powf(300.1 - 300.1) / total_sum,
        E.powf(20.0 - 300.1) / total_sum,
        E.powf(5.2 - 300.1) / total_sum,
        E.powf(213.3 - 300.1) / total_sum,
    ]);

    let input_samples = Vec::from([inputs]);
    let expected_output_samples = Vec::from([expected_outputs]);

    let mut activation_layer = SoftMax::new();
    let actual_outputs_samples = activation_layer
        .propagate(&input_samples, &None, &None)
        .await;

    assert_approx_equal_matrix(&actual_outputs_samples, &expected_output_samples, 5);
}

#[test]
fn should_return_correct_value() {
    pollster::block_on(test());
}
