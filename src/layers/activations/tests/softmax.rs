#[allow(unused_imports)]
use std::f32::consts::E;
#[allow(unused_imports)]
use crate::layers::activations::softmax::SoftMax;
#[allow(unused_imports)]
use crate::layers::Layer;
#[allow(unused_imports)]
use crate::utils::approx_eq::assert_approx_equal_matrix;

#[test]
fn should_return_correct_value() {
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
        .propagate(&input_samples);

    assert_approx_equal_matrix(&actual_outputs_samples, &expected_output_samples, 5);
}