#[allow(unused_imports)]
use std::f64::consts::E;
use crate::layers::activations::softmax::SoftMaxF64;
use crate::layers::layer::Layer;

// sometimes fails even though the values are the same
// because of floating-point numbers non-exactness
#[allow(dead_code)]
async fn test() {
    let inputs = Vec::from([300.1, 20.0, 5.2, 213.3]);
    let total_sum = E.powf(300.1) + E.powf(20.0) + E.powf(5.2) + E.powf(213.3);
    let expected_outputs = Vec::from([
        E.powf(300.1) / total_sum,
        E.powf(20.0) / total_sum,
        E.powf(5.2) / total_sum,
        E.powf(213.3) / total_sum,
    ]);

    let input_samples = Vec::from([inputs]);
    let expected_output_samples = Vec::from([expected_outputs]);

    let mut activation_layer = SoftMaxF64::new();
    let actual_outputs = activation_layer.propagate(&input_samples, &None, &None).await;

    assert_eq!(actual_outputs, expected_output_samples);
}

#[test]
fn should_return_correct_value() {
    pollster::block_on(test());
}
