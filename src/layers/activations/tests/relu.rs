use crate::layers::activations::activation::ActivationLayerF64;
#[allow(unused_imports)]
use crate::layers::activations::relu::ReLUF64;
use crate::layers::layer::Layer;

#[allow(dead_code)]
async fn test_should_be_0_when_x_is_negative() {
    let x: Vec<f64> = Vec::from([-30.0, -40.0, -1.0, -0.3, -0.99]);

    let mut activation_layer = ReLUF64::new();
    let actual_outputs = activation_layer
        .propagate(&Vec::from([x]), &None, &None)
        .await;

    let expected_outputs = Vec::from([Vec::from([0.0, 0.0, 0.0, 0.0, 0.0])]);

    assert_eq!(actual_outputs, expected_outputs);
}

#[test]
fn should_be_0_when_x_is_negative() {
    pollster::block_on(test_should_be_0_when_x_is_negative());
}

#[allow(dead_code)]
async fn test_should_be_x_when_x_is_larger_than_negative_one() {
    let x: Vec<f64> = Vec::from([-30.0, 40.0, 21.0, -0.3, -0.99]);

    let mut activation_layer = ReLUF64::new();
    let actual_outputs = activation_layer
        .propagate(&Vec::from([x]), &None, &None)
        .await;

    let expected_outputs = Vec::from([Vec::from([0.0, 40.0, 21.0, 0.0, 0.0])]);

    assert_eq!(actual_outputs, expected_outputs);
}

#[test]
fn should_be_x_when_x_is_larger_than_negative_one() {
    pollster::block_on(test_should_be_x_when_x_is_larger_than_negative_one());
}

#[allow(dead_code)]
async fn test_differential_should_return_correct_value() {
    let x: Vec<f64> = Vec::from([-30.0, 40.0, 21.0, -0.3, -0.99]);

    let mut activation_layer = ReLUF64::new();
    activation_layer
        .propagate(&Vec::from([x.to_vec()]), &None, &None)
        .await;

    let output_index = 3;

    let expected_derivatives = Vec::from([0.0, 1.0, 1.0, 0.0, 0.0]);

    for (i, _) in (&x).iter().enumerate() {
        let actual_differential =
            activation_layer.differential_of_output_with_respect_to_input(0, i, output_index);

        assert_eq!(actual_differential, expected_derivatives[i]);
    }
}

#[test]
fn differential_should_return_correct_value() {
    pollster::block_on(test_differential_should_return_correct_value());
}
