#[allow(unused_imports)]
use crate::layers::{
    activations::sigmoid::Sigmoid,
    Layer
};
#[allow(unused_imports)]
use crate::utils::approx_eq::assert_approx_equal;

#[test]
fn should_return_correct_value() {
    let mut tanh_layer = Sigmoid::new();
    let inputs: Vec<f32> = Vec::from([100.3]);

    let expected_outputs: Vec<f32> = Vec::from([
        1.0 / (1.0 + std::f32::consts::E.powf(-100.3))
    ]);

    let actual_sample_outputs = tanh_layer
        .propagate(&Vec::from([inputs]));

    assert_approx_equal(&actual_sample_outputs[0], &expected_outputs, 2);
}
