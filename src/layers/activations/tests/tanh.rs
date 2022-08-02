#[allow(unused_imports)]
use crate::layers::activations::tanh::TanH;
#[allow(unused_imports)]
use crate::layers::Layer;
#[allow(unused_imports)]
use crate::utils::approx_eq::assert_approx_equal;

#[test]
fn should_return_correct_value() {
    let mut tanh_layer = TanH::new();
    let inputs: Vec<f32> = Vec::from([100.3, 13.1, 14.3, 91.2]);

    let expected_outputs: Vec<f32> = Vec::from([
        inputs[0].tanh(),
        inputs[1].tanh(),
        inputs[2].tanh(),
        inputs[3].tanh(),
    ]);

    let actual_sample_outputs = tanh_layer
        .propagate(&Vec::from([inputs]));

    assert_approx_equal(&actual_sample_outputs[0], &expected_outputs, 2);
}
