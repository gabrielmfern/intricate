@group(0)
@binding(0)
var<storage, read_write> flattened_layer_weights: array<f32>;

@group(0)
@binding(1)
var<storage> flattened_layer_output_to_error_derivatives: array<f32>;

@group(0)
@binding(2)
var<storage> flattened_layer_inputs: array<f32>;

@group(0)
@binding(3)
var<uniform> learning_rate: f32;

@group(0)
@binding(4)
var<uniform> inputs_amount: u32;

@group(0)
@binding(5)
var<uniform> outputs_amount: u32;

@group(0)
@binding(6)
var<uniform> samples_amount: u32;

@group(0)
@binding(7)
var<uniform> samples_amount_float: f32;

fn calculate_weight_gradient_for_sample(input_index: u32, output_index: u32, sample_index: u32) -> f32 {
    var flattened_input_index: u32 = samples_amount * inputs_amount + input_index;
    var flattened_output_to_error_derivative_index: u32 = samples_amount * outputs_amount + output_index;

    var sample_input: f32 = flattened_layer_inputs[flattened_input_index];
    var sample_output_to_error_derivative: f32 = flattened_layer_output_to_error_derivatives[flattened_output_to_error_derivative_index];
    
    return sample_output_to_error_derivative * sample_input;
}

@compute
@workgroup_size(250)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var input_index: u32 = global_id.x;
    var output_index: u32 = global_id.y;

    var weight_gradient: f32 = 0.0;

    for (var sample_index: u32 = 0u; sample_index < samples_amount; sample_index++) {
        weight_gradient += calculate_weight_gradient_for_sample(input_index, output_index, sample_index);
    }

    weight_gradient *= learning_rate / samples_amount_float;

    flattened_layer_weights[input_index * outputs_amount + output_index] += weight_gradient;
}
