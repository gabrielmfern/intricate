@group(0)
@binding(0)
var<storage> flattened_layer_output_to_error_derivatives: array<f32>;

@group(0)
@binding(1)
var<storage> flattened_layer_inputs: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> flattened_layer_weights: array<f32>;

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

fn compute_sample_weight_gradient(sample_index: u32, input_index: u32, output_index: u32) -> f32 {
    var flattened_input_index: u32 = sample_index * inputs_amount + input_index;
    var flattened_output_to_error_derivative_index: u32 = sample_index * outputs_amount + output_index;

    var sample_input: f32 = flattened_layer_inputs[flattened_input_index];
    var sample_output_to_error_derivative: f32 = flattened_layer_output_to_error_derivatives[flattened_output_to_error_derivative_index];
    
    return sample_input * sample_output_to_error_derivative;
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var input_index: u32 = global_id.x;
    var output_index: u32 = global_id.y;

    var weight_gradient: f32 = 0.0;

    for (var sample_index: u32 = 0u; sample_index < samples_amount; sample_index++) {
        weight_gradient += compute_sample_weight_gradient(sample_index, input_index, output_index);
    }

    weight_gradient *= learning_rate / samples_amount_float;

    var flattened_weight_index: u32 = input_index * outputs_amount + output_index; 
    var old_weight: f32 = flattened_layer_weights[flattened_weight_index];

    flattened_layer_weights[flattened_weight_index] = old_weight + weight_gradient;
}