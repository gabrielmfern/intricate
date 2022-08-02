@group(0)
@binding(0)
var<storage, read_write> flattened_input_to_error_derivatives: array<f32>;

@group(0)
@binding(1)
var<storage> flattened_output_to_error_derivatives: array<f32>;

@group(0)
@binding(2)
var<storage> flattened_layer_weights: array<f32>;

@group(0)
@binding(3)
var<uniform> samples_amount: u32;

@group(0)
@binding(4)
var<uniform> outputs_amount: u32;

@group(0)
@binding(5) 
var<uniform> inputs_amount: u32;

fn compute_input_to_error_derivative(sample_index: u32, input_index: u32) -> f32 {
    var all_outputs_contributions: f32 = 0.0;

    for (var output_index: u32 = 0u; output_index < outputs_amount; output_index++) {
        var flattened_weight_index: u32 = input_index * outputs_amount + output_index;
        var flattened_output_index: u32 = sample_index * outputs_amount + output_index;

        all_outputs_contributions += flattened_output_to_error_derivatives[flattened_output_index] * flattened_layer_weights[flattened_weight_index];
    }

    return all_outputs_contributions;
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var sample_index: u32 = global_id.x;
    var input_index: u32 = global_id.y;

    var flattened_input_index: u32 = sample_index * inputs_amount + input_index;

    flattened_input_to_error_derivatives[flattened_input_index] = compute_input_to_error_derivative(sample_index, input_index);
}