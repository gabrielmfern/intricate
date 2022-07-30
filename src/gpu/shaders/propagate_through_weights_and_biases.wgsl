@group(0)
@binding(0)
var<storage, read_write> flattened_sample_outputs: array<f32>;

@group(0)
@binding(1)
var<storage> flattened_sample_inputs: array<f32>;

@group(0)
@binding(2)
var<storage> flattened_weights: array<f32>;

@group(0)
@binding(3)
var<storage> biases: array<f32>;

@group(0)
@binding(4)
var<uniform> outputs_amount: u32;

@group(0)
@binding(5)
var<uniform> inputs_amount: u32;

@group(0)
@binding(6)
var<uniform> samples_amount: u32;

fn calculate_output_sample_for_all_inputs(sample_index: u32, output_index: u32) -> f32 {
    var output_accum: f32 = 0.0;
    for (var input_index: u32 = 0u; input_index < inputs_amount; input_index++) {
        var flattened_input_index: u32 = inputs_amount * sample_index + input_index;
        var flattened_weight_index: u32 = outputs_amount * input_index + output_index;

        var input: f32 = flattened_sample_inputs[flattened_input_index];
        var weight: f32 = flattened_weights[flattened_weight_index];
        output_accum += input * weight;
    }
    return output_accum;
}

@compute
@workgroup_size(250)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var sample_index = global_id.x;
    var output_index = global_id.y;

    var flattened_output_index = outputs_amount * sample_index + output_index;
    flattened_sample_outputs[flattened_output_index] = calculate_output_sample_for_all_inputs(sample_index, output_index);
}
