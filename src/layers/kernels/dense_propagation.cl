kernel void dense_propagate(
    global float* flattened_input_samples,
    global float* biases,
    global float* flattened_weights,
    
    global float* flattened_output_samples,

    int inputs_amount
) {
    int sample_index = get_global_id(0);
    int samples_amount = get_global_size(0);

    int output_index = get_global_id(1);
    int outputs_amount = get_global_size(1);

    if (sample_index >= samples_amount) {
        return;
    }
    if (output_index >= outputs_amount) {
        return;
    }

    int flattened_output_index = samples_index * outputs_amount + output_index;

    float output = biases[output_index];

    for (int input_index = 0; input_index < inputs_amount; input_index++) {
        int flattened_input_index = sample_index * inputs_amount + input_index;
        int flattened_weight_index = input_index * outputs_amount + output_index;

        output += (float) (flattened_input_samples[flattened_input_index] * flattened_weights[flattened_weight_index);
    }

    flattened_output_samples[flattened_output_index] = output;
}