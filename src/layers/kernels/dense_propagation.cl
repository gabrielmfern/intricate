kernel void dense_propagate(
    global float* flattened_input_samples,
    global float* biases,
    global float* flattened_weights,
    
    global float* flattened_output_samples,

    int inputs_amount,
    int samples_amount,
    int outputs_amount
) {
    int sample_index = get_global_id(0);
    int output_index = get_global_id(1);

    if (sample_index >= samples_amount) {
        return;
    }
    if (output_index >= outputs_amount) {
        return;
    }

    int flattened_output_index = sample_index * outputs_amount + output_index;

    float bias = (float)biases[output_index];
    float output = bias;

    int input_row_part = sample_index * inputs_amount;
    for (int input_index = 0; input_index < inputs_amount; input_index++) {
        int flattened_input_index = input_row_part + input_index;
        int flattened_weight_index = input_index * outputs_amount + output_index;

        float input = flattened_input_samples[flattened_input_index];
        float weight = flattened_weights[flattened_weight_index];

        output += (float) (input * weight);
    }

    flattened_output_samples[flattened_output_index] = output;
}