kernel void propagate(
    global float* flattened_input_samples,

    global float* flattened_output_samples
) {
    int index = get_global_id(0);

    flattened_output_samples[index] = tanh(flattened_input_samples[index]);
}

kernel void back_propagate(
    global float* flattened_loss_to_output_derivatives,
    global float* flattened_output_samples,

    global float* flattened_loss_to_input_derivatives,

    int outputs_amount
) {
    int sample_index = get_global_id(0);
    int samples_amount = get_global_size(0);

    int input_index = get_global_id(1);
    int inputs_amount = get_global_size(1);

    int flat_input_i = sample_index * inputs_amount + input_index;

    float total = 0.0;

    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;

        float output = flattened_output_samples[flat_output_i];
        float output_to_input_derivative = 1 - output * output;

        total += output_to_input_derivative * flattened_loss_to_output_derivatives[flat_output_i];
    }

    flattened_loss_to_input_derivatives[flat_input_i] = total;
}