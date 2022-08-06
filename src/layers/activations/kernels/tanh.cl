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

    global float* flattened_loss_to_input_derivatives
) {
    int index = get_global_id(0);

    float val = flattened_output_samples[index];
    float output_to_input_derivative = 1 - val * val;

    flattened_loss_to_input_derivatives[index] = output_to_input_derivative * flattened_loss_to_output_derivatives[index];
}