kernel void propagate(
    global float* flattened_input_samples,

    global float* flattened_output_samples,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    flattened_output_samples[index] = tanh((float) flattened_input_samples[index]);
}

kernel void back_propagate(
    global float* flattened_loss_to_output_derivatives,
    global float* flattened_output_samples,

    global float* flattened_loss_to_input_derivatives,

    int outputs_amount,
    int samples_amount,
    int inputs_amount
) {
    int sample_index = get_global_id(0);
    int input_index = get_global_id(1);
    
    if (sample_index >= samples_amount) {
        return;
    }
    if (input_index >= inputs_amount) {
        return;
    }

    int flat_input_i = sample_index * inputs_amount + input_index;

    float total = 0.0f; 

    float output = (float)flattened_output_samples[flat_input_i];
    float output_to_input_derivative = 1.0f - output * output;

    int row_part = sample_index * outputs_amount;
    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_output_i = row_part + output_index;

        float loss_to_output_derivative = (float)flattened_loss_to_output_derivatives[flat_output_i];

        total += loss_to_output_derivative;
    }

    flattened_loss_to_input_derivatives[flat_input_i] = output_to_input_derivative * total;
}