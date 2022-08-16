kernel void propagate(
    global float* flattened_input_samples,

    global float* flattened_output_samples,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    float input = (float) flattened_input_samples[index];
    flattened_output_samples[index] = max((float)0.0, (float)input);
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
    // int samples_amount = get_global_size(0);

    int input_index = get_global_id(1);
    // int inputs_amount = get_global_size(1);
    
    if (sample_index >= samples_amount) {
        return;
    }
    if (input_index >= inputs_amount) {
        return;
    }

    int flat_input_i = sample_index * inputs_amount + input_index;

    float total = (float)0.0; 

    float output = (float)flattened_output_samples[flat_input_i];

    float output_to_input_derivative = 0.0;
    if (output <= 0.0) {
        output_to_input_derivative = 0.0;
    } else {
        output_to_input_derivative = 1.0;
    }

    int row_part = sample_index * outputs_amount;
    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_output_i = row_part + output_index;

        float loss_to_output_derivative = (float)flattened_loss_to_output_derivatives[flat_output_i];

        total += (float)loss_to_output_derivative;
    }

    flattened_loss_to_input_derivatives[flat_input_i] = (float)output_to_input_derivative * total;
}