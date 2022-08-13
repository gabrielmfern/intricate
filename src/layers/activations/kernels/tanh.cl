kernel void propagate(
    global float* flattened_input_samples,

    global float* flattened_output_samples,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    flattened_output_samples[index] = (float)tanh(flattened_input_samples[index]);
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

    // float output = (float)flattened_output_samples[flat_input_i];
    // float output_to_input_derivative = (float)1.0 - output * output;

    // float loss_to_output_derivative = (float)flattened_loss_to_output_derivatives[flat_input_i];
    float total = (float)0.0; 
    // output_to_input_derivative * loss_to_output_derivative;

    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;

        float output = (float)flattened_output_samples[flat_input_i];
        float output_to_input_derivative = (float)1.0 - output * output;
        float loss_to_output_derivative = (float)flattened_loss_to_output_derivatives[flat_output_i];

        total += (float)output_to_input_derivative * (float)loss_to_output_derivative;
        // printf("last + %e * %e = %e \n", output_to_input_derivative, loss_to_output_derivative, total);
    }

    flattened_loss_to_input_derivatives[flat_input_i] = total;
}