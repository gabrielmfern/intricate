kernel void weights_gradient_calculation(
    global float* flattened_output_to_loss_derivatives,
    global float* flattened_input_samples,

    global float* flattened_gradients,

    int samples_amount,
    int outputs_amount,
    int inputs_amount
) {
    int input_index = get_global_id(0);

    int output_index = get_global_id(1);

    if (input_index >= inputs_amount) {
        return;
    }
    if (output_index >= outputs_amount) {
        return;
    }

    int flat_weight_i = input_index * outputs_amount + output_index;

    float weight_gradient_contributions = (float)0.0;
    float f_samples_amount = (float)samples_amount;

    for (int sample_index = 0; sample_index < samples_amount; sample_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;
        int flat_input_i = sample_index * inputs_amount + input_index;

        float loss_to_output_derivative = (float)flattened_output_to_loss_derivatives[flat_output_i];
        float input = (float)flattened_input_samples[flat_input_i];

        weight_gradient_contributions += loss_to_output_derivative * input;
    }

    // should this be averaged among the samples?
    flattened_gradients[flat_weight_i] = weight_gradient_contributions / f_samples_amount;
}

kernel void bias_gradient_calculation(
    global float* flattened_output_to_loss_derivatives,

    global float* gradients,

    int samples_amount,
    int outputs_amount
) {
    int output_index = get_global_id(0);

    if (output_index >= outputs_amount) {
        return;
    }

    float bias_gradient = (float)0.0;

    for (int sample_index = 0; sample_index < samples_amount; sample_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;

        bias_gradient += (float)flattened_output_to_loss_derivatives[flat_output_i];
    }

    gradients[output_index] = bias_gradient / (float)samples_amount;
}

kernel void compute_loss_derivative_with_respect_to_inputs(
    global float* flattened_weights,
    global float* flattened_loss_to_output_derivatives,

    global float* flattened_loss_to_input_derivatives,

    int samples_amount,
    int outputs_amount,
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

    float loss_to_input_derivative = (float)0.0;

    int weight_row_part = input_index * outputs_amount;
    int output_row_part = sample_index * outputs_amount;

    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_weight_i = weight_row_part + output_index;
        int flat_output_i = output_row_part + output_index;

        float weight = (float)flattened_weights[flat_weight_i];
        float derivative = (float)flattened_loss_to_output_derivatives[flat_output_i];

        loss_to_input_derivative += weight * derivative;
    }

    int flat_input_i = sample_index * inputs_amount + input_index;

    flattened_loss_to_input_derivatives[flat_input_i] = (float) loss_to_input_derivative;
}