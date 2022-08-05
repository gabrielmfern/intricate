kernel void weights_gradient_application(
    global float* flattened_output_to_loss_derivatives,
    global float* flattened_input_samples,
    global float* flattened_weights,

    global float* flattened_new_weights,

    int samples_amount,
    float learning_rate
) {
    int input_index = get_global_id(0);
    int inputs_amount = get_global_size(0);

    int output_index = get_global_id(1);
    int outputs_amount = get_global_size(1);

    int flat_weight_i = input_index * outputs_amount + output_index;

    float weight_gradient = 0.0;

    for (int sample_index = 0; sample_index < samples_amount; sample_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;
        int flat_input_i = sample_index * inputs_amount + input_index;
        weight_gradient += flattened_output_to_loss_derivatives[flat_output_i] * flattened_input_samples[flat_input_i];
    }

    weight_gradient *= learning_rate / (float)samples_amount;

    flattened_new_weights[flat_weight_i] += flattened_weights[flat_weight_i] + weight_gradient;
}

kernel void bias_gradient_application(
    global float* flattened_output_to_loss_derivatives,
    global float* biases,

    global float* new_biases,

    int samples_amount,
    float learning_rate
) {
    int output_index = get_global_id(0);
    int outputs_amount = get_global_size(0);

    float bias_gradient = 0.0;

    for (int sample_index = 0; sample_index < samples_amount; sample_index++) {
        int flat_output_i = sample_index * outputs_amount + output_index;

        bias_gradient += flattened_output_to_loss_derivatives[flat_output_i];
    }

    bias_gradient *= learning_rate / (float)samples_amount;

    new_biases[output_index] = biases[output_index] + bias_gradient;
}

kernel void compute_loss_derivative_with_respect_to_inputs(
    global float* flattened_weights,
    global float* flattened_loss_to_output_derivatives,

    global float* flattened_loss_to_input_derivatives,

    int outputs_amount
) {
    int sample_index = get_global_id(0);
    int samples_amount = get_global_size(0);

    int input_index = get_global_id(1);
    int inputs_amount = get_global_size(1);

    float loss_to_input_derivative = 0.0;

    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_weight_i = input_index * outputs_amount + output_index;
        int flat_output_i = sample_index * outputs_amount + output_index;
        loss_to_input_derivative += flattened_weights[flat_weight_i] * flattened_loss_to_output_derivatives[flat_output_i];
    }

    int flat_input_i = sample_index * inputs_amount + input_index;

    flattened_loss_to_input_derivatives[flat_input_i] = loss_to_input_derivative;
}
