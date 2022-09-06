kernel void calculate_exponentials(
    global float* inputs,
    global float* exponentials,

    global float* max_input_per_sample,

    int samples_amount,
    int numbers_amount
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }
    
    int input_index = get_global_id(1);

    if (input_index >= numbers_amount) {
        return;
    }

    int flat_input_i = sample_index * numbers_amount + input_index;
    float max_val = (float)max_input_per_sample[sample_index];

    // this -max_val is to normalize these values to not have NaN calculation results everywhere
    float dist = (float)inputs[flat_input_i] - max_val;

    exponentials[flat_input_i] = exp(dist);
}

kernel void sum_exponentials_per_sample(
    global float* exponentials,
    global float* exponential_sum_per_sample,

    int samples_amount,
    int numbers_amount
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    float exponential_sum = 0.0f;
    int row_part = sample_index * numbers_amount;
    for (int input_index = 0; input_index < numbers_amount; input_index++) {
        int flat_i = row_part + input_index;

        exponential_sum += exponentials[flat_i];
    }

    exponential_sum_per_sample[sample_index] = exponential_sum;
}

kernel void calculate_max_input_per_sample(
    global float* inputs,
    global float* max_input_per_sample,

    int samples_amount,
    int numbers_amount
) {
    int sample_index = get_global_id(0);
    
    if (sample_index >= samples_amount) {
        return;
    }
    if (numbers_amount == 0) {
        return;
    }

    int row_part = sample_index * numbers_amount;
    float max_input = (float)inputs[row_part];
    // printf("%e\n", max_input);
    for (int input_index = 1; input_index < numbers_amount; input_index++) {
        int flat_i = row_part + input_index;
        if ((float)inputs[flat_i] > max_input) {
            max_input = (float)inputs[flat_i];
        }
    }
    // printf("%e\n", max_input);

    max_input_per_sample[sample_index] = max_input;
}

kernel void propagate(
    global float* exponentials,
    global float* flattened_output_samples,
    global float* exponentials_sum_per_sample,

    int numbers_amount,
    int samples_amount
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    int input_index = get_global_id(1);

    if (input_index >= numbers_amount) {
        return;
    }

    int flat_i = sample_index * numbers_amount + input_index;
    float exponentials_sum = (float)exponentials_sum_per_sample[sample_index];

    flattened_output_samples[flat_i] = exponentials[flat_i] / exponentials_sum;
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
    float input_associated_output = (float)flattened_output_samples[flat_input_i];

    int row_part = sample_index * outputs_amount;
    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_output_i = row_part + output_index;

        float output = (float)flattened_output_samples[flat_output_i];
        float output_to_input_derivative = 0.0f;
        if (input_index == output_index) {
            output_to_input_derivative = output * ((float)1.0f - output);
        } else {
            output_to_input_derivative = -input_associated_output * output;
        }

        float loss_to_output_derivative = (float)flattened_loss_to_output_derivatives[flat_output_i];

        total += output_to_input_derivative * (float)loss_to_output_derivative;
    }

    flattened_loss_to_input_derivatives[flat_input_i] = total;
}