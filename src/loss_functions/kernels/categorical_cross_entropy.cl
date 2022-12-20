kernel void compute_loss(
    global float* output_samples,
    global float* expected_output_samples,

    global float* sample_losses,

    int outputs_amount,
    int samples_amount
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    float sample_loss = 0.0f;

    int row_part = sample_index * outputs_amount;
    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_i = row_part + output_index;
        float output = (float) output_samples[flat_i];
        float expected_output = (float) expected_output_samples[flat_i];
        output = min(max(output, 0.0000001f), 0.9999999f);
        sample_loss -= expected_output * log(output);
            /* + (1.0f - expected_output) * log(1.0f - output); */
    }

    sample_losses[sample_index] = sample_loss;
}

kernel void compute_loss_to_output_derivatives(
    global float* output_samples,
    global float* expected_output_samples,

    global float* loss_to_output_derivatives,

    int samples_amount,
    int outputs_amount
) {
    int sample_index = get_global_id(0);

    int output_index = get_global_id(1);

    if (sample_index >= samples_amount) {
        return;
    }
    
    if (output_index > outputs_amount) {
        return;
    }

    int flat_i = sample_index * outputs_amount + output_index;

    float output = (float) output_samples[flat_i];
    float expected_output = (float) expected_output_samples[flat_i];
    output = min(max(output, 0.0000001f), 0.9999999f);

    loss_to_output_derivatives[flat_i] = -expected_output / output
        + (1.0f - expected_output) / (1.0f - output);
}

kernel void normalize_outputs(
    global float* outputs,
    global float* per_sample_total_sum,
    global float* normalized_outputs,

    int samples_amount,
    int outputs_amount
) {
    int sample_index = get_global_id(0);
    if (sample_index >= samples_amount) {
        return;
    }

    int output_index = get_global_id(1);
    if (output_index >= outputs_amount) {
        return;
    }

    int global_output_index = sample_index * outputs_amount + output_index;

    normalized_outputs[global_output_index] = outputs[global_output_index] / per_sample_total_sum[sample_index];
}