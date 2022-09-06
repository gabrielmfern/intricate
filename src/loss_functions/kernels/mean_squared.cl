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
        float output_dist = (float) (output_samples[flat_i] - expected_output_samples[flat_i]);
        sample_loss += output_dist * output_dist;
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
    if (output_index >= outputs_amount) {
        return;
    }

    int flat_i = sample_index * outputs_amount + output_index;

    float dist = (float) (output_samples[flat_i] - expected_output_samples[flat_i]);

    loss_to_output_derivatives[flat_i] = 2.0f / (float)outputs_amount * dist;
}