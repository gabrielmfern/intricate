kernel void compute_loss(
    global float* output_samples,
    global float* expected_output_samples,

    global float* sample_losses,

    int outputs_amount
) {
    int sample_index = get_global_id(0);
    int samples_amount = get_global_size(0);

    float sample_loss = 0.0;

    for (int output_index = 0; output_index < outputs_amount; output_index++) {
        int flat_i = sample_index * outputs_amount + output_index;
        float output_dist = (float) (output_samples[flat_i] - expected_output_samples[flat_i]);
        sample_loss += output_dist * output_dist;
    }

    sample_losses[sample_index] = sample_loss;
}

kernel void compute_loss_to_output_derivatives(
    global float* output_samples,
    global float* expected_output_samples,

    global float* loss_to_output_derivatives
) {
    int sample_index = get_global_id(0);
    int samples_amount = get_global_size(0);

    int output_index = get_global_id(1);
    int outputs_amount = get_global_size(1);

    int flat_i = sample_index * outputs_amount + output_index;

    float dist = (float) (output_samples[flat_i] - expected_output_samples[flat_i]);

    loss_to_output_derivatives[flat_i] = 2 / (float)samples_amount * dist;
}
