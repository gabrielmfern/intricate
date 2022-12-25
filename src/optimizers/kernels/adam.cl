kernel void compute_update_vectors(
    global float* gradients,

    global float* last_moment_first_estimate,
    global float* current_moment_first_estimate,

    global float* last moment_second_estimate,
    global float* current_moment_second_estimate,

    global float* update_vector,

    float decay_rate_beta_1,
    float decay_rate_beta_2,
    float learning_rate_alpha,
    float timestep,
    float safety_epsilon
) {
    int i = get_global_id(0);

    current_moment_first_estimate[i] = gradients[i] * (1.0f - decay_rate_beta_1);
    if (last_moment_first_estimate != NULL) {
        current_moment_first_estimate[i] += last_moment_first_estimate[i] * decay_rate_beta_1
    }

    current_moment_second_estimate[i] = gradients[i] * gradients[i] * (1.0f - decay_rate_beta_2);
    if (last_moment_second_estimate != NULL) {
        current_moment_second_estimate[i] += last_moment_second_estimate[i] * decay_rate_beta_2;
    }

    float bias_corrected_moment_first_estimate = current_moment_first_estimate[i] 
        * 1.0f / (1.0f - pow(decay_rate_beta_1, timestep));
    float bias_corrected_moment_second_estimate = current_moment_second_estimate[i]
        * 1.0f / (1.0f - pow(decay_rate_beta_2, timestep));

    update_vector[i] = learning_rate_alpha * bias_corrected_moment_first_estimate 
        / (sqrt(bias_corrected_moment_second_estimate) + safety_epsilon);
}