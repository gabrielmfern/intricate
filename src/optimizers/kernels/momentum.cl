kernel void compute_update_vector(
    global float* gradients,
    global float* last_update_vector, // starts with all zeros
    global float* update_vector,

    float momentum_gamma,
    float learning_rate
) {
    int i = get_global_id(0);

    update_vector[i] = gradients[i] * learning_rate + last_update_vector[i] * momentum_gamma;
    last_update_vector[i] = update_vector[i];
}