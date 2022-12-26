kernel void compute_update_vector_and_update_gradient_history_summation(
    global float* gradients,
    global float* gradients_history_summation,
    global float* update_vector,

    int has_gradients_history_summation,

    float learning_rate,
    float epsilon
) {
    int i = get_global_id(0);

    float update_scalar = gradients[i] * learning_rate;
    if (has_gradients_history_summation == 1) {
        update_scalar *= rsqrt(gradients_history_summation[i] + epsilon);
    }

    update_vector[i] = update_scalar;

    gradients_history_summation[i] += gradients[i] * gradients[i];
}