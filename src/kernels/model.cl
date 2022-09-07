kernel void compute_accuracy_per_output(
    global float *outputs,
    global float *expected_outputs,

    global float *accuracies,

    int count
) {
    int index = get_global_id(0);

    if (index >= count) {
        return;
    }

    float expected_output = (float)expected_outputs[index];
    float output = (float)outputs[index];
    accuracies[index] = 1.0f - fabs(output - expected_output) / fmax(expected_output, output);
}