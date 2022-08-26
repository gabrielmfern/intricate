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

    accuracies[index] = fabs(outputs[index] - expected_outputs[index]) / expected_outputs[index];
}
