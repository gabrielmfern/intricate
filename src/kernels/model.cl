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

    if (expected_outputs[index] == 0) {
        accuracies[index] = 1 - fabs(outputs[index] - expected_outputs[index]);
    } else {
        accuracies[index] = 1 - fabs(outputs[index] - expected_outputs[index]) / expected_outputs[index];
    }
    /* printf("accuracy %d: %e\n", index, accuracies[index]); */
}