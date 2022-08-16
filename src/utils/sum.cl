// this kernel this to be repeatedly evaluated in a array
// about log(N) or ceil(log(N)) to get to the whole sum of the array 
// where N is the total amount of numbers
// and the log here is being taken with 
// a base that is the size of the local workgroups
kernel void sum_all_values_in_workgroups(
    global float* original,
    global float* reduced,

    local float* workgroup_state,

    int buffer_length
) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_size = get_local_size(0);

    workgroup_state[local_id] = original[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_size > buffer_length) {
        group_size = buffer_length;
    }

    int half_size = group_size / 2;
    while (half_size > 0) {
        // if the id in the work group is in the first half
        if (local_id < half_size) {
            if (global_id < buffer_length) {
                // sum it and the corresponding value in the other half together into the local_id
                workgroup_state[local_id] += workgroup_state[local_id + half_size];
                if (local_id == 0) {
                    if ((half_size * 2) < group_size) {
                        workgroup_state[0] += workgroup_state[group_size - 1];
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        group_size = half_size;
        half_size = group_size / 2;
    }


    if (local_id == 0) {
        // printf("%d-%d: %e\n", get_group_id(0), local_id, workgroup_state[local_id]);

        // after summing all of the items in the work group
        // should just take them and associate it with the sum of the
        // current workgroup in the reduced array
        reduced[get_group_id(0)] = workgroup_state[0];
    }
}

// TODO: write another kernel taht would reduce the size of the buffer 
// with some manual loop as to pass into the next iteration of the reduce and to be divisble by
// the max local workgroup size