// this kernel has to be repeatedly evaluated in a buffer
// about log(N) or ceil(log(N)) to get to the whole sum of the values of the buffer 
// where N is the total count of this buffer
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

    if (global_id >= buffer_length) {
        return;
    }

    int group_size = get_local_size(0);

    if (group_size > buffer_length) {
        group_size = buffer_length;
    }

    workgroup_state[local_id] = original[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    int half_size = group_size / 2;
    while (group_size > 1) {
        // if the id in the work group is in the first half
        if (local_id < half_size) {
            // sum it and the corresponding value in the other half together into the local_id
            workgroup_state[local_id] += workgroup_state[local_id + half_size];
            if (local_id == 0) {
                if ((half_size * 2) < group_size) {
                    workgroup_state[0] += workgroup_state[group_size - 1];
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

kernel void sum_all_values_in_row_work_groups(
    global float* original,
    global float* reduced,

    local float* workgroup_state,

    int buffer_width,
    int buffer_height
) {
    int global_y = get_global_id(0);
    int global_x = get_global_id(1);

    if (global_y >= buffer_height) {
        return;
    }

    if (global_x >= buffer_width) {
        return;
    }

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    int group_size_y = get_local_size(0);
    int group_size_x = get_local_size(1);

    int x_group_id = get_group_id(1);

    if (group_size_y > buffer_height) {
        group_size_y = buffer_height;
    }

    if (group_size_x > buffer_width) {
        group_size_x = buffer_width;
    }

    if ((x_group_id + 1) * group_size_x > buffer_width) {
        group_size_x = buffer_width - x_group_id * group_size_x;
    }

    int local_id = local_y * group_size_x + local_x;
    int global_id = global_y * buffer_width + global_x;
    workgroup_state[local_id] = original[global_id];

    int half_size_x = group_size_x / 2;
    while (group_size_x > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_x < half_size_x) {
            /* printf("(local_id %d, global_id %d, global_x %d, group_size %d, buffer_width %d) %e + %e\n", local_id, global_id, global_x, group_size_x, buffer_width, workgroup_state[local_id], workgroup_state[local_id + half_size_x]); */
            workgroup_state[local_id] += workgroup_state[local_id + half_size_x];

            if (local_x == 0) {
                if ((half_size_x * 2) < group_size_x) {
                    int last_id = local_y * group_size_x + group_size_x - 1;
                    /* printf("(local_id %d, global_id %d, group_size %d, last_id %d) %e + %e\n", local_id, global_id, group_size_x, last_id, workgroup_state[local_id], workgroup_state[last_id]); */
                    workgroup_state[local_id] += workgroup_state[last_id];
                }
            }
        }

        group_size_x = half_size_x;
        half_size_x  = group_size_x / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_x == 0) {
        reduced[global_y * (get_global_size(1) / get_local_size(1) + buffer_width % 2) + x_group_id] = workgroup_state[local_id];
    }
}

kernel void scale(
    global float *nums,
    global float *result,
    
    float scaler,
    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    result[index] = (float)nums[index] * scaler;
}

kernel void squareroot(
    global float *first,
    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = sqrt(first[index]);
}

kernel void inverse_sqrt(
    global float *first,
    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = rsqrt(first[index]);
}

kernel void add_num(
    global float *first,

    global float *result,

    float num,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = first[index] + num;
}

kernel void add(
    global float *first,
    global float *second,

    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = first[index] + second[index];
}

kernel void subtract(
    global float *first,
    global float *second,

    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = first[index] - second[index];
}

kernel void multiply(
    global float *first,
    global float *second,

    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = first[index] * second[index];
}

kernel void divide(
    global float *first,
    global float *second,

    global float *result,

    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    result[index] = first[index] / second[index];
}