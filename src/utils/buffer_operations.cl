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

    if (group_size > buffer_length) {
        group_size = buffer_length;
    }

    workgroup_state[local_id] = (float)original[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    int half_size = group_size / 2;
    while (group_size > 1) {
        // if the id in the work group is in the first half
        if (local_id < half_size) {
            // sum it and the corresponding value in the other half together into the local_id
            workgroup_state[local_id] += workgroup_state[local_id + half_size];
            if (local_id == 0) {
                if ((half_size * 2) < group_size) {
                    workgroup_state[0] = (float) (workgroup_state[0] + workgroup_state[group_size - 1]);
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

kernel void scale_inplace(
    global float *self,
    
    float scaler,
    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = (float)self[index] * scaler;
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

kernel void inverse_sqrt_inplace(
    global float *buf,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    buf[index] = 1 / sqrt(buf[index]);
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

    result[index] = 1 / sqrt(first[index]);
}

kernel void shift_inplace(
    global float *buf,

    float num,
    int size
) {
    int index = get_global_id(0);
    
    if (index >= size) {
        return;
    }

    buf[index] = buf[index] + num;
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

kernel void add_inplace(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] + other[index];
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

kernel void subtract_inplace(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] - other[index];
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

kernel void multiply_inplace(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] * other[index];
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

kernel void divide_inplace(
    global float *self,
    global float *other,

    int size
) {
    int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    self[index] = self[index] / other[index];
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