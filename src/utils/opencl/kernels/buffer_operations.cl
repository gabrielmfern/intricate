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

    int reduced_width,
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

    int local_id_0 = get_local_id(0);
    int local_x = get_local_id(1);

    int group_size_y = get_local_size(0);
    int group_size_x = get_local_size(1);

    // adjust the x_group_id to be the actual x_group_id in just the current row
    int x_group_id = get_group_id(1) % reduced_width;

    if (group_size_y > buffer_height) {
        group_size_y = buffer_height;
    }

    if (group_size_x > buffer_width) {
        group_size_x = buffer_width;
    }

    // adjust for the last summation group in the row
    if ((x_group_id + 1) * group_size_x > buffer_width) {
        group_size_x = buffer_width - x_group_id * group_size_x;
    }

    int local_id = local_id_0 * group_size_x + local_x;
    int global_id = global_y * buffer_width + global_x;
    workgroup_state[local_id] = original[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    int initial_group_size_x = group_size_x;
    int half_size_x = group_size_x / 2;
    while (group_size_x > 1) {
        if (local_x < half_size_x) {
            workgroup_state[local_id] += workgroup_state[local_id + half_size_x];

            if (local_x == 0) {
                if ((half_size_x * 2) < group_size_x) {
                    int last_id = local_id_0 * initial_group_size_x + group_size_x - 1;
                    workgroup_state[local_id] += workgroup_state[last_id];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        group_size_x = half_size_x;
        half_size_x  = group_size_x / 2;
    }

    if (local_x == 0) {
        reduced[global_y * reduced_width + x_group_id] = workgroup_state[local_id];
    }
}

uint reverse_bits(uint x, uint amount_bits) {
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16) >> (32u - amount_bits);
}

float2 cis(float theta) {
    return (float2) (cos(M_PI_F * theta), sin(M_PI_F * theta));
}

float2 complex_multiplication(float2 a, float2 b) {
    return (float2) (a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void padd_2d(
    global float *self,
    global float *result,

    uint old_width,
    uint old_height
) {
    uint matrix_y = get_global_id(0); // the get_global_size(0) should be the old_height + padding_y
    uint matrix_x = get_global_id(1); // the get_global_size(1) should be the old_width + padding_x
    uint sample_index = get_global_id(2);
    uint global_linear_id = sample_index * get_global_size(0) * get_global_size(1) 
        + matrix_y * get_global_size(1) 
        + matrix_x;

    if (matrix_x >= old_width
    || matrix_y >= old_height) {
        result[global_linear_id] = 0.0f;
    } else {
        uint old_global_linear_id = sample_index * old_width * old_height
            + matrix_y * old_width + matrix_x;
        result[global_linear_id] = self[old_global_linear_id];
    }
}

kernel void slice_2d(
    global float *self,
    global float *result,

    uint start_x,
    uint start_y,
    
    uint old_width,
    uint old_height
) {
    uint matrix_y = get_global_id(1); // the get_global_size(1) should be the endY - start_y
    uint matrix_x = get_global_id(2); // the get_global_size(2) should be the endX - start_x 
    uint sample_index = get_global_id(2);

    uint global_old_linear_id = sample_index * old_width * old_height 
        + (matrix_y + start_y) * old_width + matrix_x + start_x;
    uint global_linear_id = sample_index * get_global_size(0) * get_global_size(1) 
        + matrix_y * get_global_size(1) 
        + matrix_x;

    result[global_linear_id] = self[global_old_linear_id];
}

kernel void get_real_part(
    global float2 *self,
    global float *result
) {
    uint global_index = get_global_id(0);
    result[global_index] = self[global_index].x;
}

kernel void complex_point_wise_multiply(
    global float2 *self,
    global float2 *other,
    global float2 *result
) {
    uint i = get_global_id(0);
    result[i] = complex_multiplication(self[i], other[i]);
}

kernel void ifft(
    global float2 *nums,
    global float *result,
    global float2 *complex_result,
    uint N,
    uint logN
) {
    uint sample_index = get_global_id(0);
    uint signal_index = get_global_id(1);

    uint index = 2u * signal_index;

    uint initial_signal_index = sample_index * N;

    uint reverse = reverse_bits(index, logN);
    complex_result[initial_signal_index + index] = nums[initial_signal_index + reverse];

    index += 1u;
    reverse = reverse_bits(index, logN);
    complex_result[initial_signal_index + index] = nums[initial_signal_index + reverse];

    float fN = (float) N;

    for (uint s = 1u; s <= logN; s++) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        uint m_half = 1u << (s - 1u);

        uint k = signal_index / m_half * (1u << s);
        uint j = signal_index % m_half;
        uint k_plus_j = k + j;

        float2 twiddle = cis((float)j / (float)m_half);

        uint first_half_index = initial_signal_index + k_plus_j;
        uint second_half_index = first_half_index + m_half;

        float2 t = complex_multiplication(twiddle, complex_result[second_half_index]);
        float2 u = complex_result[first_half_index];
        complex_result[first_half_index] = u + t;
        complex_result[second_half_index] = u - t;
    }
}

kernel void fft(
    global float *nums,
    global float2 *result,
    uint N,
    uint logN
) {
    uint sample_index = get_global_id(0);
    uint signal_index = get_global_id(1);

    uint index = 2u * signal_index;

    uint initial_signal_index = sample_index * N;

    uint reverse = reverse_bits(index, logN);
    result[initial_signal_index + index] = (float2) (nums[initial_signal_index + reverse], 0.0);

    index += 1u;
    reverse = reverse_bits(index, logN);
    result[initial_signal_index + index] = (float2) (nums[initial_signal_index + reverse], 0.0);

    for (uint s = 1u; s <= logN; s++) {
        barrier(CLK_GLOBAL_MEM_FENCE);

        uint m_half = 1u << (s - 1u);

        uint k = signal_index / m_half * (1u << s);
        uint j = signal_index % m_half;
        uint k_plus_j = k + j;

        float2 twiddle = cis(-(float)j / (float)m_half);

        uint first_half_index = initial_signal_index + k_plus_j;
        uint second_half_index = first_half_index + m_half;

        float2 t = complex_multiplication(twiddle, result[second_half_index]);
        float2 u = result[first_half_index];
        result[first_half_index] = u + t;
        result[second_half_index] = u - t;
    }
}

kernel void complex_transpose(
    global float2 *nums,
    global float2 *result,
    uint height,
    uint width
) {
    uint sample_index = get_global_id(2);
    uint matrix_index_y = get_global_id(1);
    uint matrix_index_x = get_global_id(0);

    uint y = matrix_index_x;
    uint x = matrix_index_y;
    uint t_width = height;
    uint t_height = width;
    result[sample_index * t_width * t_height + y * t_width + x] = nums[get_global_linear_id()];
}

/* kernel void fft_2d( */
/*     global float *nums, */
/*     global float2 *result, */
/*     uint height, */
/*     uint width, */
/*     uint log_height, */
/*     uint log_width */
/* ) { */
/*     uint2 image_position; */
/*     image_position.y = get_global_id(0); */
/*     image_position.x = get_global_id(1); */

/*     uint first_collumn_in_row_index = image_position.y * width; */

/*     int is_x_before_half = image_position.x < width >> 1; */

/*     if (is_x_before_half) { */
/*         uint index = 2u * image_position.x; */
/*         uint reverse = reverse_bits(index, log_width); */
/*         result[first_collumn_in_row_index + index] = (float2) (nums[first_collumn_in_row_index + reverse], 0.0); */

/*         index += 1u; */
/*         reverse = reverse_bits(index, log_width); */
/*         result[first_collumn_in_row_index + index] = (float2) (nums[first_collumn_in_row_index + reverse], 0.0); */
/*     } */

/*     barrier(CLK_GLOBAL_MEM_FENCE); */

/*     for (uint s = 1u; s <= log_width; s++) { */
/*         if (is_x_before_half) { */
/*             uint m_half = 1u << (s - 1u); */

/*             uint k = image_position.x / m_half * (1u << s); */
/*             uint j = image_position.x % m_half; */
/*             uint k_plus_j = k + j; */
/*             float2 twiddle = cis(-M_PI_F * (float)j / (float)m_half); */

/*             uint first_half_index = first_collumn_in_row_index + k_plus_j; */
/*             uint second_half_index = first_half_index + m_half; */

/*             float2 t = complex_multiplication(twiddle, result[second_half_index]); */
/*             float2 u = result[first_half_index]; */
/*             result[first_half_index] = u + t; */
/*             result[second_half_index] = u - t; */
/*         } */

/*         barrier(CLK_GLOBAL_MEM_FENCE); */
/*     } */

/*     uint global_index = image_position.y * width + image_position.x; */
/*     float2 result_value = result[global_index]; */
/*     printf("x: %d, y: %d, global_index: %d, value: %e + i%e\n", image_position.x, image_position.y, global_index, result_value.x, result_value.y); */

/*     if (image_position.y < height >> 1) { */
/*         uint index = 2u * image_position.y; */
/*         uint reverse = reverse_bits(index, log_height); */
/*         result[index * width + image_position.x] = (float2) (nums[reverse * width + image_position.x], 0.0); */

/*         index += 1u; */
/*         reverse = reverse_bits(index, log_height); */
/*         result[index * width + image_position.x] = (float2) (nums[reverse * width + image_position.x], 0.0); */

/*         for (uint s = 1u; s <= log_height; s++) { */
/*             barrier(CLK_GLOBAL_MEM_FENCE); */

/*             uint m_half = 1u << (s - 1u); */

/*             uint k = image_position.y / m_half * (1u << s); */
/*             uint j = image_position.y % m_half; */
/*             uint k_plus_j = k + j; */
/*             float2 twiddle = cis(-M_PI_F * (float)j / (float)m_half); */

/*             uint first_half_index = k_plus_j * width + image_position.x; */
/*             uint second_half_index = first_half_index + m_half * width; */

/*             float2 t = complex_multiplication(twiddle, result[second_half_index]); */
/*             float2 u = result[first_half_index]; */
/*             result[first_half_index] = u + t; */
/*             result[second_half_index] = u - t; */
/*         } */
/*     } */
/* } */

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