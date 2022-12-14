int get_filter_starting_pixel_index(
    int filter_id,

    int filter_width,
    int filter_height
) {
    return filter_width * filter_height * filter_id;
}

int get_pixel_x_from_id(
    int id,

    int width
) {
    return id % width;
}

int get_pixel_y_from_id(
    int id,
    int x,

    int width
) {
    return (id  - x) / width;
}

int get_image_pixel_id(
    int local_id, 

    int filter_id,
    int filter_starting_pixel_index,

    int image_width,
    int filter_width
) {
    int local_pixel_x = get_pixel_x_from_id(local_id, filter_width);
    int local_pixel_y = get_pixel_y_from_id(local_id, local_pixel_x, filter_width);

    int filter_starting_global_pixel_x = get_pixel_x_from_id(filter_starting_pixel_index, image_width);
    int filter_starting_global_pixel_y = get_pixel_y_from_id(filter_starting_pixel_index, filter_starting_global_pixel_x, image_width);

    int pixel_id = (filter_starting_global_pixel_y + local_pixel_y) * image_width
        + filter_starting_global_pixel_x + local_pixel_x;

    return pixel_id;
}

kernel void convolute(
    global float* image,
    constant float* filter,
    global float* output,

    local float* filtered,

    int image_width,
    int image_volume,

    int output_image_volume,

    int filter_width,
    int filter_height,
    int filter_volume,

    int samples_amount
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    int filter_index = get_group_id(1);
    int filter_pixel_index = get_local_id(1);

    int filter_starting_global_pixel_id = filter_index 
        + (filter_width - 1) * (int)floor((float)filter_index / (float)(image_width - filter_width + 1));

    int pixel_index = get_image_pixel_id(
        filter_pixel_index,

        filter_index,
        filter_starting_global_pixel_id,

        image_width,
        filter_width
    );

    filtered[filter_pixel_index] = 
        image[sample_index * image_volume + pixel_index] // the pixel
      * filter[filter_pixel_index]; // multiplied by the respective filter weight
    barrier(CLK_LOCAL_MEM_FENCE);

    if (filter_pixel_index == 0) {
        float result = 0.0f;

        for (int i = 0; i < filter_volume; i++) {
            result += filtered[i];
        }

        output[sample_index * output_image_volume + filter_index] = result;
    }
}

kernel void compute_gradients_for_one_filter_pixel(
    global float* image,
    global float* error_to_output_derivatives,
    global float* filter_pixel_gradients,

    int image_width,
    int image_volume,

    int filter_width,
    int filter_volume,
    
    int samples_amount,

    int output_width,
    int output_height,
    int output_volume,

    int pixel_index,
    int pixel_y,
    int pixel_x
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    int output_y = get_global_id(1);

    if (output_y >= output_width) {
        return;
    }

    int output_x = get_global_id(2);

    if (output_x >= output_height) {
        return;
    }

    int input_y = output_y + pixel_y;
    int input_x = output_x + pixel_x;

    int input_index = input_y * image_width + input_x;
    int global_input_index = sample_index * image_volume + input_index;

    int output_index = output_y * output_width + output_x;
    int global_output_index = sample_index * output_volume + output_index;

    filter_pixel_gradients[global_output_index] 
        = (float)image[global_input_index] * (float)error_to_output_derivatives[global_output_index];
}

kernel void compute_loss_to_input_derivatives(
    constant float* filter,
    global float* loss_to_output_derivatives,
    global float* loss_to_input_derivatives,

    int samples_amount,

    int filter_width,
    int filter_height,

    int output_height,
    int output_width,

    int inputs_amount,
    int inputs_width
) {
    int sample_index = get_global_id(0);

    if (sample_index >= samples_amount) {
        return;
    }

    int input_index = get_global_id(1);
    
    if (input_index >= inputs_amount) {
        return;
    }

    float loss_to_input_derivative = 0;

    int input_y = (int)floor((float)input_index / (float)inputs_width);
    int input_x = input_index % inputs_width;
    
    for (int output_y = 0; output_y < output_height; output_y++) {
        int filter_y = input_y - output_y + 1;
        if (filter_y >= 0) {
            for (int output_x = 0; output_x < output_width; output_x++) {
                int filter_x = input_x - output_x + 1;
                if (filter_x >= 0) {
                    int filter_index = filter_y * filter_width + filter_x;
                    loss_to_input_derivative += (float)filter[filter_index];
                }
            }
        }
    }
    
    int input_derivative_index = sample_index * inputs_amount + input_index;

    loss_to_input_derivatives[input_derivative_index] = loss_to_input_derivative;
}