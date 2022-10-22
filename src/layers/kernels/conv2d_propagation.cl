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