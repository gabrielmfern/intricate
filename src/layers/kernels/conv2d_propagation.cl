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
    const float* filter,
    global float* output,

    local float* filtered,

    int image_width,
    int filter_width,
    int filter_height,
    int filter_volume
) {
    int filter_index = get_group_id(0);
    int filter_pixel_index = get_local_id(0);

    int filter_starting_global_pixel_id = filter_index * filter_width * filter_height;

    int pixel_index = get_image_pixel_id(
        filter_pixel_index,

        filter_index,
        0,

        image_width,
        filter_width
    );

    filtered[filter_pixel_index] = image[filter_pixel_index] * filter[filter_pixel_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (filter_pixel_index == 0) {
        float result = 0.0f;

        for (int i = 0; i < filter_volume; i++) {
            result += filtered[i];
        }

        output[filter_index] = result;
    }
}