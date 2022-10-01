int get_image_pixel_id(
    int local_id, 
    int filter_Id,

    int image_width,
    int image_height,

    int filter_width,
    int filter_height
) {
    int 

    return 0;
}

kernel void convolute(
    global float* image,
    global float* filter,
    global float* output,

    local float* filtered,

    int image_width,
    int image_height,

    int filter_width,
    int filter_height
) {
    int filter_index = get_group_id(0);
    int filter_pixel_index = get_local_id(0);
}