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

kernel void compute_gradients_per_sample(
    global float* image,
    global float* error_to_output_derivatives,
    global float* sample_gradients_per_filter_pixel,

    int image_width,
    int image_volume,

    int filter_width,
    int filter_volume,

    int output_width,
    int output_height,
    int output_volume
) {
    int filter_y = get_global_id(0);
    int filter_x = get_global_id(1);
    int sample_index = get_global_id(2);

    float pixel_gradient = 0.0f;
    for (int output_y = 0; output_y < output_height; output_y++) {
        int input_y = output_y + filter_y;
        for (int output_x = 0; output_x < output_width; output_x++) {
            int input_x = output_x + filter_x;

            int input_index = input_y * image_width + input_x;
            int global_input_index = sample_index * image_volume + input_index;

            int output_index = output_y * output_width + output_x;
            int global_output_index = sample_index * output_volume + output_index;

            pixel_gradient += (float)image[global_input_index] * (float)error_to_output_derivatives[global_output_index];
        }
    }

    sample_gradients_per_filter_pixel[get_global_linear_id()] 
        = pixel_gradient;
}

// kernel void compute_gradients_for_biases(
//    global float* loss_to_output_derivatives,
//    global float* gradients,
// 
//    int samples_amount,
//    int outputs_amount
//  {
//    int output_index = get_global_id(0);
//    if (output_index >= outputs_amount) {
//        return;
//    }
// 
//    float bias_gradient = 0.0f;
// 
//    for (int sample_index = 0; sample_index < samples_amount; sample_index++) {
//        int flat_output_i = sample_index * outputs_amount + output_index;
// 
//        bias_gradient += (float)loss_to_output_derivatives[flat_output_i];
//    }
// 
//    gradients[output_index] = bias_gradient / (float)samples_amount;
// 

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