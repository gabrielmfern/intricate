//! The module that contains the Sigmoid activation function.

use opencl3::{
    device::cl_float,
    memory::Buffer
};

use intricate_macros::ActivationLayer;

use savefile_derive::Savefile;

use crate::utils::OpenCLState;

const PROGRAM_NAME: &str = "SIGMOID";
const PROGRAM_SOURCE: &str = include_str!("kernels/sigmoid.cl");
const PROPAGATE_KERNEL_NAME: &str = "propagate";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

#[derive(Debug, Savefile, ActivationLayer)]
/// The classical Sigmoid activation function, somewhat similar to the `TanH` activation function
/// but still different. Will squash the outputs of the last layer between -1 and 1.
///
/// # Example
///
/// ```rust
/// use intricate::layers::{
///     activations::Sigmoid,
///     Layer,
/// };
///
/// let my_sigmoid = Sigmoid::new_raw(10);
/// ```
pub struct Sigmoid<'a> {
    /// The amount of inputs expected for this instance of the Sigmoid activation function.
    pub inputs_amount: usize,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The cloned last inputs forward passed into this instance of the Sigmoid activation
    /// function.
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The last outputs that came from the last forward pass to this instance of the Sigmoid
    /// activation function.
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_state: Option<&'a mut OpenCLState>
}

#[cfg(test)]
mod sigmoid_tests {
    use std::{f32::consts::E, ptr};

    use opencl3::{
        command_queue::{CL_BLOCKING, CL_NON_BLOCKING},
        device::cl_float,
        memory::{Buffer, CL_MEM_READ_ONLY}
    };
    use rand::{thread_rng, Rng};

    use crate::{
        layers::Layer, types::CompilationOrOpenCLError,
        utils::{approx_eq::assert_approx_equal_distance, setup_opencl, opencl::DeviceType},
    };

    use super::Sigmoid;

    #[test]
    fn should_propagate_to_correct_values() -> Result<(), CompilationOrOpenCLError> {
        let mut state = setup_opencl(DeviceType::GPU)?;

        let context = state.context;
        let queue = state.queues.first().unwrap();

        let samples_amount = 423;
        let numbers_amount = 141;

        let mut sigmoid = Sigmoid::new(numbers_amount);
        sigmoid.init(&mut state)?;

        let mut rng = thread_rng();
        let input_samples: Vec<f32> = (0..(samples_amount * numbers_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        let expected_outputs: Vec<f32> =
            input_samples.iter().map(|x| 1.0 / (1.0 + E.powf(-x))).collect();

        let mut input_samples_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            numbers_amount * samples_amount,
            ptr::null_mut(),
        )?;

        queue
            .enqueue_write_buffer(
                &mut input_samples_buffer,
                CL_BLOCKING,
                0,
                input_samples.as_slice(),
                &[],
            )?
            .wait()?;

        let actual_outputs_buffer = sigmoid.propagate(&input_samples_buffer)?;

        let mut actual_outputs = vec![0.0; numbers_amount * samples_amount];
        let actual_outputs_slice = actual_outputs.as_mut_slice();
        queue
            .enqueue_read_buffer(
                &actual_outputs_buffer,
                CL_BLOCKING,
                0,
                actual_outputs_slice,
                &[],
            )?
            .wait()?;

        assert_approx_equal_distance(&expected_outputs, &actual_outputs, 0.01);

        Ok(())
    }

    #[test]
    fn should_back_propagate_returning_the_correct_derivatives(
    ) -> Result<(), CompilationOrOpenCLError> {
        let mut state = setup_opencl(DeviceType::GPU)?;

        let context = state.context;
        let queue = state.queues.first().unwrap();

        let samples_amount = 432;
        let numbers_amount = 331;

        let mut tanh = Sigmoid::new(numbers_amount);
        tanh.init(&mut state)?;

        let mut rng = thread_rng();
        let input_samples: Vec<f32> = (0..(samples_amount * numbers_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        let first_derivatives: Vec<f32> = (0..(samples_amount * numbers_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();

        let mut input_samples_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            numbers_amount * samples_amount,
            ptr::null_mut(),
        )?;
        let mut first_derivatives_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            numbers_amount * samples_amount,
            ptr::null_mut(),
        )?;

        queue
            .enqueue_write_buffer(
                &mut first_derivatives_buffer,
                CL_BLOCKING,
                0,
                first_derivatives.as_slice(),
                &[],
            )?
            .wait()?;

        queue
            .enqueue_write_buffer(
                &mut input_samples_buffer,
                CL_BLOCKING,
                0,
                input_samples.as_slice(),
                &[],
            )?
            .wait()?;

        tanh.propagate(&input_samples_buffer)?;

        let expected_loss_to_input_derivatives: Vec<Vec<f32>> = (0..samples_amount)
            .into_iter()
            .map(|i| {
                (0..numbers_amount) // inputs
                    .into_iter()
                    .map(|j| {
                        let input = input_samples[i * numbers_amount + j];
                        let sigmoid = 1.0 / (1.0 + E.powf(-input));
                        sigmoid
                            * (1.0 - sigmoid)
                            * (0..numbers_amount) // outputs
                                .into_iter()
                                .map(|k| first_derivatives[i * numbers_amount + k])
                                .sum::<f32>()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        let actual_loss_to_input_derivatives_buffer = tanh
            .back_propagate(true, &first_derivatives_buffer, 0.0)?
            .unwrap();
        let mut actual_loss_to_input_derivatives = vec![0.0; numbers_amount * samples_amount];
        let actual_loss_to_input_derivatives_slice =
            actual_loss_to_input_derivatives.as_mut_slice();
        queue
            .enqueue_read_buffer(
                &actual_loss_to_input_derivatives_buffer,
                CL_NON_BLOCKING,
                0,
                actual_loss_to_input_derivatives_slice,
                &[],
            )?
            .wait()?;

        println!("derivatives CPU: {:?}", &expected_loss_to_input_derivatives,);
        println!("\nderivatives GPU: {:?}", &actual_loss_to_input_derivatives);

        assert_approx_equal_distance(
            &actual_loss_to_input_derivatives,
            &expected_loss_to_input_derivatives
                .iter()
                .map(|v| v.to_vec())
                .flatten()
                .collect(),
            0.01,
        );

        Ok(())
    }
}