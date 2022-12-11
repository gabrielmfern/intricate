//! The module that contains the TanH activation function.

use opencl3::{device::cl_float, memory::Buffer};

use intricate_macros::ActivationLayer;

use savefile_derive::Savefile;

use crate::utils::OpenCLState;

const PROGRAM_NAME: &str = "TANH";
const PROGRAM_SOURCE: &str = include_str!("kernels/tanh.cl");
const PROPAGATE_KERNEL_NAME: &str = "propagate";
const BACK_PROPAGATE_KERNEL_NAME: &str = "back_propagate";

#[derive(Debug, Savefile, ActivationLayer)]
/// The `Hyperbolic Tangent` activation function, similar to the Sigmoid but different,
/// also squashed its inputs between -1 and 1.
pub struct TanH<'a> {
    /// The amount of inputs this instance of TanH expects.
    pub inputs_amount: usize,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The cloned inputs last forward passed into this TaNH.
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The outputs that came out from the last forward pass into this TanH.
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_state: Option<&'a OpenCLState>,
}

#[cfg(test)]
mod tanh_tests {
    use std::ptr;

    use opencl3::{
        command_queue::{CL_BLOCKING, CL_NON_BLOCKING},
        device::cl_float,
        memory::{Buffer, CL_MEM_READ_ONLY},
    };
    use rand::{thread_rng, Rng};

    use crate::{
        layers::Layer,
        utils::{approx_eq::assert_approx_equal_distance, opencl::DeviceType, setup_opencl},
    };

    use super::TanH;

    #[test]
    fn should_propagate_returning_correct_values() -> () {
        let state = setup_opencl(DeviceType::GPU).unwrap();
        let context = &state.context;
        let queue = state.queues.first().unwrap();

        let samples_amount = 423;
        let numbers_amount = 1341;

        let mut tanh = TanH::new(numbers_amount);
        tanh.init(&state).unwrap();

        let mut rng = thread_rng();
        let input_samples: Vec<f32> = (0..(samples_amount * numbers_amount))
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        let expected_outputs: Vec<f32> = input_samples.iter().map(|x| x.tanh()).collect();

        let mut input_samples_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            numbers_amount * samples_amount,
            ptr::null_mut(),
        )
        .unwrap();

        queue
            .enqueue_write_buffer(
                &mut input_samples_buffer,
                CL_BLOCKING,
                0,
                input_samples.as_slice(),
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        let actual_outputs_buffer = tanh.propagate(&input_samples_buffer).unwrap();

        let mut actual_outputs = vec![0.0; numbers_amount * samples_amount];
        let actual_outputs_slice = actual_outputs.as_mut_slice();
        queue
            .enqueue_read_buffer(
                &actual_outputs_buffer,
                CL_BLOCKING,
                0,
                actual_outputs_slice,
                &[],
            )
            .unwrap()
            .wait()
            .unwrap();

        assert_approx_equal_distance(&expected_outputs, &actual_outputs, 0.01);
    }

    #[test]
    fn should_back_propagate_returning_the_correct_derivatives(
    ) {
        let state = setup_opencl(DeviceType::GPU).unwrap();
        let context = &state.context;
        let queue = state.queues.first().unwrap();

        let samples_amount = 432;
        let numbers_amount = 331;

        let mut tanh = TanH::new(numbers_amount);
        tanh.init(&state).unwrap();

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
        ).unwrap();
        let mut first_derivatives_buffer = Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            numbers_amount * samples_amount,
            ptr::null_mut(),
        ).unwrap();

        queue
            .enqueue_write_buffer(
                &mut first_derivatives_buffer,
                CL_BLOCKING,
                0,
                first_derivatives.as_slice(),
                &[],
            ).unwrap()
            .wait().unwrap();

        queue
            .enqueue_write_buffer(
                &mut input_samples_buffer,
                CL_BLOCKING,
                0,
                input_samples.as_slice(),
                &[],
            ).unwrap()
            .wait().unwrap();

        tanh.propagate(&input_samples_buffer).unwrap();

        let expected_loss_to_input_derivatives: Vec<Vec<f32>> = (0..samples_amount)
            .into_iter()
            .map(|i| {
                (0..numbers_amount) // inputs
                    .into_iter()
                    .map(|j| {
                        let input = input_samples[i * numbers_amount + j];
                        (1.0 - input.tanh().powf(2.0))
                            * (0..numbers_amount) // outputs
                                .into_iter()
                                .map(|k| first_derivatives[i * numbers_amount + k])
                                .sum::<f32>()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        let actual_loss_to_input_derivatives_buffer = tanh
            .compute_loss_to_input_derivatives(&first_derivatives_buffer).unwrap();
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
            ).unwrap()
            .wait().unwrap();

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
    }
}