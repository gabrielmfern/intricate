use std::ptr;

use opencl3::{
    memory::{Buffer, CL_MEM_READ_ONLY},
    types::{cl_float, CL_NON_BLOCKING},
};
use rand::{thread_rng, Rng};

use super::CategoricalCrossEntropy;
use crate::utils::{approx_eq::assert_approx_equal_distance, setup_opencl, OpenCLState};
use crate::{loss_functions::LossFunction, utils::opencl::DeviceType};

#[test]
fn should_compute_derivatives_up_to_a_certain_precision() {
    let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU).unwrap();

    let context = &opencl_state.context;

    let mut gpu_loss = CategoricalCrossEntropy::new();
    gpu_loss.init(&opencl_state).unwrap();

    let outputs_amount: usize = 61;
    let samples_amount: usize = 113;
    let mut rng = rand::thread_rng();

    let output_samples: Vec<f32> = (0..(samples_amount * outputs_amount))
        .into_iter()
        .map(|_| rng.gen_range(0.0_f32..1.0_f32))
        .collect();
    let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
        .into_iter()
        .map(|_| rng.gen_range(0.0_f32..1.0_f32))
        .collect();

    let expected_derivatives: Vec<f32> = expected_outputs
        .iter()
        .zip(&output_samples)
        .map(|(expected_output, output)| (expected_output, output.max(0.0000001).min(0.9999999)))
        .map(|(expected_output, output)| -expected_output / output)
        .collect();

    let mut outputs_buf = Buffer::<cl_float>::create(
        context,
        CL_MEM_READ_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )
    .unwrap();
    let mut expected_outputs_buf = Buffer::<cl_float>::create(
        context,
        CL_MEM_READ_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )
    .unwrap();

    let queue = opencl_state.queues.first().unwrap();

    queue
        .enqueue_write_buffer(
            &mut outputs_buf,
            CL_NON_BLOCKING,
            0,
            output_samples.as_slice(),
            &[],
        )
        .unwrap()
        .wait()
        .unwrap();
    queue
        .enqueue_write_buffer(
            &mut expected_outputs_buf,
            CL_NON_BLOCKING,
            0,
            expected_outputs.as_slice(),
            &[],
        )
        .unwrap()
        .wait()
        .unwrap();

    let buf = gpu_loss
        .compute_loss_derivative_with_respect_to_output_samples(
            &outputs_buf,
            &expected_outputs_buf,
            samples_amount,
        )
        .unwrap();
    let mut derivatives_vec = vec![0.0; samples_amount * outputs_amount];
    let derivatives_slice = derivatives_vec.as_mut_slice();

    queue
        .enqueue_read_buffer(&buf, CL_NON_BLOCKING, 0, derivatives_slice, &[])
        .unwrap()
        .wait()
        .unwrap();

    assert_approx_equal_distance(&expected_derivatives, &derivatives_vec, 0.01);
}

#[test]
fn should_compute_loss_up_to_a_certain_precision() {
    let opencl_state: OpenCLState = setup_opencl(DeviceType::GPU).unwrap();
    let context = &opencl_state.context;

    let mut loss = CategoricalCrossEntropy::new();
    loss.init(&opencl_state).unwrap();

    let mut rng = thread_rng();
    let samples_amount = 1000;
    let outputs_amount = 290;
    let outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
        .into_iter()
        .map(|_| rng.gen_range(0.0_f32..1.0_f32))
        .collect();
    let expected_outputs: Vec<f32> = (0..(samples_amount * outputs_amount))
        .into_iter()
        .map(|_| rng.gen_range(0.0_f32..1.0_f32))
        .collect();

    let expected_loss: f32 = expected_outputs
        .iter()
        .zip(&outputs)
        .map(|(expected_output, output)| (expected_output, output.max(0.0000001).min(0.9999999)))
        .map(|(expected_output, output)| {
            -expected_output * output.ln()
            // + (1.0 - expected_output) * (1.0 - output).ln())
        })
        .sum::<f32>()
        / samples_amount as f32;
    let mut outputs_buf = Buffer::<cl_float>::create(
        context,
        CL_MEM_READ_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )
    .unwrap();
    let mut expected_outputs_buf = Buffer::<cl_float>::create(
        context,
        CL_MEM_READ_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )
    .unwrap();

    let queue = opencl_state.queues.first().unwrap();

    queue
        .enqueue_write_buffer(
            &mut outputs_buf,
            CL_NON_BLOCKING,
            0,
            outputs.as_slice(),
            &[],
        )
        .unwrap()
        .wait()
        .unwrap();
    queue
        .enqueue_write_buffer(
            &mut expected_outputs_buf,
            CL_NON_BLOCKING,
            0,
            expected_outputs.as_slice(),
            &[],
        )
        .unwrap()
        .wait()
        .unwrap();

    let actual_loss = loss
        .compute_loss(&outputs_buf, &expected_outputs_buf, samples_amount)
        .unwrap();

    let largest_loss = expected_loss.max(actual_loss);
    println!(
        "|({} - {}) / {}| <= 0.1%",
        expected_loss, actual_loss, largest_loss
    );
    assert!((expected_loss - actual_loss).abs() / largest_loss <= 0.001);
}
