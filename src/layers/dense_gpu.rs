#[allow(unused_imports)]
use opencl3::{
    command_queue::{CommandQueue, CL_BLOCKING, CL_NON_BLOCKING},
    context::Context,
    device::{cl_float, get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    error_codes::{cl_int, ClError},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
};
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IntoParallelIterator};
use savefile_derive::Savefile;
use std::mem;
use std::ptr;

#[allow(unused_imports)]
use crate::{layers::Layer, utils::approx_eq::assert_approx_equal_distance};

#[allow(unused_imports)]
use super::{dense::Dense, OpenCLLayer};

const PROPAGATION_PROGRAM_SORUCE: &str = include_str!("kernels/dense_propagation.cl");
const BACK_PROPAGATION_PROGRAM_SOURCE: &str =
    include_str!("kernels/dense_back_propagation.cl");
const PROPAGATION_KERNEL_NAME: &str = "dense_propagate";
const WEIGHTS_GRADIENT_APPLICATION_KERNEL_NAME: &str = "weights_gradient_application";
const BIAS_GRADIENT_APPLICATION_KERNEL_NAME: &str = "bias_gradient_application";
const LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME: &str = "compute_loss_derivative_with_respect_to_inputs";

#[test]
fn should_apply_gradients_just_like_normal_dense() -> Result<(), ClError> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;

    let first_device = Device::new(*device_ids.get(0).expect("There is no GPU device!"));

    let context = Context::from_device(&first_device)?;
    let queue = CommandQueue::create_with_properties(&context, first_device.id(), 0, 0)?;

    let samples_amount = 100;
    let inputs_amount = 10;
    let outputs_amount = 5;

    let mut gpu_dense = DenseGPU::new(inputs_amount, outputs_amount, &context, &queue)?;

    let mut normal_dense = Dense::new(inputs_amount, outputs_amount);
    normal_dense.weights = gpu_dense.weights.to_vec();
    normal_dense.biases = gpu_dense.biases.to_vec();

    let loss_to_output_derivatives = vec![vec![0.5; outputs_amount]; samples_amount]; 

    let input_samples = vec![vec![0.1; inputs_amount]; samples_amount];
    normal_dense.last_inputs = input_samples.to_vec();

    let mut input_samples_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY,
        samples_amount * inputs_amount,
        ptr::null_mut(),
    )?;

    let input_samples_gpu_write_event = queue.enqueue_write_buffer(
        &mut input_samples_buffer,
        CL_BLOCKING,
        0,
        input_samples
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect::<Vec<f32>>()
            .as_slice(),
        &[],
    )?;

    input_samples_gpu_write_event.wait()?;

    gpu_dense.last_inputs_buffer = Some(&input_samples_buffer);

    let mut loss_to_output_derivatives_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY,
        samples_amount * outputs_amount,
        ptr::null_mut(),
    )?;

    let derivatives_write_event = queue.enqueue_write_buffer(
        &mut loss_to_output_derivatives_buffer,
        CL_BLOCKING,
        0,
        loss_to_output_derivatives
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect::<Vec<f32>>()
            .as_slice(),
        &[],
    )?;

    derivatives_write_event.wait()?;

    gpu_dense.back_propagate(false, &loss_to_output_derivatives_buffer, 0.05)?;
    normal_dense.back_propagate(false, &loss_to_output_derivatives, 0.05);

    gpu_dense.sync_data_from_gpu_with_cpu()?;

    println!("new weights GPU: {:?}", gpu_dense.weights);
    println!("new weights CPU: {:?}", normal_dense.weights);

    assert_approx_equal_distance(
        &gpu_dense.weights.iter().map(|x| x.to_vec()).flatten().collect(),
        &normal_dense.weights.iter().map(|x| x.to_vec()).flatten().collect(),
        0.05,
    );

    println!("new biases GPU: {:?}", gpu_dense.biases);
    println!("new biases CPU: {:?}", normal_dense.biases);

    assert_approx_equal_distance(
        &gpu_dense.biases, 
        &normal_dense.biases, 
        0.05
    );

    Ok(())
}

#[test]
fn should_propagate_to_same_value_as_normal_dense() -> Result<(), ClError> {
    let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;

    let first_device = Device::new(*device_ids.get(0).expect("There is no GPU device!"));

    let context = Context::from_device(&first_device)?;
    let queue = CommandQueue::create_with_properties(&context, first_device.id(), 0, 0)?;

    let samples_amount = 100;
    let inputs_amount = 20;
    let outputs_amount = 5;

    let mut gpu_dense = DenseGPU::new(inputs_amount, outputs_amount, &context, &queue)?;

    let mut normal_dense = Dense::new(inputs_amount, outputs_amount);
    normal_dense.weights = gpu_dense.weights.to_vec();
    normal_dense.biases = gpu_dense.biases.to_vec();

    let input_samples = vec![vec![0.1; inputs_amount]; samples_amount];

    let expected_outputs = normal_dense.propagate(&input_samples);

    let mut input_samples_buffer = Buffer::<cl_float>::create(
        &context,
        CL_MEM_READ_ONLY,
        samples_amount * inputs_amount,
        ptr::null_mut(),
    )?;

    let input_samples_gpu_write_event = queue.enqueue_write_buffer(
        &mut input_samples_buffer,
        CL_BLOCKING,
        0,
        input_samples
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect::<Vec<f32>>()
            .as_slice(),
        &[],
    )?;

    input_samples_gpu_write_event.wait()?;

    let gpu_outputs_buffer = gpu_dense.propagate(&input_samples_buffer)?;

    let mut outputs_vec = vec![0.0; samples_amount * outputs_amount];
    let gpu_flattend_outputs = outputs_vec.as_mut_slice();

    let read_flattened_outputs_gpu = queue.enqueue_read_buffer(
        gpu_outputs_buffer,
        CL_NON_BLOCKING,
        0,
        gpu_flattend_outputs,
        &[],
    )?;

    read_flattened_outputs_gpu.wait()?;

    let flattened_expected_outputs = expected_outputs
        .iter()
        .map(|x| x.to_vec())
        .flatten()
        .collect();

    assert_approx_equal_distance(&outputs_vec, &flattened_expected_outputs, 0.2);

    Ok(())
}

#[derive(Debug, Savefile)]
/// A densely connected layer, this layer consists of some inputs
/// and the weights that connect each input to all outputs,
/// its propagation results in a dot product between these weights
/// and the inputs received in the propagation method
///
/// For this layer all the definitions are the same, the only difference
/// is computation is done in all the devices Intricate is able to find
pub struct DenseGPU<'a> {
    pub inputs_amount: usize,
    pub outputs_amount: usize,

    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub weights_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub biases_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub last_inputs_buffer: Option<&'a Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    propagation_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    propagation_program: Option<Program>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    weights_gradient_application_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    bias_gradient_application_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    loss_to_input_differentiation_kernel: Option<Kernel>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    back_propagation_program: Option<Program>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_context: Option<&'a Context>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_queue: Option<&'a CommandQueue>,
}

impl<'a> DenseGPU<'a> {

    #[allow(dead_code)]

    pub fn new(
        inputs_amount: usize,
        outputs_amount: usize,
        context: &'a Context,
        queue: &'a CommandQueue,
    ) -> Result<DenseGPU<'a>, ClError> {
        let mut rng = rand::thread_rng();

        let weights = (0..inputs_amount)
            .into_iter()
            .map(|_| {
                (0..outputs_amount)
                    .into_iter()
                    .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        let mut weights_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            inputs_amount * outputs_amount,
            ptr::null_mut(),
        )?;

        let biases = (0..outputs_amount)
            .into_iter()
            .map(|_| rng.gen_range(-1.0_f32..=1.0_f32))
            .collect::<Vec<f32>>();
        let mut biases_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            outputs_amount,
            ptr::null_mut(),
        )?;

        let weights_gpu_write_event = queue.enqueue_write_buffer(
            &mut weights_buffer,
            CL_NON_BLOCKING,
            0,
            weights
                .par_iter()
                .map(|x| x.to_vec())
                .flatten()
                .collect::<Vec<f32>>()
                .as_slice(),
            &[],
        )?;
        let biases_gpu_write_event = queue.enqueue_write_buffer(
            &mut biases_buffer,
            CL_NON_BLOCKING,
            0,
            biases.as_slice(),
            &[],
        )?;

        weights_gpu_write_event.wait()?;
        biases_gpu_write_event.wait()?;

        let propagation_program_compilation_result =
            Program::create_and_build_from_source(context, PROPAGATION_PROGRAM_SORUCE, "");
        if propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the dense_propagation.cl Program:\n{:?}",
                propagation_program_compilation_result.err().unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let back_propagation_program_compilation_result =
            Program::create_and_build_from_source(&context, BACK_PROPAGATION_PROGRAM_SOURCE, "");
        if back_propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the dense_back_propagation.cl Program:\n{:?}",
                back_propagation_program_compilation_result 
                    .err()
                    .unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let propagation_program = propagation_program_compilation_result.unwrap();
        let propagation_kernel = Kernel::create(&propagation_program, PROPAGATION_KERNEL_NAME)?;

        let back_propagation_program = back_propagation_program_compilation_result.unwrap();
        let bias_gradient_application_kernel = Kernel::create(&back_propagation_program, BIAS_GRADIENT_APPLICATION_KERNEL_NAME)?;
        let weights_gradient_application_kernel = Kernel::create(&back_propagation_program, WEIGHTS_GRADIENT_APPLICATION_KERNEL_NAME)?;
        let loss_to_input_differentiation_kernel = Kernel::create(&back_propagation_program, LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME)?;

        Ok(DenseGPU {
            inputs_amount,
            outputs_amount,
            weights_buffer: Some(weights_buffer),
            biases_buffer: Some(biases_buffer),
            weights,
            biases,
            propagation_kernel: Some(propagation_kernel),
            propagation_program: Some(propagation_program),
            back_propagation_program: Some(back_propagation_program),
            weights_gradient_application_kernel: Some(weights_gradient_application_kernel),
            bias_gradient_application_kernel: Some(bias_gradient_application_kernel),
            loss_to_input_differentiation_kernel: Some(loss_to_input_differentiation_kernel),
            last_inputs_buffer: None,
            last_outputs_buffer: None,
            opencl_queue: Some(queue),
            opencl_context: Some(context),
        })
    }
}

impl<'a> OpenCLLayer<'a> for DenseGPU<'a> {
    fn clean_up_gpu_state(&mut self) -> () {
        if self.weights_buffer.is_some() {
            drop(self.weights_buffer.as_ref().unwrap());
        }

        if self.biases_buffer.is_some() {
            drop(self.biases_buffer.as_ref().unwrap());
        }

        if self.last_inputs_buffer.is_some() {
            drop(self.last_inputs_buffer.as_ref().unwrap());
        }

        if self.last_outputs_buffer.is_some() {
            drop(self.last_outputs_buffer.as_ref().unwrap());
        }
    }

    fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), ClError> {
        assert!(self.weights_buffer.is_some());
        assert!(self.biases_buffer.is_some());

        let mut weights_flat_vec = vec![0.0; self.inputs_amount * self.outputs_amount];
        let weights_flat_slice = weights_flat_vec.as_mut_slice();

        let mut biases_vec = vec![0.0; self.outputs_amount];
        let biases_slice = biases_vec.as_mut_slice();

        let queue = self.opencl_queue.as_ref().unwrap();

        let read_weights_event = queue.enqueue_read_buffer(
            self.weights_buffer.as_ref().unwrap(), 
            CL_NON_BLOCKING, 
            0, 
            weights_flat_slice,
            &[]
        )?;

        let read_biases_event = queue.enqueue_read_buffer(
            self.biases_buffer.as_ref().unwrap(),
            CL_NON_BLOCKING,
            0, 
            biases_slice, 
            &[]
        )?;

        read_weights_event.wait()?;
        read_biases_event.wait()?;

        self.biases = biases_vec;
        self.weights = (0..self.inputs_amount).into_par_iter()
            .map(|i| {
                let row_part = i * self.outputs_amount;
                (0..self.outputs_amount)
                    .into_iter()
                    .map(|j| {
                        let flat_index = row_part + j;
                        weights_flat_vec[flat_index]
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        Ok(())
    }

    fn send_to_gpu(
        &mut self,
        queue: &'a CommandQueue,
        context: &'a Context,
    ) -> Result<(), ClError> {
        assert!(!self.weights.is_empty());
        assert!(!self.biases.is_empty());

        let mut weights_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.inputs_amount * self.outputs_amount,
            ptr::null_mut(),
        )?;
        let mut biases_buffer = Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.outputs_amount,
            ptr::null_mut(),
        )?;

        let weights_gpu_write_event = queue.enqueue_write_buffer(
            &mut weights_buffer,
            CL_NON_BLOCKING,
            0,
            self.weights
                .par_iter()
                .map(|x| x.to_vec())
                .flatten()
                .collect::<Vec<f32>>()
                .as_slice(),
            &[],
        )?;
        let biases_gpu_write_event = queue.enqueue_write_buffer(
            &mut biases_buffer,
            CL_NON_BLOCKING,
            0,
            self.biases.as_slice(),
            &[],
        )?;

        weights_gpu_write_event.wait()?;
        biases_gpu_write_event.wait()?;

        self.weights_buffer = Some(weights_buffer);
        self.biases_buffer = Some(biases_buffer);

        self.opencl_context = Some(context);
        self.opencl_queue = Some(queue);

        let propagation_program_compilation_result =
            Program::create_and_build_from_source(context, PROPAGATION_PROGRAM_SORUCE, "");
        if propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the dense_propagation.cl Program:\n{:?}",
                propagation_program_compilation_result.err().unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let back_propagation_program_compilation_result =
            Program::create_and_build_from_source(&context, BACK_PROPAGATION_PROGRAM_SOURCE, "");
        if back_propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the dense_back_propagation.cl Program:\n{:?}",
                back_propagation_program_compilation_result 
                    .err()
                    .unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let propagation_program = propagation_program_compilation_result.unwrap();
        let propagation_kernel = Kernel::create(&propagation_program, PROPAGATION_KERNEL_NAME)?;

        let back_propagation_program = back_propagation_program_compilation_result.unwrap();
        let bias_gradient_application_kernel = Kernel::create(&back_propagation_program, BIAS_GRADIENT_APPLICATION_KERNEL_NAME)?;
        let weights_gradient_application_kernel = Kernel::create(&back_propagation_program, WEIGHTS_GRADIENT_APPLICATION_KERNEL_NAME)?;
        let loss_to_input_differentiation_kernel = Kernel::create(&back_propagation_program, LOSS_TO_INPUT_DIFFERENTIATION_KERNEL_NAME)?;

        self.propagation_program = Some(propagation_program);
        self.propagation_kernel = Some(propagation_kernel);

        self.back_propagation_program = Some(back_propagation_program);
        self.weights_gradient_application_kernel = Some(weights_gradient_application_kernel);
        self.bias_gradient_application_kernel = Some(bias_gradient_application_kernel);
        self.loss_to_input_differentiation_kernel = Some(loss_to_input_differentiation_kernel);

        Ok(())
    }

    fn get_last_inputs(&self) -> Option<&'a Buffer<cl_float>> {
        self.last_inputs_buffer
    }

    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>> {
        self.last_outputs_buffer.as_ref()
    }

    fn get_inputs_amount(&self) -> usize {
        self.inputs_amount
    }

    fn get_outputs_amount(&self) -> usize {
        self.outputs_amount
    }

    fn propagate(
        &mut self,
        input_samples: &'a Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, ClError> {
        assert!(self.opencl_context.is_some());
        assert!(self.opencl_queue.is_some());

        if self.last_inputs_buffer.is_some() {
            drop(self.last_inputs_buffer.as_ref().unwrap());
        }
        if self.last_outputs_buffer.is_some() {
            drop(self.last_outputs_buffer.as_ref().unwrap());
        }

        self.last_inputs_buffer = Some(input_samples);

        let samples_amount =
            input_samples.size()? / self.inputs_amount / mem::size_of::<cl_float>();

        let outputs_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_ONLY,
            self.outputs_amount * samples_amount,
            ptr::null_mut(),
        )?;

        let arg_inputs_amount: cl_int = self.inputs_amount as cl_int;

        let kernel_event = ExecuteKernel::new(self.propagation_kernel.as_ref().unwrap())
            .set_arg(input_samples)
            .set_arg(self.biases_buffer.as_ref().unwrap())
            .set_arg(self.weights_buffer.as_ref().unwrap())
            .set_arg(&outputs_buffer)
            .set_arg(&arg_inputs_amount)
            .set_global_work_sizes(&[samples_amount, self.outputs_amount])
            .enqueue_nd_range(&self.opencl_queue.unwrap())?;

        kernel_event.wait()?;

        self.last_outputs_buffer = Some(outputs_buffer);
        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Buffer<cl_float>,
        learning_rate: cl_float,
    ) -> Result<Option<Buffer<cl_float>>, ClError> {
        assert!(self.last_inputs_buffer.is_some());
        assert!(self.opencl_context.is_some());
        assert!(self.opencl_queue.is_some());

        let samples_amount = layer_output_to_error_derivative.size()? / self.outputs_amount
            * mem::size_of::<cl_float>();
        let queue = self.opencl_queue.unwrap();
        let mut layer_input_to_error_derivatives_buffer = None;

        if should_calculate_input_to_error_derivative {
             layer_input_to_error_derivatives_buffer = Some(Buffer::<cl_float>::create(
                self.opencl_context.unwrap(),
                CL_MEM_READ_WRITE, 
                samples_amount * self.inputs_amount, 
                ptr::null_mut()
            )?);

            let layer_loss_to_input_differentiation_kernel_event =
                ExecuteKernel::new(self.loss_to_input_differentiation_kernel.as_ref().unwrap())
                    .set_arg(self.weights_buffer.as_ref().unwrap())
                    .set_arg(layer_output_to_error_derivative)
                    .set_arg(layer_input_to_error_derivatives_buffer.as_ref().unwrap())
                    .set_arg(&(self.inputs_amount as cl_int))
                    .enqueue_nd_range(queue)?;

            layer_loss_to_input_differentiation_kernel_event.wait()?;
        }

        let new_weights_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            self.inputs_amount * self.outputs_amount,
            ptr::null_mut(),
        )?;

        let weights_apply_gradients_kernel_event =
            ExecuteKernel::new(self.weights_gradient_application_kernel.as_ref().unwrap())
                .set_arg(layer_output_to_error_derivative)
                .set_arg(self.last_inputs_buffer.unwrap())
                .set_arg(self.weights_buffer.as_ref().unwrap())
                .set_arg(&new_weights_buffer)
                .set_arg(&(samples_amount as cl_int))
                .set_arg(&(learning_rate as cl_float))
                .set_global_work_sizes(&[self.inputs_amount, self.outputs_amount])
                .enqueue_nd_range(queue)?;

        weights_apply_gradients_kernel_event.wait()?;

        let new_biases_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_WRITE,
            self.outputs_amount,
            ptr::null_mut()
        )?;

        let bias_apply_gradients_kernel_event =
            ExecuteKernel::new(self.bias_gradient_application_kernel.as_ref().unwrap())
                .set_arg(layer_output_to_error_derivative)
                .set_arg(self.biases_buffer.as_ref().unwrap())
                .set_arg(&new_biases_buffer)
                .set_arg(&(samples_amount as cl_int))
                .set_arg(&(learning_rate as cl_float))
                .set_global_work_size(self.outputs_amount)
                .enqueue_nd_range(queue)?;

        bias_apply_gradients_kernel_event.wait()?;

        self.weights_buffer = Some(new_weights_buffer);
        self.biases_buffer = Some(new_biases_buffer);

        Ok(layer_input_to_error_derivatives_buffer)
    }
}