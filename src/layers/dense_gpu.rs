use opencl3::{
    command_queue::{CommandQueue, CL_NON_BLOCKING},
    context::Context,
    device::cl_float,
    error_codes::{ClError, cl_int},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, ClMem},
    program::Program, types::cl_event,
};
use std::mem;
use rand::Rng;
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use savefile_derive::Savefile;
use std::ptr;

use super::OpenCLLayer;

const PROPAGATION_KERNEL_SORUCE: &str = include_str!("kernels/dense_propagation.cl");

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
    opencl_context: Option<&'a Context>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_queue: Option<&'a CommandQueue>,
}

impl<'a> DenseGPU<'a> {
    /// Sends the weights and the biases of the current layer to the GPU
    /// as to be used in the propagation and back propagation
    ///
    /// mostly used after loading the layer using load_file and then
    /// there is a need to resend the data to the GPU since Savefile doesn't
    /// load the data into the GPU by itself
    pub fn send_to_gpu(&mut self, queue: &'a CommandQueue, context: &'a Context) -> Result<(), ClError> {
        assert!(!self.weights.is_empty());
        assert!(!self.biases.is_empty());

        let mut weights_buffer =Buffer::<cl_float>::create(
            context,
            CL_MEM_READ_WRITE,
            self.inputs_amount * self.outputs_amount,
            ptr::null_mut()
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
            self.weights.par_iter().map(|x| x.to_vec()).flatten().collect::<Vec<f32>>().as_slice(), 
            &[]
        )?;
        let biases_gpu_write_event = queue.enqueue_write_buffer(
            &mut biases_buffer,
            CL_NON_BLOCKING, 
            0, 
            self.biases.as_slice(),
            &[]
        )?;

        weights_gpu_write_event.wait()?;
        biases_gpu_write_event.wait()?;

        self.weights_buffer = Some(weights_buffer);
        self.biases_buffer = Some(biases_buffer);

        self.opencl_context = Some(context);
        self.opencl_queue = Some(queue);

        let propagation_program_compilation_result =
            Program::create_and_build_from_source(context, PROPAGATION_KERNEL_SORUCE, "");
        if propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the DenseGPU PROPAGATION_KERNEL:\n{:?}",
                propagation_program_compilation_result.err().unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let propagation_program = propagation_program_compilation_result.unwrap();
        let propagation_kernel = Kernel::create(&propagation_program, "PROPAGATO_KERNEL")?;

        self.propagation_program = Some(propagation_program);
        self.propagation_kernel = Some(propagation_kernel);

        Ok(())
    }

    #[allow(dead_code)]

    pub fn new(
        inputs_amount: usize,
        outputs_amount: usize,
        context: &'a Context,
        queue: &'a CommandQueue,
    ) -> Result<DenseGPU<'a>, ClError> {
        let mut rng = rand::thread_rng();

        let mut weights = (0..inputs_amount)
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
            ptr::null_mut()
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
            weights.par_iter().map(|x| x.to_vec()).flatten().collect::<Vec<f32>>().as_slice(), 
            &[]
        )?;
        let biases_gpu_write_event = queue.enqueue_write_buffer(
            &mut biases_buffer,
            CL_NON_BLOCKING, 
            0, 
            biases.as_slice(), 
            &[]
        )?;

        weights_gpu_write_event.wait()?;
        biases_gpu_write_event.wait()?;

        let propagation_program_compilation_result =
            Program::create_and_build_from_source(context, PROPAGATION_KERNEL_SORUCE, "");
        if propagation_program_compilation_result.is_err() {
            println!(
                "A compilation error was found in the DenseGPU PROPAGATION_KERNEL:\n{:?}",
                propagation_program_compilation_result.err().unwrap()
            );
            println!("Please report this issue at https://github.com/gabrielmfern/intricate");
            panic!();
        }
        let propagation_program = propagation_program_compilation_result.unwrap();
        let propagation_kernel = Kernel::create(&propagation_program, "PROPAGATO_KERNEL")?;

        Ok(DenseGPU {
            inputs_amount,
            outputs_amount,
            weights_buffer: Some(weights_buffer),
            biases_buffer: Some(biases_buffer),
            weights,
            biases,
            propagation_kernel: Some(propagation_kernel),
            propagation_program: Some(propagation_program),
            last_inputs_buffer: None,
            last_outputs_buffer: None,
            opencl_queue: Some(queue),
            opencl_context: Some(context)
        })
    }
}

impl<'a> OpenCLLayer<'a> for DenseGPU<'a> {
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

        self.last_inputs_buffer = Some(input_samples);

        let samples_amount = input_samples.size()? / mem::size_of::<cl_float>();

        let outputs_buffer = Buffer::<cl_float>::create(
            self.opencl_context.unwrap(),
            CL_MEM_READ_ONLY,
            self.outputs_amount * samples_amount,
            ptr::null_mut(),
        )?;

        let arg_inputs_amount: cl_int = self.inputs_amount as cl_int;

        let kernel_event = ExecuteKernel::new(self.propagation_kernel.as_ref().unwrap())
            .set_arg(input_samples)
            .set_arg(&self.biases_buffer)
            .set_arg(&self.weights_buffer)
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
        _should_calculate_input_to_error_derivative: bool,
        _layer_output_to_error_derivative: &Buffer<cl_float>,
        _learning_rate: cl_float,
    ) -> Result<Option<&Buffer<cl_float>>, ClError> {
        Err(ClError(-1))
        // assert!(!self.last_inputs.is_empty());
        // let samples_amount = layer_output_to_error_derivative.len();
        // let float_samples_amount = samples_amount as f32;
        //
        // // apply the gradients averaging the calculations between the samples
        // // but becomes extremely hard to calculate on very large neural networks
        // // with a large amount of samples to train on
        // self.weights = todo!();
        //
        // self.biases = todo!();
        //
        // if should_calculate_input_to_error_derivative {
        //     let layer_input_to_error_derivatives = todo!();
        //
        //     Some(layer_input_to_error_derivatives)
        // } else {
        //     None
        // }
    }
}
