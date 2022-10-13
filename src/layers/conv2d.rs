//! The module that defines the covolutional layer

use std::mem;

use opencl3::{
    device::cl_float,
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE}, types::cl_int,
};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use savefile_derive::Savefile;

use crate::{
    types::{ModelLayer, SyncDataError},
    utils::{
        opencl::{ensure_program, BufferLike, BufferOperations, EnsureKernelsAndProgramError},
        OpenCLState,
    },
};

use super::{Layer, LayerInitializationError, LayerPropagationError};

const CONV2D_PROGRAM_NAME: &str = "CONV2D_PROPAGATION";
const PROPAGATION_PROGRAM_SORUCE: &str = include_str!("kernels/conv2d_propagation.cl");

const PROPAGATION_KERNEL_NAME: &str = "convolute";

pub(crate) fn compile_conv2d(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let prop_kernels = &[PROPAGATION_KERNEL_NAME.to_string()];

    ensure_program(
        opencl_state,
        CONV2D_PROGRAM_NAME.to_string(),
        PROPAGATION_PROGRAM_SORUCE.to_string(),
        "".to_string(),
        prop_kernels,
    )?;

    Ok(())
}

#[derive(Debug, Savefile)]
/// A layer that tries to compact data from a 2D image, or just a matrix,
/// without loosing spatial information. It does this by passing a filter from
/// side-to-side in the image multiplying it based on the filter's weights.
///
/// This makes it so that the size of the image gets much more increased
/// based on the size of the filter without loosing information.
///
/// This type of layer proves to be extremely useful when working with images,
/// as it makes both the model much more lightweight as well as it makes it
/// perform much, much better.
///
/// # Examples
///
/// ```rust
/// use intricate::layers::Conv2D;
///
/// // this will make a conv layer that will go through a 28x28 image
/// // with a 3x3 filter
/// let my_layer: Conv2D = Conv2D::new_raw((28, 28), (3, 3));
/// ```
pub struct Conv2D<'a> {
    /// The size of the inputs, width and height respectively.
    pub inputs_size: (usize, usize),
    /// The size of the filter, width and height respectively.
    pub filter_size: (usize, usize),

    /// This is a vec containing the certain weight for a pixel in the filter.
    /// This a vec that contains rows instead of columns.
    pub filter_pixel_weights: Vec<Vec<f32>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The allocated buffer with OpenCL that contains the flattened filter pixel weights.
    pub filter_pixel_weights_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The buffer that contains the flattened inputs per sample that were last forwad passed into
    /// this Dense layer.
    pub last_inputs_buffer: Option<Buffer<cl_float>>,
    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The buffer that contains the flattened outputs per sample that last came out of a forward
    /// pass into this Dense layer.
    pub last_outputs_buffer: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> Conv2D<'a> {
    /// Creates a new 2D Convolutional layer with a random filter ready for being used
    /// in a Model.
    pub fn new(inputs_size: (usize, usize), filter_size: (usize, usize)) -> ModelLayer<'a> {
        Self::new_raw(inputs_size, filter_size).into()
    }

    /// Crates a new raw 2D Convolutional layer with a random filter.
    pub fn new_raw(inputs_size: (usize, usize), filter_size: (usize, usize)) -> Self {
        let mut rng = thread_rng();
        Conv2D {
            inputs_size,
            filter_size,
            filter_pixel_weights: (0..filter_size.1)
                .map(|_| {
                    (0..filter_size.0)
                        .map(|_| rng.gen_range(-1f32..1f32))
                        .collect()
                })
                .collect(),
            filter_pixel_weights_buffer: None,
            last_inputs_buffer: None,
            last_outputs_buffer: None,
            opencl_state: None,
        }
    }
}

impl<'a> Layer<'a> for Conv2D<'a> {
    fn get_last_inputs(&self) -> Option<&Buffer<cl_float>> {
        self.last_inputs_buffer.as_ref()
    }

    fn get_last_outputs(&self) -> Option<&Buffer<cl_float>> {
        self.last_outputs_buffer.as_ref()
    }

    fn get_inputs_amount(&self) -> usize {
        self.inputs_size.0 * self.inputs_size.1
    }

    fn get_outputs_amount(&self) -> usize {
        (self.inputs_size.0 - self.filter_size.0 + 1)
            * (self.inputs_size.1 - self.filter_size.1 + 1)
    }

    fn clean_up_gpu_state(&mut self) -> () {
        if self.filter_pixel_weights_buffer.is_some() {
            drop(self.filter_pixel_weights_buffer.as_ref().unwrap());
        }

        if self.last_inputs_buffer.is_some() {
            drop(self.last_inputs_buffer.as_ref().unwrap());
        }

        if self.last_outputs_buffer.is_some() {
            drop(self.last_outputs_buffer.as_ref().unwrap());
        }
    }

    fn sync_data_from_buffers_to_host(&mut self) -> Result<(), SyncDataError> {
        if self.opencl_state.is_none() {
            return Err(SyncDataError::NotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(SyncDataError::NoCommandQueue);
        }

        if self.filter_pixel_weights_buffer.is_none() {
            return Err(SyncDataError::NotAllocatedInDevice {
                field_name: "filter_pixel_weights_buffer".to_string(),
            });
        }

        let filter_weights_buffer = self.filter_pixel_weights_buffer.as_ref().unwrap();

        let filter_weights = Vec::<f32>::from_buffer(filter_weights_buffer, false, state)?;

        self.filter_pixel_weights = (0..self.filter_size.1)
            .into_par_iter()
            .map(|y| {
                (0..self.filter_size.0)
                    .map(|x| {
                        let flat_index = y * self.filter_size.0 + x;
                        filter_weights[flat_index]
                    })
                    .collect()
            })
            .collect();

        Ok(())
    }

    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), LayerInitializationError> {
        if self.filter_pixel_weights.is_empty() {
            return Err(LayerInitializationError::EmptyParameter(
                "filter_pixel_weights".to_string(),
            ));
        }

        self.filter_pixel_weights_buffer = Some(
            self.filter_pixel_weights
                .par_iter()
                .flatten()
                .map(|x| *x)
                .collect::<Vec<f32>>()
                .to_buffer(false, opencl_state)?,
        );

        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn propagate(
        &mut self,
        image: &Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
        if self.opencl_state.is_none() {
            return Err(LayerPropagationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerPropagationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();
        let context = &state.context;

        let image_size = image.size()?;
        let image_volume = image_size / mem::size_of::<cl_float>();

        if image_volume % self.get_inputs_amount() != 0 {
            return Err(LayerPropagationError::InputsDontMatchExpectedShape);
        }

        if image_volume / self.inputs_size.0 != self.inputs_size.1 
        || image_volume / self.inputs_size.1 != self.inputs_size.0 {
            return Err(LayerPropagationError::InputsDontMatchExpectedShape);
        }

        self.last_inputs_buffer = Some(image.clone(state)?);

        let program = state.get_prgm(CONV2D_PROGRAM_NAME)?;
        let kernel = program.get_krnl(PROPAGATION_KERNEL_NAME)?;

        let convolution = Buffer::create(
            &state.context,
            CL_MEM_READ_WRITE,
            self.get_outputs_amount(),
            std::ptr::null_mut(),
        )?;
        
        let filter_volume = self.filter_size.0 * self.filter_size.1;

        ExecuteKernel::new(kernel)
            .set_arg(image)
            .set_arg(self.filter_pixel_weights_buffer.as_ref().unwrap())
            .set_arg(&convolution) 
            // the max size for local workgroups has to fit the filter
            .set_arg_local_buffer(filter_volume)
            .set_arg(&(self.inputs_size.0 as cl_int))
            .set_arg(&(self.filter_size.0 as cl_int))
            .set_arg(&(self.filter_size.1 as cl_int))
            .set_arg(&(filter_volume as cl_int))
            .set_global_work_size(self.get_outputs_amount() * filter_volume)
            .set_local_work_size(filter_volume)
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        self.last_outputs_buffer = Some(convolution);

        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn compute_gradients(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Vec<super::Gradient>, super::LayerGradientComputationError> {
        todo!()
    }

    fn optimize_parameters(
        &mut self,
        optimizer: &dyn crate::optimizers::Optimizer<'a>,
        layer_index: usize,
    ) -> Result<(), super::ParametersOptimizationError> {
        todo!()
    }

    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: &[super::Gradient],
        optimizer: &mut dyn crate::optimizers::Optimizer<'a>,
        layer_model_index: usize,
    ) -> Result<(), super::LayerGradientApplicationError> {
        todo!()
    }

    fn compute_loss_to_input_derivatives(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, super::LayerLossToInputDifferentiationError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::Conv2D;
    use crate::{layers::Layer, utils::{setup_opencl, opencl::{DeviceType, BufferLike}}};

    #[test]
    fn should_convolute_correctly() -> () {
        let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to setup opencl");
        let image = vec![
            0.33f32, 0.14, 0.99, 1.0,
            0.51, 0.32, 0.91, 0.1,
            0.8, 0.4, 0.5, 0.2,
        ].to_buffer(false, &opencl_state).expect("unable to get image buffer");
        let filter = vec![
            0.3, 0.4, 0.9,
            0.1, 0.2, 1.0,
            0.2, 0.5, 0.81
        ].to_buffer(false, &opencl_state).expect("unable to get filter buffer");
        let convolution = vec![
            0.33 * 0.3 + 0.14 * 0.4 + 0.99 * 0.9 
          + 0.51 * 0.1 + 0.32 * 0.2 + 0.91 * 1.0 
          + 0.8 * 0.2 + 0.4 * 0.5 + 0.5 * 0.81,

            0.14 * 0.3 + 0.99 * 0.4 + 1.0 * 0.9 
          + 0.32 * 0.1 + 0.91 * 0.2 + 0.1 * 1.0
          + 0.4 * 0.2 + 0.5 * 0.5 + 0.2 * 0.81,
        ];

        let mut layer = Conv2D::new_raw((4, 3), (3, 3));
        layer.init(&opencl_state).expect("unable to init Conv2D");
        layer.filter_pixel_weights_buffer = Some(filter);

        let result_buffer = layer.propagate(&image).expect("unable to propagate conv2d layer");
        let result = Vec::<f32>::from_buffer(result_buffer, false, &opencl_state)
            .expect("unable to get resulting convolution buffer");

        assert_eq!(result, convolution);
    }
}