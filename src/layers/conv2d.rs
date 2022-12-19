//! The module that defines the covolutional layer

use std::{collections::HashMap, mem};

use opencl3::{
    device::cl_float,
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    types::cl_int,
};
use rayon::prelude::*;
use savefile_derive::Savefile;

use crate::{
    optimizers::Optimizer,
    types::{ModelLayer, SyncDataError},
    utils::{
        find_divsor_of_n_closest_to_m, find_multiple_of_n_closest_to_m,
        opencl::{
            empty_buffer, ensure_program, BufferLike, BufferOperations,
            EnsureKernelsAndProgramError, InplaceBufferOperations,
        },
        OpenCLState,
    },
};

use super::{
    compute_update_vectors,
    initializers::{ConstantInitializer, GlorotUniformInitializer, Initializer, InitializerTrait},
    Gradient, Layer, LayerGradientApplicationError, LayerGradientComputationError,
    LayerInitializationError, LayerLossToInputDifferentiationError, LayerPropagationError,
    ParametersOptimizationError,
};

const CONV2D_PROGRAM_NAME: &str = "CONV2D";
const PROGRAM_SORUCE: &str = include_str!("kernels/conv2d.cl");

const PROPAGATION_KERNEL_NAME: &str = "convolute";
const COMPUTE_WEIGHT_GRADIENTS_KERNEL_NAME: &str = "compute_gradients_for_one_filter_pixel";
// const COMPUTE_BIAS_GRADIENTS_KERNEL_NAME: &str = "compute_gradients_for_biases";
const COMPUTE_LOSS_TO_INPUT_DERIVATIVES_KERNEL_NAME: &str = "compute_loss_to_input_derivatives";

pub(crate) fn compile_conv2d(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let prop_kernels = &[
        PROPAGATION_KERNEL_NAME.to_string(),
        COMPUTE_WEIGHT_GRADIENTS_KERNEL_NAME.to_string(),
        // COMPUTE_BIAS_GRADIENTS_KERNEL_NAME.to_string(),
        COMPUTE_LOSS_TO_INPUT_DERIVATIVES_KERNEL_NAME.to_string(),
    ];

    ensure_program(
        opencl_state,
        CONV2D_PROGRAM_NAME.to_string(),
        PROGRAM_SORUCE.to_string(),
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
/// A small caviat about this layer is that is uses **local work groups**
/// to pass the filter through the image, in a way that the size of the  local work group
/// in your GPU needs to be the at least the volume of your filter.
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
    pub weights: Vec<Vec<f32>>,

    /// This is a vec containing the biases for a pixel in the filter.
    pub biases: Vec<f32>,

    /// The initializer that will be used to generate the initial parameters for the filter's
    /// weights.
    pub initializers: HashMap<String, Initializer>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The allocated buffer with OpenCL that contains the flattened filter pixel weights.
    pub weights_buff: Option<Buffer<cl_float>>,

    #[savefile_ignore]
    #[savefile_introspect_ignore]
    /// The allocated buffer with OpenCL that contains the flattened filter biases.
    pub biases_buff: Option<Buffer<cl_float>>,

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
        let mut initializers = HashMap::with_capacity(2);
        initializers.insert(
            "weights".to_string(),
            GlorotUniformInitializer::new().into(),
        );
        initializers.insert("biases".to_string(), ConstantInitializer::new(0.0).into());

        Conv2D {
            inputs_size,
            filter_size,
            weights: Vec::default(),
            biases: Vec::default(),
            initializers,
            weights_buff: None,
            biases_buff: None,
            last_inputs_buffer: None,
            last_outputs_buffer: None,
            opencl_state: None,
        }
    }
}

impl<'a> Layer<'a> for Conv2D<'a> {
    fn get_flattened_parameter_data(&self, parameter: &str) -> Option<Vec<f32>> {
        match parameter {
            "weights" => {
                Some(self.weights.par_iter().flatten().map(|x| *x).collect())
            },
            "biases" => {
                Some(self.biases.to_vec())
            },
            _ => {
                None
            }
        }
    }

    fn get_initializer_for_parameter<'b>(&'b self, parameter: &str) -> Option<&'b Initializer> {
        self.initializers.get(parameter)
    }

    fn set_initializer_for_parameter(
        mut self,
        initializer: Initializer,
        parameter: &'a str,
    ) -> ModelLayer<'a> {
        self.initializers.insert(parameter.to_string(), initializer);
        self.into()
    }

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
        if self.weights_buff.is_some() {
            drop(self.weights_buff.as_ref().unwrap());
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

        if self.weights_buff.is_none() {
            return Err(SyncDataError::NotAllocatedInDevice {
                field_name: "weights".to_string(),
            });
        }

        if self.biases_buff.is_none() {
            return Err(SyncDataError::NotAllocatedInDevice {
                field_name: "biases".to_string(),
            });
        }

        let filter_weights_buffer = self.weights_buff.as_ref().unwrap();
        let biases_buffer = self.biases_buff.as_ref().unwrap();

        let filter_weights = Vec::<f32>::from_buffer(filter_weights_buffer, false, state)?;

        self.weights = (0..self.filter_size.1)
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

        let biases = Vec::<f32>::from_buffer(biases_buffer, false, state)?;

        self.biases = biases;

        Ok(())
    }

    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), LayerInitializationError> {
        if self.weights.is_empty() {
            if let Some(initializer) = self.get_initializer_for_parameter("weights") {
                self.weights = initializer.initialize_2d(self.filter_size, self);
            } else {
                return Err(LayerInitializationError::MissingParameterInitializer(
                    "weights",
                ));
            }
        }

        if self.biases.is_empty() {
            if let Some(initializer) = self.get_initializer_for_parameter("biases") {
                self.biases = initializer.initialize_1d(
                    // (
                    //     self.inputs_size.0 - self.filter_size.0 + 1, // the output width
                    //     self.inputs_size.1 - self.filter_size.1 + 1, // the output height
                    // ),
                    1,
                    self,
                );
            } else {
                return Err(LayerInitializationError::MissingParameterInitializer(
                    "biases",
                ));
            }
        }

        self.weights_buff = Some(
            self.weights
                .par_iter()
                .flatten()
                .map(|x| *x)
                .collect::<Vec<f32>>()
                .to_buffer(false, opencl_state)?,
        );

        self.biases_buff = Some(
            self.biases
                .to_buffer(false, opencl_state)?,
        );

        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn propagate(
        &mut self,
        inputs: &Buffer<cl_float>,
    ) -> Result<&Buffer<cl_float>, LayerPropagationError> {
        if self.opencl_state.is_none() {
            return Err(LayerPropagationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerPropagationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        let inputs_size = inputs.size()?;
        let inputs_volume = inputs_size / mem::size_of::<cl_float>();

        let image_volume = self.get_inputs_amount();
        let convolution_volume = self.get_outputs_amount();

        if inputs_volume % image_volume != 0 {
            return Err(LayerPropagationError::InputsDontMatchExpectedShape);
        }

        let samples_amount = inputs_volume / image_volume;

        self.last_inputs_buffer = Some(inputs.clone(state)?);
        let program = state.get_prgm(CONV2D_PROGRAM_NAME)?;
        let kernel = program.get_krnl(PROPAGATION_KERNEL_NAME)?;

        let outputs = empty_buffer(
            convolution_volume * samples_amount,
            CL_MEM_READ_WRITE,
            state,
        )?;

        let filter_volume = self.filter_size.0 * self.filter_size.1;

        let max_local_size = state.devices.first().unwrap().max_work_group_size()?;

        // can be like this because inside the kernel the threads that run with a glboal id above
        // the samples_amount stay inactive
        let samples_global_size = find_multiple_of_n_closest_to_m(max_local_size, samples_amount);

        ExecuteKernel::new(kernel)
            .set_arg(inputs)
            .set_arg(self.weights_buff.as_ref().unwrap())
            .set_arg(self.biases_buff.as_ref().unwrap())
            .set_arg(&outputs)
            // the max size for local workgroups has to fit the filter
            .set_arg_local_buffer(filter_volume * std::mem::size_of::<f32>())
            .set_arg(&(self.inputs_size.0 as cl_int))
            .set_arg(&(image_volume as cl_int))
            .set_arg(&(convolution_volume as cl_int))
            .set_arg(&(self.filter_size.0 as cl_int))
            .set_arg(&(self.filter_size.1 as cl_int))
            .set_arg(&(filter_volume as cl_int))
            .set_arg(&(samples_amount as cl_int))
            .set_global_work_sizes(&[
                samples_global_size,
                self.get_outputs_amount() * filter_volume,
            ])
            .set_local_work_sizes(&[
                // gcd(find_divsor_of_n_closest_to_m(max_local_size, max_local_size / filter_volume), samples_amount),
                find_divsor_of_n_closest_to_m(max_local_size, max_local_size / filter_volume),
                filter_volume,
            ])
            .enqueue_nd_range(queue)?
            .wait()?;

        self.last_outputs_buffer = Some(outputs);

        Ok(self.last_outputs_buffer.as_ref().unwrap())
    }

    fn compute_gradients(
        &self,
        layer_output_to_error_derivatives: &Buffer<cl_float>,
    ) -> Result<Vec<Gradient>, LayerGradientComputationError> {
        if self.opencl_state.is_none() {
            return Err(LayerGradientComputationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerGradientComputationError::NoCommandQueueFound);
        }

        if self.last_inputs_buffer.is_none() {
            return Err(LayerGradientComputationError::HasNotPropagatedBeforeCalculation);
        }

        let queue = state.queues.first().unwrap();

        let derivatives_size = layer_output_to_error_derivatives.size()?;
        let derivatives_volume = derivatives_size / mem::size_of::<cl_float>();

        let image_volume = self.get_inputs_amount();
        let convolution_volume = self.get_outputs_amount();

        if derivatives_volume % convolution_volume != 0 {
            return Err(LayerGradientComputationError::DerivativesDontMatchExpectedShape);
        }

        let samples_amount = derivatives_volume / convolution_volume;

        let filter_volume = self.filter_size.0 * self.filter_size.1;

        let mut weights_gradients: Vec<f32> = Vec::with_capacity(filter_volume);

        let program = state.get_prgm(CONV2D_PROGRAM_NAME)?;
        let compute_gradient_weights_kernel =
            program.get_krnl(COMPUTE_WEIGHT_GRADIENTS_KERNEL_NAME)?;
        // let compute_gradient_biases_kernel =
        //     program.get_krnl(COMPUTE_BIAS_GRADIENTS_KERNEL_NAME)?;

        for pixel_index in 0..filter_volume {
            let filter_pixel_gradients = empty_buffer(
                convolution_volume * samples_amount,
                CL_MEM_READ_WRITE,
                state,
            )?;

            let pixel_y =
                ((pixel_index + 1) as f32 / self.filter_size.0 as f32).ceil() as usize - 1;
            let pixel_x = pixel_index % self.filter_size.0;

            let convolution_width = self.inputs_size.0 - self.filter_size.0 + 1;
            let convolution_height = self.inputs_size.1 - self.filter_size.1 + 1;

            ExecuteKernel::new(compute_gradient_weights_kernel)
                .set_arg(self.last_inputs_buffer.as_ref().unwrap())
                .set_arg(layer_output_to_error_derivatives)
                .set_arg(&filter_pixel_gradients)
                .set_arg(&(self.inputs_size.0 as cl_int))
                .set_arg(&(image_volume as cl_int))
                .set_arg(&(self.filter_size.0 as cl_int))
                .set_arg(&(filter_volume as cl_int))
                .set_arg(&(samples_amount.clone() as cl_int))
                .set_arg(&(convolution_width as cl_int))
                .set_arg(&(convolution_height as cl_int))
                .set_arg(&(convolution_volume as cl_int))
                .set_arg(&(pixel_index as cl_int))
                .set_arg(&(pixel_y as cl_int))
                .set_arg(&(pixel_x as cl_int))
                .set_global_work_sizes(&[samples_amount, convolution_height, convolution_width])
                .enqueue_nd_range(queue)?
                .wait()?;

            weights_gradients.push(filter_pixel_gradients.sum(state)? / samples_amount as f32);
        }

        let bias_gradients = layer_output_to_error_derivatives.sum(state)? / samples_amount as f32;
        // empty_buffer(self.get_outputs_amount(), CL_MEM_READ_WRITE, state)?;

        // ExecuteKernel::new(compute_gradient_biases_kernel)
        //     .set_arg(layer_output_to_error_derivatives)
        //     .set_arg(&bias_gradients)
        //     .set_arg(&(samples_amount as cl_int))
        //     .set_arg(&(convolution_volume as cl_int))
        //     .set_global_work_size(convolution_volume)
        //     .enqueue_nd_range(queue)?
        //     .wait()?;

        queue.finish()?;

        Ok(vec![
            Gradient {
                optimizable: true,
                parameter_id: "weights".to_string(),
                value: weights_gradients.to_buffer(false, state)?,
            },
            Gradient {
                optimizable: true,
                parameter_id: "biases".to_string(),
                value: vec![bias_gradients].to_buffer(false, state)?,
            },
        ])
    }

    fn optimize_parameters(
        &mut self,
        optimizer: &dyn crate::optimizers::Optimizer<'a>,
        layer_index: usize,
        timestep: usize,
    ) -> Result<(), ParametersOptimizationError> {
        if self.weights_buff.is_none() {
            return Err(ParametersOptimizationError::EmptyParameter(
                "weights".to_string(),
            ));
        }

        if self.biases_buff.is_none() {
            return Err(ParametersOptimizationError::EmptyParameter(
                "biases".to_string(),
            ));
        }

        optimizer.optimize_parameters(
            self.weights_buff.as_mut().unwrap(),
            "weights".to_string(),
            timestep,
            layer_index,
        )?;

        optimizer.optimize_parameters(
            self.biases_buff.as_mut().unwrap(),
            "biases".to_string(),
            timestep,
            layer_index,
        )?;

        Ok(())
    }

    fn apply_gradients(
        &mut self,
        per_parameter_type_gradients: &[Gradient],
        optimizer: &mut dyn Optimizer<'a>,
        layer_model_index: usize,
        timestep: usize,
    ) -> Result<(), LayerGradientApplicationError> {
        if self.opencl_state.is_none() {
            return Err(LayerGradientApplicationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if per_parameter_type_gradients.len() != 2 {
            return Err(LayerGradientApplicationError::GradientsDontMatchExpectedShape);
        }

        let update_vectors = compute_update_vectors(
            optimizer,
            per_parameter_type_gradients,
            layer_model_index,
            timestep,
            state,
        )?;

        let weights_buff = self.weights_buff.as_mut().unwrap();
        weights_buff.subtract_inplc(&update_vectors[0], state)?;
        let biases_buff = self.biases_buff.as_mut().unwrap();
        biases_buff.subtract_inplc(&update_vectors[1], state)?;

        Ok(())
    }

    fn compute_loss_to_input_derivatives(
        &self,
        layer_output_to_error_derivative: &Buffer<cl_float>,
    ) -> Result<Buffer<cl_float>, LayerLossToInputDifferentiationError> {
        if self.opencl_state.is_none() {
            return Err(LayerLossToInputDifferentiationError::LayerNotInitialized);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.first().is_none() {
            return Err(LayerLossToInputDifferentiationError::NoCommandQueueFound);
        }

        if self.last_inputs_buffer.is_none() {
            return Err(LayerLossToInputDifferentiationError::HasNotPropagatedBeforeCalculation);
        }

        let queue = state.queues.first().unwrap();

        let derivatives_size = layer_output_to_error_derivative.size()?;
        let derivatives_volume = derivatives_size / mem::size_of::<cl_float>();

        let image_volume = self.get_inputs_amount();
        let convolution_volume = self.get_outputs_amount();

        if derivatives_volume % convolution_volume != 0 {
            return Err(LayerLossToInputDifferentiationError::DerivativesDontMatchExpectedShape);
        }

        if self.weights_buff.is_none() {
            return Err(LayerLossToInputDifferentiationError::MissingParameter(
                "weights",
            ));
        }

        if self.biases_buff.is_none() {
            return Err(LayerLossToInputDifferentiationError::MissingParameter(
                "biases",
            ));
        }

        let samples_amount = derivatives_volume / convolution_volume;
        let filter_width = self.filter_size.0;
        let filter_height = self.filter_size.1;
        let outputs_width = self.inputs_size.0 - self.filter_size.0 + 1;
        let outputs_height = self.inputs_size.1 - self.filter_size.1 + 1;

        let program = state.get_prgm(CONV2D_PROGRAM_NAME)?;
        let kernel = program.get_krnl(COMPUTE_LOSS_TO_INPUT_DERIVATIVES_KERNEL_NAME)?;

        let loss_to_input_derivatives_buffer = empty_buffer(
            self.get_inputs_amount() * samples_amount,
            CL_MEM_READ_WRITE,
            state,
        )?;

        ExecuteKernel::new(kernel)
            .set_arg(self.weights_buff.as_ref().unwrap())
            .set_arg(layer_output_to_error_derivative)
            .set_arg(&loss_to_input_derivatives_buffer)
            .set_arg(&(samples_amount as cl_int))
            .set_arg(&(filter_width as cl_int))
            .set_arg(&(filter_height as cl_int))
            .set_arg(&(outputs_height as cl_int))
            .set_arg(&(outputs_width as cl_int))
            .set_arg(&(image_volume as cl_int))
            .set_arg(&(self.inputs_size.0 as cl_int))
            .set_global_work_sizes(&[samples_amount, image_volume])
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        Ok(loss_to_input_derivatives_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::Conv2D;
    use crate::{
        layers::Layer,
        utils::{
            approx_eq,
            opencl::{BufferLike, DeviceType},
            setup_opencl,
        },
    };

    // #[test]
    // fn should_compute_loss_to_input_derivatives_correctly() -> () {
    //     let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to eetup opencl");
    //     let images = vec![
    //         0.1,  0.3,  0.4,   0.9,
    //         0.23, 0.29, 0.34, 0.15,
    //         0.93, 0.31, 0.11, 0.44,
    //         0.15, 0.14, 0.19, 0.32,

    //         0.45, 0.21, 0.42,  0.2,
    //         0.12, 0.23, 0.21, 0.31,
    //         0.86, 0.28, 0.25, 0.83,
    //         0.25, 0.11, 0.64, 0.33,
    //     ].to_buffer(false, &opencl_state).expect("unable to create images buffer");
    //     let filter = vec![
    //         0.51, 0.94, 0.12,
    //         0.31, 0.44, 0.99,
    //         0.11, 0.05, 0.31
    //     ].to_buffer(false, &opencl_state).expect("unable to create filter buffer");
    //     let loss_to_output_derivatives = vec![
    //         0.1, 0.3,
    //         0.4, 0.8,

    //         0.2, 0.4,
    //         0.5, 0.5,
    //     ].to_buffer(false, &opencl_state).expect("unable to create loss to output derivatives buffer");
    //     let expected_loss_to_input_derivatives: Vec<f32> = vec![
    //         0.51 * 0.1,
    //         0.51 * 0.3 + 0.94 * 0.1,
    //         0.94 * 0.3 + 0.12 * 0.1,
    //         0.12 * 0.3,
    //         0.31 * 0.1 + 0.51 * 0.4,
    //         0.44 * 0.1 + 0.31 * 0.3 + 0.94 * 0.4 + 0.51 * 0.8,
    //         0.99 * 0.1 + 0.44 * 0.3 + 0.12 * 0.4 + 0.94 * 0.8,
    //     ];
    // }

    #[test]
    fn should_compute_gradients_correctly() -> () {
        let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to setup opencl");
        let images: Vec<f32> = vec![
            0.1, 0.3, 0.4, 0.9, 0.23, 0.29, 0.34, 0.15, 0.93, 0.31, 0.11, 0.44, 0.15, 0.14, 0.19,
            0.32, 0.45, 0.21, 0.42, 0.2, 0.12, 0.23, 0.21, 0.31, 0.86, 0.28, 0.25, 0.83, 0.25,
            0.11, 0.64, 0.33,
        ];
        let output_to_loss_derivatives = vec![0.1, 0.3, 0.4, 0.8, 0.2, 0.4, 0.5, 0.5];
        let expected_gradients = vec![
            (0.1 * 0.1
                + 0.3 * 0.3
                + 0.23 * 0.4
                + 0.29 * 0.8
                + 0.45 * 0.2
                + 0.21 * 0.4
                + 0.12 * 0.5
                + 0.23 * 0.5)
                / 2.0,
            (0.3 * 0.1
                + 0.4 * 0.3
                + 0.29 * 0.4
                + 0.34 * 0.8
                + 0.21 * 0.2
                + 0.42 * 0.4
                + 0.23 * 0.5
                + 0.21 * 0.5)
                / 2.0,
            (0.4 * 0.1
                + 0.9 * 0.3
                + 0.34 * 0.4
                + 0.15 * 0.8
                + 0.42 * 0.2
                + 0.2 * 0.4
                + 0.21 * 0.5
                + 0.31 * 0.5)
                / 2.0,
            (0.23 * 0.1
                + 0.29 * 0.3
                + 0.93 * 0.4
                + 0.31 * 0.8
                + 0.12 * 0.2
                + 0.23 * 0.4
                + 0.86 * 0.5
                + 0.28 * 0.5)
                / 2.0,
            (0.29 * 0.1
                + 0.34 * 0.3
                + 0.31 * 0.4
                + 0.11 * 0.8
                + 0.23 * 0.2
                + 0.21 * 0.4
                + 0.28 * 0.5
                + 0.25 * 0.5)
                / 2.0,
            (0.34 * 0.1
                + 0.15 * 0.3
                + 0.11 * 0.4
                + 0.44 * 0.8
                + 0.21 * 0.2
                + 0.31 * 0.4
                + 0.25 * 0.5
                + 0.83 * 0.5)
                / 2.0,
            (0.93 * 0.1
                + 0.31 * 0.3
                + 0.15 * 0.4
                + 0.14 * 0.8
                + 0.86 * 0.2
                + 0.28 * 0.4
                + 0.25 * 0.5
                + 0.11 * 0.5)
                / 2.0,
            (0.31 * 0.1
                + 0.11 * 0.3
                + 0.14 * 0.4
                + 0.19 * 0.8
                + 0.28 * 0.2
                + 0.25 * 0.4
                + 0.11 * 0.5
                + 0.64 * 0.5)
                / 2.0,
            (0.11 * 0.1
                + 0.44 * 0.3
                + 0.19 * 0.4
                + 0.32 * 0.8
                + 0.25 * 0.2
                + 0.83 * 0.4
                + 0.64 * 0.5
                + 0.33 * 0.5)
                / 2.0,
        ];

        let samples_buff = images
            .to_buffer(false, &opencl_state)
            .expect("unable to get the image's buffer");
        let output_to_loss_derivatives_buff = output_to_loss_derivatives
            .to_buffer(false, &opencl_state)
            .expect("unable to get the output to loss derivatives buffer");

        let mut conv2d = Conv2D::new_raw((4, 4), (3, 3));
        conv2d
            .init(&opencl_state)
            .expect("unable to initialize raw conv2D layer");

        conv2d.last_inputs_buffer = Some(samples_buff);

        let actual_gradients_buff = &conv2d
            .compute_gradients(&output_to_loss_derivatives_buff)
            .expect("unable to compute conv2d gradients")[0]
            .value;

        let actual_gradients = Vec::<f32>::from_buffer(actual_gradients_buff, false, &opencl_state)
            .expect("unable to convert from the actual gradients buffer to a vector");

        approx_eq::assert_approx_equal(&actual_gradients, &expected_gradients, 1);
    }

    #[test]
    fn should_convolute_correctly() -> () {
        let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to setup opencl");
        let image = vec![
            0.33, 0.14, 0.99, 1.0, 0.51, 0.32, 0.91, 0.1, 0.8, 0.4, 0.5, 0.2, 0.33, 0.14, 0.99,
            1.0, 0.51, 0.32, 0.91, 0.1, 0.8, 0.4, 0.5, 0.2,
        ]
        .to_buffer(false, &opencl_state)
        .expect("unable to get image buffer");
        let filter = vec![0.3, 0.4, 0.9, 0.1, 0.2, 1.0, 0.2, 0.5, 0.81]
            .to_buffer(false, &opencl_state)
            .expect("unable to get filter buffer");
        let convolution = vec![
            0.33 * 0.3
                + 0.14 * 0.4
                + 0.99 * 0.9
                + 0.51 * 0.1
                + 0.32 * 0.2
                + 0.91 * 1.0
                + 0.8 * 0.2
                + 0.4 * 0.5
                + 0.5 * 0.81,
            0.14 * 0.3
                + 0.99 * 0.4
                + 1.0 * 0.9
                + 0.32 * 0.1
                + 0.91 * 0.2
                + 0.1 * 1.0
                + 0.4 * 0.2
                + 0.5 * 0.5
                + 0.2 * 0.81,
            0.33 * 0.3
                + 0.14 * 0.4
                + 0.99 * 0.9
                + 0.51 * 0.1
                + 0.32 * 0.2
                + 0.91 * 1.0
                + 0.8 * 0.2
                + 0.4 * 0.5
                + 0.5 * 0.81,
            0.14 * 0.3
                + 0.99 * 0.4
                + 1.0 * 0.9
                + 0.32 * 0.1
                + 0.91 * 0.2
                + 0.1 * 1.0
                + 0.4 * 0.2
                + 0.5 * 0.5
                + 0.2 * 0.81,
        ];

        let mut layer = Conv2D::new_raw((4, 3), (3, 3));
        layer.init(&opencl_state).expect("unable to init Conv2D");
        layer.weights_buff = Some(filter);

        let result_buffer = layer
            .propagate(&image)
            .expect("unable to propagate conv2d layer");
        let result = Vec::<f32>::from_buffer(result_buffer, false, &opencl_state)
            .expect("unable to get resulting convolution buffer");

        assert_eq!(result, convolution);
    }
}
