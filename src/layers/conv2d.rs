//! The module that defines the covolutional layer

// TODO: Add stride, padding and improve gradient calculation that is tremendously slow rn

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
        opencl::{
            empty_buffer, ensure_program, opencl_state::EnsureKernelsAndProgramError, BufferLike,
            BufferOperations, InplaceBufferOperations,
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

const COMPUTE_WEIGHT_GRADIENTS_KERNEL_NAME: &str = "compute_gradients_per_sample";
// const COMPUTE_BIAS_GRADIENTS_KERNEL_NAME: &str = "compute_gradients_for_biases";
const COMPUTE_LOSS_TO_INPUT_DERIVATIVES_KERNEL_NAME: &str = "compute_loss_to_input_derivatives";

pub(crate) fn compile_conv2d(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    let prop_kernels = &[
        COMPUTE_WEIGHT_GRADIENTS_KERNEL_NAME,
        // COMPUTE_BIAS_GRADIENTS_KERNEL_NAME,
        COMPUTE_LOSS_TO_INPUT_DERIVATIVES_KERNEL_NAME,
    ];

    ensure_program(
        opencl_state,
        CONV2D_PROGRAM_NAME,
        PROGRAM_SORUCE,
        "",
        prop_kernels,
    )?;

    Ok(())
}

#[derive(Debug, Savefile)]
/// A layer that tries to compact data from a 2D image, or just a matrix,
/// without loosing spatial information. It does this by passing a specified amount of filters from
/// side-to-side in the image multiplying it based on the filter's weights.
///
/// This makes it so that the size of the image gets much more increased
/// based on the size of the filter without loosing information.
///
/// This type of layer proves to be extremely useful when working with images,
/// as it makes both the model much more memory efficient and does make the model
/// perform much, much better. (take a look at the `MNIST` example)
///
/// # Examples
///
/// ```rust
/// use intricate::layers::Conv2D;
///
/// // this will make a conv layer that will go through a 28x28 image
/// // with a 3x3 filter
/// let my_layer: Conv2D = Conv2D::new_raw((28, 28), (3, 3), 1);
/// ```
pub struct Conv2D<'a> {
    /// The size of the inputs, width and height respectively.
    pub inputs_size: (usize, usize),
    /// The size of the filter, width and height respectively.
    pub filter_sizes: (usize, usize),
    /// The amount of filters that are going to contain possible features of certain types of
    /// matrices.
    pub filters_amount: usize,

    /// This is a vec containing the certain weight for a pixel in one of the filters.
    pub weights: Vec<Vec<Vec<f32>>>,

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
    pub fn new(
        inputs_size: (usize, usize),
        filter_sizes: (usize, usize),
        filters_amount: usize,
    ) -> ModelLayer<'a> {
        Self::new_raw(inputs_size, filter_sizes, filters_amount).into()
    }

    /// Crates a new raw 2D Convolutional layer with a random filter.
    pub fn new_raw(
        inputs_size: (usize, usize),
        filter_sizes: (usize, usize),
        filters_amount: usize,
    ) -> Self {
        let mut initializers = HashMap::with_capacity(2);
        initializers.insert(
            "weights".to_string(),
            GlorotUniformInitializer::new().into(),
        );
        initializers.insert("biases".to_string(), ConstantInitializer::new(0.0).into());

        Conv2D {
            inputs_size,
            filter_sizes,
            filters_amount,
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
            "weights" => Some(
                self.weights
                    .par_iter()
                    .flatten()
                    .flatten()
                    .map(|x| *x)
                    .collect(),
            ),
            "biases" => Some(self.biases.to_vec()),
            _ => None,
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
        (self.inputs_size.0 - self.filter_sizes.0 + 1)
            * (self.inputs_size.1 - self.filter_sizes.1 + 1)
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

        let filter_volume = self.filter_sizes.0 * self.filter_sizes.1;

        self.weights = (0..self.filters_amount)
            .into_par_iter()
            .map(|z| {
                (0..self.filter_sizes.1)
                    .map(|y| {
                        (0..self.filter_sizes.0)
                            .map(|x| {
                                let flat_index = z * filter_volume + y * self.filter_sizes.0 + x;
                                filter_weights[flat_index]
                            })
                            .collect()
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
                self.weights = initializer.initialize_3d(
                    (
                        self.filters_amount,
                        self.filter_sizes.1,
                        self.filter_sizes.0,
                    ),
                    self,
                );
            } else {
                return Err(LayerInitializationError::MissingParameterInitializer(
                    "weights",
                ));
            }
        }

        if self.biases.is_empty() {
            if let Some(initializer) = self.get_initializer_for_parameter("biases") {
                self.biases = initializer.initialize_1d(self.filters_amount, self);
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
                .flatten()
                .map(|x| *x)
                .collect::<Vec<f32>>()
                .to_buffer(false, opencl_state)?,
        );

        self.biases_buff = Some(self.biases.to_buffer(false, opencl_state)?);

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

        let inputs_size = inputs.size()?;
        let inputs_volume = inputs_size / mem::size_of::<cl_float>();

        let image_volume = self.get_inputs_amount();

        if inputs_volume % image_volume != 0 {
            return Err(LayerPropagationError::InputsDontMatchExpectedShape);
        }

        self.last_inputs_buffer = Some(inputs.clone(state)?);

        if self.weights_buff.is_none() {
            return Err(LayerPropagationError::LayerNotInitialized);
        }

        let filter_weights = self.weights_buff.as_ref().unwrap();

        let padded_width = self.inputs_size.0 + self.filter_sizes.0 - 1;
        let padded_height = self.inputs_size.1 + self.filter_sizes.1 - 1;
        let convolution_width = self.inputs_size.0 - self.filter_sizes.0 + 1;
        let convolution_height = self.inputs_size.1 - self.filter_sizes.1 + 1;
        let x_range_start = (padded_width - convolution_width) / 2;
        let y_range_start = (padded_height - convolution_height) / 2;
        let x_range = x_range_start..(x_range_start + convolution_width - 1);
        let y_range = y_range_start..(y_range_start + convolution_height - 1);

        let convolution = inputs.convolve_2d(
            &state,
            &filter_weights,
            self.inputs_size.0,
            self.inputs_size.1,
            self.filter_sizes.0,
            self.filter_sizes.1,
            (x_range, y_range),
        )?;

        self.last_outputs_buffer = Some(convolution);

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

        let convolution_volume = self.get_outputs_amount();

        if derivatives_volume % convolution_volume != 0 {
            return Err(LayerGradientComputationError::DerivativesDontMatchExpectedShape);
        }

        let filter_volume = self.filter_sizes.0 * self.filter_sizes.1;

        let samples_amount = derivatives_volume / convolution_volume;

        let convolution_width = self.inputs_size.0 - self.filter_sizes.0 + 1;
        let convolution_height = self.inputs_size.1 - self.filter_sizes.1 + 1;

        let padded_width = self.inputs_size.0 + convolution_width - 1;
        let padded_height = self.inputs_size.1 + convolution_height - 1;

        let x_range_start = (padded_width - self.filter_sizes.0) / 2;
        let y_range_start = (padded_height - self.filter_sizes.1) / 2;
        let x_range = x_range_start..(x_range_start + self.filter_sizes.0 - 1);
        let y_range = y_range_start..(y_range_start + self.filter_sizes.1 - 1);

        let inputs = self.get_last_inputs().unwrap();
        let mut weight_gradients = inputs
            .sampled_convolve_2d(
                state,
                &layer_output_to_error_derivatives,
                self.inputs_size.0,
                self.inputs_size.1,
                convolution_width,
                convolution_height,
                (x_range, y_range),
            )?
            .tranpose(state, filter_volume, samples_amount)?
            .sum_2d_per_row(state, samples_amount)?;
        weight_gradients.scale_inplc(1.0 / samples_amount as f32, state)?;

        let mut bias_gradients = layer_output_to_error_derivatives
            .tranpose(
                state,
                convolution_volume * self.filters_amount,
                samples_amount,
            )?
            .sum_2d_per_row(state, samples_amount)?;
        bias_gradients.scale_inplc(1.0 / samples_amount as f32, state)?;

        queue.finish()?;

        Ok(vec![
            Gradient {
                optimizable: true,
                parameter_id: "weights".to_string(),
                value: weight_gradients,
            },
            Gradient {
                optimizable: true,
                parameter_id: "biases".to_string(),
                value: bias_gradients,
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
        let filter_width = self.filter_sizes.0;
        let filter_height = self.filter_sizes.1;
        let outputs_width = self.inputs_size.0 - self.filter_sizes.0 + 1;
        let outputs_height = self.inputs_size.1 - self.filter_sizes.1 + 1;

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
            approx_eq::{self, assert_approx_equal_distance},
            opencl::{BufferLike, DeviceType},
            setup_opencl,
        },
    };

    #[test]
    fn should_compute_gradients_correctly() -> () {
        let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to setup opencl");
        let image = vec![
            0.33, 0.14, 0.99, 1.0, 0.1, 
            0.51, 0.31, 0.91, 0.1, 0.3, 
            0.8,  0.4,  0.5,  0.2, 0.1,

            0.31, 0.41, 2.32, 1.2, 0.41,
            0.91, 0.32, 0.93, 0.12,0.34,
            0.3,  0.2,  0.91, 0.41,2.3
        ]
        .to_buffer(false, &opencl_state)
        .expect("unable to get image buffer");

        let filter = vec![0.3, 0.4, 0.9, 0.1, 0.2, 1.0, 0.2, 0.5, 0.81]
            .to_buffer(false, &opencl_state)
            .expect("unable to get filter buffer");
        let bias = vec![0.123]
            .to_buffer(false, &opencl_state)
            .expect("unable to get the biases buffer");
        let expected_result = vec![
          4.3590 / 2.0, 5.0604 / 2.0, 5.5927 / 2.0,
          3.3297 / 2.0, 2.0929 / 2.0, 2.4738 / 2.0,
          2.6763 / 2.0, 2.2349 / 2.0, 4.9649 / 2.0
        ];

        let mut layer = Conv2D::new_raw((5, 3), (3, 3), 1);
        layer.init(&opencl_state).expect("unable to init Conv2D");
        layer.weights_buff = Some(filter);
        layer.biases_buff = Some(bias);

        layer.last_inputs_buffer = Some(image);

        let output_to_loss_derivatives_buff = vec![
            0.8170, 0.4417, 0.2387,
            1.1320, 0.9614, 1.3130
        ].to_buffer(false, &opencl_state)
            .unwrap();

        let actual_gradients_buff = &layer
            .compute_gradients(&output_to_loss_derivatives_buff)
            .expect("unable to compute conv2d gradients")[0]
            .value;

        let actual_gradients = Vec::<f32>::from_buffer(actual_gradients_buff, false, &opencl_state)
            .expect("unable to convert from the actual gradients buffer to a vector");

        approx_eq::assert_approx_equal(&actual_gradients, &expected_result, 1);
    }

    #[test]
    fn should_convolute_correctly() -> () {
        let opencl_state = setup_opencl(DeviceType::GPU).expect("unable to setup opencl");
        let image = vec![
            0.33, 0.14, 0.99, 1.0, 0.1, 
            0.51, 0.31, 0.91, 0.1, 0.3, 
            0.8,  0.4,  0.5,  0.2, 0.1,

            0.31, 0.41, 2.32, 1.2, 0.41,
            0.91, 0.32, 0.93, 0.12,0.34,
            0.3,  0.2,  0.91, 0.41,2.3
        ]
        .to_buffer(false, &opencl_state)
        .expect("unable to get image buffer");

        let filter = vec![0.3, 0.4, 0.9, 0.1, 0.2, 1.0, 0.2, 0.5, 0.81]
            .to_buffer(false, &opencl_state)
            .expect("unable to get filter buffer");
        let bias = vec![0.123]
            .to_buffer(false, &opencl_state)
            .expect("unable to get the biases buffer");
        let expected_result = vec![
            2.8340, 2.1430, 1.4790,
            4.3271, 3.2961, 4.2520
        ];

        let mut layer = Conv2D::new_raw((5, 3), (3, 3), 1);
        layer.init(&opencl_state).expect("unable to init Conv2D");
        layer.weights_buff = Some(filter);
        layer.biases_buff = Some(bias);

        let result_buffer = layer
            .propagate(&image)
            .expect("unable to propagate conv2d layer");
        let result = Vec::<f32>::from_buffer(result_buffer, false, &opencl_state)
            .expect("unable to get resulting convolution buffer");

        assert_approx_equal_distance(&dbg!(result), dbg!(&expected_result), 0.01);
    }
}