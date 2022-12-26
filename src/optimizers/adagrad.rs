//! The module that contains the Adagrad optimizer.

use std::collections::HashMap;

use opencl3::{
    device::cl_float,
    error_codes::ClError,
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
    types::cl_int,
};

use crate::utils::{
    opencl::{empty_buffer, ensure_program, opencl_state::EnsureKernelsAndProgramError},
    OpenCLState,
};

use super::{OptimizationError, Optimizer};

const PROGRAM_NAME: &str = "ADAGRAD_OPTIMIZER";
const PROGRAM_SOURCE: &str = include_str!("./kernels/adagrad.cl");
const COMPUTE_UPDATE_VECTOR_AND_UPDATE_GHS_KERNEL_NAME: &str =
    "compute_update_vector_and_update_gradient_history_summation";

pub(crate) fn compile_adagrad(state: &mut OpenCLState) -> Result<(), EnsureKernelsAndProgramError> {
    ensure_program(
        state,
        PROGRAM_NAME,
        PROGRAM_SOURCE,
        "",
        &[COMPUTE_UPDATE_VECTOR_AND_UPDATE_GHS_KERNEL_NAME],
    )
}

#[derive(Debug)]
/// The Adagrad Optimizer does a gradient-based optimization that adapts the learning rates for
/// parameters that are much more necessary than others for the given purpose of the Model.
pub struct AdagradOptimizer<'a> {
    learning_rate: f32,
    epsilon: f32,

    gradients_history_summation_per_parameter: HashMap<(usize, String), Buffer<cl_float>>,
    // TODO: add a way to perhaps save the Optimizer with this gradient history to train the Model
    // later
    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> AdagradOptimizer<'a> {
    /// Creates a new uninitialized instance of the Adagrad optimizer.
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        AdagradOptimizer {
            learning_rate,
            epsilon,
            gradients_history_summation_per_parameter: HashMap::default(),
            // gradients_history_summation_per_parameter_host: HashMap::default(),
            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for AdagradOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn optimize_parameters(
        &self,
        _parameters: &mut Buffer<cl_float>,
        _parameter_id: String,
        _timestep: usize,
        _layer_index: usize,
    ) -> Result<(), OptimizationError> {
        Ok(())
    }

    fn compute_update_vectors(
        &mut self,
        gradients: &Buffer<cl_float>,
        parameter_id: String,
        _timestep: usize,
        layer_index: usize,
    ) -> Result<Buffer<cl_float>, OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(OptimizationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(COMPUTE_UPDATE_VECTOR_AND_UPDATE_GHS_KERNEL_NAME)?;

        let gradients_size = gradients.size()?;
        let gradients_count = gradients_size / std::mem::size_of::<f32>();

        let mut update_vector = empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;

        if let Some(gradients_history_summation) = self
            .gradients_history_summation_per_parameter
            .get(&(layer_index, parameter_id.to_string()))
        {
            if gradients_size != gradients_history_summation.size()? {
                return Err(OptimizationError::InvalidParametersSize(format!(
                    "The gradients size does not match the gradients history summation size for the Adagrad optimizer with the parameter_id '{}'",
                    parameter_id.to_string(),
                )));
            }
            ExecuteKernel::new(kernel)
                .set_arg(gradients)
                .set_arg(gradients_history_summation)
                .set_arg(&mut update_vector)
                .set_arg(&(1 as cl_int))
                .set_arg(&(self.learning_rate as cl_float))
                .set_arg(&(self.epsilon as cl_float))
                .set_global_work_size(gradients_count)
                .enqueue_nd_range(queue)?
                .wait()?;
        } else {
            let mut gradients_history_summation =
                empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;
            ExecuteKernel::new(kernel)
                .set_arg(gradients)
                .set_arg(&mut gradients_history_summation)
                .set_arg(&mut update_vector)
                .set_arg(&(0 as cl_int))
                .set_arg(&(self.learning_rate as cl_float))
                .set_arg(&(self.epsilon as cl_float))
                .set_global_work_size(gradients_count)
                .enqueue_nd_range(queue)?
                .wait()?;
            self.gradients_history_summation_per_parameter.insert(
                (layer_index, parameter_id.to_string()),
                gradients_history_summation,
            );
        }

        Ok(update_vector)
    }
}
