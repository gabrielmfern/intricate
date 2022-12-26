//! A module that contains the momentum based optimizer that tries to dampen the training process
//! as to make it improve faster.

use std::collections::HashMap;

use opencl3::{
    device::cl_float,
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_WRITE},
};

use super::{OptimizationError, Optimizer};
use crate::utils::{
    opencl::{empty_buffer, ensure_program, opencl_state::EnsureKernelsAndProgramError},
    OpenCLState,
};

const PROGRAM_NAME: &str = "NESTEROV_OPTIMIZER";
const PROGRAM_SOURCE: &str = include_str!("./kernels/nesterov.cl");
const OPTIMIZE_PARAMETERS_KERNEL_NAME: &str = "optimize_parameters";
const COMPUTE_UPDATE_VECTOR_KERNEL_NAME: &str = "compute_update_vector";

pub(crate) fn compile_nesterov(
    state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    ensure_program(
        state,
        PROGRAM_NAME,
        PROGRAM_SOURCE,
        "",
        &[
            OPTIMIZE_PARAMETERS_KERNEL_NAME,
            COMPUTE_UPDATE_VECTOR_KERNEL_NAME,
        ],
    )
}

#[derive(Debug)]
/// The Nesterov Accelerated Momentum Optimizer is an optimizer that evolves on the Momentum
/// Optimizer as to accelerate the momentum based the way the momentum optimizer is predictable for
/// the next training step.
pub struct NesterovOptimizer<'a> {
    learning_rate: f32,
    momentum_gamma: f32,

    last_update_vectors: HashMap<(usize, String), Buffer<cl_float>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> NesterovOptimizer<'a> {
    /// Creates a new instance of a Optimizer based on Acclerated Momentum, that tries to speed up the
    /// training process in the right direction.
    ///
    /// The **momentum_gamma** parameter here is how much of the last update vector should be
    /// considered in the current one as to simulate momentum. This value is usually just `0.9`.
    pub fn new(learning_rate: f32, momentum_gamma: f32) -> Self {
        NesterovOptimizer {
            learning_rate,
            momentum_gamma,

            last_update_vectors: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for NesterovOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), opencl3::error_codes::ClError> {
        self.opencl_state = Some(opencl_state);

        Ok(())
    }

    fn optimize_parameters(
        &self,
        parameters: &mut Buffer<cl_float>,
        parameter_id: String,
        _timestep: usize,
        layer_index: usize,
    ) -> Result<(), OptimizationError> {
        if self.opencl_state.is_none() {
            return Err(OptimizationError::UninitializedState);
        }

        let state = self.opencl_state.unwrap();

        if state.queues.is_empty() {
            return Err(OptimizationError::NoCommandQueueFound);
        }

        let queue = state.queues.first().unwrap();

        let program = state.get_prgm(PROGRAM_NAME)?;
        let kernel = program.get_krnl(OPTIMIZE_PARAMETERS_KERNEL_NAME)?;

        if let Some(parameter_last_update_vector) = self
            .last_update_vectors
            .get(&(layer_index, parameter_id.to_string()))
        {
            if parameters.size()? != parameter_last_update_vector.size()? {
                return Err(OptimizationError::InvalidParametersSize(
                    format!("The last update vector for the parameter with id '{}' does not match the size of the parameters!", parameter_id)
                ));
            }
            let parameters_count = parameters.size()? / std::mem::size_of::<cl_float>();
            ExecuteKernel::new(kernel)
                .set_arg(parameters)
                .set_arg(parameter_last_update_vector)
                .set_arg(&(self.momentum_gamma as cl_float))
                .set_global_work_size(parameters_count)
                .enqueue_nd_range(queue)?
                .wait()?;
        }

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
        let kernel = program.get_krnl(COMPUTE_UPDATE_VECTOR_KERNEL_NAME)?;

        let gradients_count = gradients.size()? / std::mem::size_of::<f32>();

        let mut update_vector = empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;

        if let Some(last_update_vector) = self
            .last_update_vectors
            .get_mut(&(layer_index, parameter_id.to_string()))
        {
            ExecuteKernel::new(kernel)
                .set_arg(gradients)
                .set_arg(last_update_vector)
                .set_arg(&mut update_vector)
                .set_arg(&(self.momentum_gamma as cl_float))
                .set_arg(&(self.learning_rate as cl_float))
                .set_global_work_size(gradients_count)
                .enqueue_nd_range(queue)?
                .wait()?;
        } else {
            let mut last_update_vector = empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;
            ExecuteKernel::new(kernel)
                .set_arg(gradients)
                .set_arg(&mut last_update_vector)
                .set_arg(&mut update_vector)
                .set_arg(&(self.momentum_gamma as cl_float))
                .set_arg(&(self.learning_rate as cl_float))
                .set_global_work_size(gradients_count)
                .enqueue_nd_range(queue)?
                .wait()?;
            self.last_update_vectors
                .insert((layer_index, parameter_id), last_update_vector);
        }

        Ok(update_vector)
    }
}
