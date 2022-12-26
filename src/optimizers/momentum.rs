//! A module that contains the momentum based optimizer that tries to dampen the training process
//! as to make it improve faster.

use std::collections::HashMap;

use opencl3::{device::cl_float, memory::{Buffer, ClMem, CL_MEM_READ_WRITE}, kernel::ExecuteKernel};

use super::{OptimizationError, Optimizer};
use crate::utils::{
    opencl::{ensure_program, opencl_state::EnsureKernelsAndProgramError, empty_buffer},
    OpenCLState,
};

const PROGRAM_NAME: &str = "MOMENTUM_OPTIMIZER";
const PROGRAM_SOURCE: &str = include_str!("./kernels/momentum.cl");
const COMPUTE_UPDATE_VECTOR_KERNEL_NAME: &str = "compute_update_vector";

pub(crate) fn compile_momentum(
    state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    ensure_program(
        state,
        PROGRAM_NAME,
        PROGRAM_SOURCE,
        "",
        &[COMPUTE_UPDATE_VECTOR_KERNEL_NAME],
    )
}

#[derive(Debug)]
/// The momentum based optimizer is one that tries to simulate momentum using a `gamma` constant
/// that defines how much of the last update vector should be added together with the current
/// update vector as to further improve the training process.
pub struct MomentumOptimizer<'a> {
    learning_rate: f32,
    momentum_gamma: f32,

    last_update_vectors: HashMap<(usize, String), Buffer<cl_float>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> MomentumOptimizer<'a> {
    /// Creates a new instance of a Optimizer based on Momentum, that tries to speed up the
    /// training process in the right direction.
    ///
    /// The **momentum_gamma** parameter here is how much of the last update vector should be
    /// considered in the current one as to simulate momentum. This value is usually just `0.9`.
    pub fn new(learning_rate: f32, momentum_gamma: f32) -> Self {
        MomentumOptimizer {
            learning_rate,
            momentum_gamma,

            last_update_vectors: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for MomentumOptimizer<'a> {
    fn init(&mut self, opencl_state: &'a OpenCLState) -> Result<(), opencl3::error_codes::ClError> {
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
            self.last_update_vectors.insert((layer_index, parameter_id), last_update_vector);
        }

        Ok(update_vector)
    }
}
