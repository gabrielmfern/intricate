//! The modulem that contains the Adam optimizer.

use std::collections::HashMap;

use opencl3::{
    kernel::ExecuteKernel,
    memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    types::{cl_float, cl_int, cl_bool},
};

use crate::utils::{
    opencl::{
        empty_buffer, ensure_program, opencl_state::EnsureKernelsAndProgramError,
    }, OpenCLState,
};

use super::{OptimizationError, Optimizer};

const PROGRAM_NAME: &str = "ADAM_OPTIMIZER";
const PROGRAM_SOURCE: &str = include_str!("kernels/adam.cl");

const COMPUTE_UPDATE_VECTORS_KERNEL: &str = "compute_update_vectors";

pub(crate) fn compile_adam(state: &mut OpenCLState) -> Result<(), EnsureKernelsAndProgramError> {
    ensure_program(
        state,
        PROGRAM_NAME,
        PROGRAM_SOURCE,
        "",
        &[COMPUTE_UPDATE_VECTORS_KERNEL],
    )
}

#[derive(Debug)]
/// An optimizer that works well with pretty much everything. It has four hyper parameters that can
/// be tuned for your Model. This Optimizer tries to combine features from other optimizers but
/// without needing as much memory requirements such as Adagrad or sometimes blowing up to infinity
/// with Nesterov's optimizer or just the basic Momentum optimizer.
pub struct AdamOptimizer<'a> {
    learning_rate_alpha: f32,
    decay_rate_beta_1: f32,
    decay_rate_beta_2: f32,

    safety_epsilon: f32,

    last_moment_first_estimate_per_parameter: HashMap<(usize, String), Buffer<cl_float>>,
    last_moment_second_estimate_per_parameter: HashMap<(usize, String), Buffer<cl_float>>,

    opencl_state: Option<&'a OpenCLState>,
}

impl<'a> AdamOptimizer<'a> {
    /// Creates a new instance of the Adam optimizer.
    ///
    /// The hyper parameters here are usually just 0.001, 0.9, 0.999 and 0.0000001 respectively.
    pub fn new(
        learning_rate: f32,
        decay_rate_beta_1: f32,
        decay_rate_beta_2: f32,
        safety_epsilon: f32,
    ) -> Self {
        AdamOptimizer {
            learning_rate_alpha: learning_rate,
            decay_rate_beta_1,
            decay_rate_beta_2,

            safety_epsilon,

            last_moment_first_estimate_per_parameter: HashMap::default(),
            last_moment_second_estimate_per_parameter: HashMap::default(),

            opencl_state: None,
        }
    }
}

impl<'a> Optimizer<'a> for AdamOptimizer<'a> {
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
        timestep: usize,
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

        let kernel = program.get_krnl(COMPUTE_UPDATE_VECTORS_KERNEL)?;

        let gradients_count = gradients.size()? / std::mem::size_of::<f32>();
        let current_moment_first_estimate =
            empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;
        let current_moment_second_esteimate =
            empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;
        let update_vector = empty_buffer(gradients_count, CL_MEM_READ_WRITE, state)?;

        let mut execute_kernel = ExecuteKernel::new(kernel);
        let last_moment_first_estimate_per_parameter = self
            .last_moment_first_estimate_per_parameter
            .get(&(layer_index, parameter_id.to_string()));
        let last_moment_second_estimate_per_parameter = self
            .last_moment_second_estimate_per_parameter
            .get(&(layer_index, parameter_id.to_string()));

        execute_kernel
            .set_arg(gradients)
            .set_arg(
                last_moment_first_estimate_per_parameter.unwrap_or(&empty_buffer(
                    1,
                    CL_MEM_READ_ONLY,
                    state,
                )?),
            )
            .set_arg(&current_moment_first_estimate)
            .set_arg(
                last_moment_second_estimate_per_parameter.unwrap_or(&empty_buffer(
                    1,
                    CL_MEM_READ_ONLY,
                    state,
                )?),
            )
            .set_arg(&current_moment_second_esteimate)
            .set_arg(&update_vector)
            .set_arg(&(gradients_count as cl_int))
            .set_arg(&(last_moment_first_estimate_per_parameter.is_some() as cl_int))
            .set_arg(&(last_moment_second_estimate_per_parameter.is_some() as cl_int))
            .set_arg(&(self.decay_rate_beta_1 as cl_float))
            .set_arg(&(self.decay_rate_beta_2 as cl_float))
            .set_arg(&(self.learning_rate_alpha as cl_float))
            .set_arg(&(timestep as cl_float))
            .set_arg(&(self.safety_epsilon as cl_float))
            .set_global_work_size(gradients_count)
            .enqueue_nd_range(&queue)?
            .wait()?;

        self.last_moment_first_estimate_per_parameter.insert(
            (layer_index, parameter_id.to_string()),
            current_moment_first_estimate,
        );
        self.last_moment_second_estimate_per_parameter.insert(
            (layer_index, parameter_id.to_string()),
            current_moment_second_esteimate,
        );

        Ok(update_vector)
    }
}