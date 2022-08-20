//! The module that contains all activations functions currently implemented for Intricate,
//! which as of v0.3.0, are:
//!
//! - ReLU (Rectified Linear Unit)
//! - Sigmoid
//! - TanH (Hyperbolic Tangent)
//! - SoftMax

pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::SoftMax;
pub use tanh::TanH;

use crate::utils::{opencl::EnsureKernelsAndProgramError, OpenCLState};

pub(crate) fn compile_activations(
    opencl_state: &mut OpenCLState,
) -> Result<(), EnsureKernelsAndProgramError> {
    relu::compile_relu(opencl_state)?;
    sigmoid::compile_sigmoid(opencl_state)?;
    softmax::compile_softmax(opencl_state)?;
    tanh::compile_tanh(opencl_state)?;

    Ok(())
}
