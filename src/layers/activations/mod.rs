//! The module that contains all activations functions currently implemented for Intricate,
//! which as of v0.3.0, are:
//!
//! - ReLU (Rectified Linear Unit)
//! - Sigmoid
//! - TanH (Hyperbolic Tangent)
//! - SoftMax

pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod softmax;

pub use sigmoid::Sigmoid;
pub use tanh::TanH;
pub use softmax::SoftMax;
pub use relu::ReLU;

// mod tests;