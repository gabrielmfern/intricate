//! This crate is completely at its basis written with the very good
//! definitions from a [video](https://youtu.be/pauPCy_s0Ok) 
//! of the channel **The Independent Code**, so much credit to him
//! at least for the calculation of gradients and the definition of
//! activation functions as layers.
//! 
//! A OpenCL accelerated library that creates, trains and runs neural networks
//! in safe Rust code.

pub mod layers;
pub mod loss_functions;
pub mod model;
pub mod model_gpu;
pub mod utils;

mod tests;