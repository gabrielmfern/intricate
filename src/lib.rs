//! A layer-driven **Machine Learning** that makes it extremly easy training, running and creating
//! Model's for predicting some type of data using OpenCL. (does not yet support multiple devices)
//!
//! This crate is completely at its basis written with the very good
//! definitions from a [video](https://youtu.be/pauPCy_s0Ok) 
//! of the channel **The Independent Code**, so much credit to him
//! at least for the calculation of gradients and the definition of
//! activation functions as layers.

#[deny(missing_docs)]

pub mod layers;
pub mod loss_functions;
pub mod model;
pub mod utils;

pub use model::Model;

pub mod types;

mod tests;