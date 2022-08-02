//! This crate is completely at its basis written with the very good
//! definitions from a [video](https://youtu.be/pauPCy_s0Ok) 
//! of the channel **The Independent Code**, so much credit to him
//! at least for the calculation of gradients and the definition of
//! activation functions as layers.
//! 
//! A GPU accelerated library that creates/trains/runs neural networks
//! in pure safe Rust code.

pub mod layers;
pub mod loss_functions;
pub mod model;
pub mod utils;
pub mod gpu;

mod tests;
