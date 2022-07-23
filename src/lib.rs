//! This crate is completely at its basis written with the very good
//! definitions from a [video](https://youtu.be/pauPCy_s0Ok) 
//! of the channel **The Independent Code**, so much credit to him.
//! 
//! A GPU accelerated library that creates/trains/runs neural networks
//! in pure Rust, safe code.
//! 
//! Intricate has a sepparation between double-precision floating-point numbers
//! and just floating-point numbers so that on cases where you don't need the precision
//! you don't to use the RAM

pub mod layers;
pub mod loss_functions;
pub mod model;
pub mod utils;
