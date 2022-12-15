//! The module that contains all the available parameter initializers for Intricate

use std::ops::Range;
use rand::prelude::*;
use rand_distr::Normal;
use rand_distr::Uniform;
use super::Layer;
use savefile_derive::Savefile;

use crate::types::ModelLayer;

/// A trait that is implemented for all Intricate layer parameter type of initializers.
pub trait Initializer {
    /// Generates just one number based on the Initializer's implementation
    fn initialize_0d<'a>(&self, layer: ModelLayer<'a>) -> f32;

    /// Generates a Vec of numbers initialized based on the Initializer's implementation
    fn initialize_1d<'a>(&self, count: usize, layer: ModelLayer<'a>) -> Vec<f32> {
        (0..count)
            .map(|_| self.initialize_0d(layer))
            .collect()
    }

    /// Generates a Matrix of numbers initialized based on the Initializer's implementation
    fn initialize_2d<'a>(&self, shape: (usize, usize), layer: ModelLayer<'a>) -> Vec<Vec<f32>> {
        (0..shape.0)
            .map(|_| self.initialize_1d(shape.1, layer))
            .collect()
    }

    /// Generates a 3D Matrix of Vec of numbers initialized based on the Initializer's implementation
    fn initialize_3d<'a>(&self, shape: (usize, usize, usize), layer: ModelLayer<'a>) -> Vec<Vec<Vec<f32>>> {
        (0..shape.0)
            .map(|_| self.initialize_2d((shape.1, shape.2), layer))
            .collect()
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that pretty much just initializes all values of a parameter with a constat value
/// provided by the **new** method
pub struct ConstantInitializer {
    pub constant: f32,
}

impl ConstantInitializer {
    pub fn new(constant: f32) -> Self {
        ConstantInitializer { constant }
    }
}

impl Initializer for ConstantInitializer {
    fn initialize_0d<'a>(&self, _: ModelLayer<'a>) -> f32 {
        self.constant
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers inside a given range provided by the **new** method
pub struct LimitedRandomInitializer {
    pub limit_interval: Range<f32>,
}

impl LimitedRandomInitializer {
    pub fn new(limit_interval: Range<f32>) -> Self {
        LimitedRandomInitializer { limit_interval }
    }
}

impl Initializer for LimitedRandomInitializer {
    fn initialize_0d<'a>(&self, _layer: ModelLayer<'a>) -> f32 {
        let mut rng = thread_rng();

        rng.gen_range(self.limit_interval)
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers in a normal distribution based on a **mean** and a
/// **standard deviation** provided by the `new` method
pub struct NormalRandomInitializer {
    pub mean: f32,
    pub standard_deviation: f32,
}

impl NormalRandomInitializer  {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        NormalRandomInitializer { mean, standard_deviation: std_dev }
    }
}

impl Initializer for NormalRandomInitializer {
    fn initialize_0d<'a>(&self, layer: ModelLayer<'a>) -> f32 {
        let distribution = Normal::new(self.mean, self.standard_deviation)
            .expect("Unable to create Normal distribution for the NormaRandomInitalizer");
        let mut rng = thread_rng();
        distribution.sample(&mut rng)
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers in a uniform distribution based on a range 
/// provided by the `new` method
pub struct UniformRandomInitializer {
    pub interval: Range<f32>,
}

impl UniformRandomInitializer {
    pub fn new(interval: Range<f32>) -> Self {
        UniformRandomInitializer { interval }
    }
}

impl Initializer for UniformRandomInitializer {
    fn initialize_0d<'a>(&self, layer: ModelLayer<'a>) -> f32 {
        let distribution = Uniform::new(self.interval.start, self.interval.end);
        let mut rng = thread_rng();
        distribution.sample(&mut rng) as f32
    }
}

#[derive(Debug, Clone)]
/// A Initializer that generates random numbers in a uniform distribution based on a range 
/// calculated using the inputs and outputs of the layer the initializer is being used on.
/// It is defined in a range of [-limit, limit] where 
/// limit = sqrt(6.0 / (inputs_amount + outputs_amount))
pub struct GlorotUniformInitializer;

impl GlorotUniformInitializer {
    pub fn new() -> Self { GlorotUniformInitializer }
}

impl Initializer for GlorotUniformInitializer {
    fn initialize_0d<'a>(&self, layer: ModelLayer<'a>) -> f32 {
        let fan_in = layer.get_inputs_amount() as f32;
        let fan_out = layer.get_outputs_amount() as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();
        let distribution = Uniform::new(-limit, limit);
        let mut rng = thread_rng();
        distribution.sample(&mut rng) as f32
    }
}

#[derive(Debug, Clone)]
/// A Initializer that generates random numbers in a normal distribution with a mmean of **0.0**
/// and a standard deviation calculated based on the inputs and outputs amount
/// The standard deviation is calculated as **sqrt(2.0 / (inputs_amount + outputs_amount))**
pub struct GlorotNormalInitializer;

impl GlorotNormalInitializer {
    pub fn new() -> Self { GlorotNormalInitializer }
}

impl Initializer for GlorotNormalInitializer {
    fn initialize_0d<'a>(&self, layer: ModelLayer<'a>) -> f32 {
        let fan_in = layer.get_inputs_amount() as f32;
        let fan_out = layer.get_outputs_amount() as f32;

        let mean = 0.0f32;
        let std_dev = (2.0f32 / (fan_in + fan_out)).sqrt();

        let distribution = Normal::new(mean, std_dev)
            .expect("Unable to create Normal distribution for the GlorotNormalInitializer");
        let mut rng = thread_rng();
        distribution.sample(&mut rng) as f32
    }
}