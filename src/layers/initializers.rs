//! The module that contains all the available parameter initializers for Intricate

use std::ops::Range;
use intricate_macros::FromForAllUnnamedVariants;
use rand::prelude::*;
use rand_distr::Normal;
use rand_distr::Uniform;
use super::Layer;
use savefile_derive::Savefile;
use savefile::{Serialize, Introspect};

/// A trait that is implemented for all Intricate layer parameter type of initializers.
pub trait InitializerTrait
    where Self: std::fmt::Debug + Serialize + Introspect,
{
    /// Generates just one number based on the Initializer's implementation
    fn initialize_0d<'a>(&self, layer: &dyn Layer<'a>) -> f32;

    /// Generates a Vec of numbers initialized based on the Initializer's implementation
    fn initialize_1d<'a>(&self, count: usize, layer: &dyn Layer<'a>) -> Vec<f32> {
        (0..count)
            .map(|_| self.initialize_0d(layer))
            .collect()
    }

    /// Generates a Matrix of numbers initialized based on the Initializer's implementation
    fn initialize_2d<'a>(&self, shape: (usize, usize), layer: &dyn Layer<'a>) -> Vec<Vec<f32>> {
        (0..shape.0)
            .map(|_| self.initialize_1d(shape.1, layer))
            .collect()
    }

    /// Generates a 3D Matrix of Vec of numbers initialized based on the Initializer's implementation
    fn initialize_3d<'a>(&self, shape: (usize, usize, usize), layer: &dyn Layer<'a>) -> Vec<Vec<Vec<f32>>> {
        (0..shape.0)
            .map(|_| self.initialize_2d((shape.1, shape.2), layer))
            .collect()
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that pretty much just initializes all values of a parameter with a constat value
/// provided by the **new** method
pub struct ConstantInitializer {
    /// The constant that all the parameters will be
    pub constant: f32,
}

impl ConstantInitializer {
    /// Creates a new Constant initializer
    pub fn new(constant: f32) -> Self {
        ConstantInitializer { constant }
    }
}

impl InitializerTrait for ConstantInitializer {
    fn initialize_0d<'a>(&self, _: &dyn Layer<'a>) -> f32 {
        self.constant
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers inside a given range provided by the **new** method
pub struct LimitedRandomInitializer {
    /// The interval that the random numbers will be generated in
    pub limit_interval: Range<f32>,
}

impl LimitedRandomInitializer {
    /// Creates a new Limited Random initializer
    pub fn new(limit_interval: Range<f32>) -> Self {
        LimitedRandomInitializer { limit_interval }
    }
}

impl InitializerTrait for LimitedRandomInitializer {
    fn initialize_0d<'a>(&self, _layer: &dyn Layer<'a>) -> f32 {
        let mut rng = thread_rng();

        rng.gen_range(self.limit_interval.clone())
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers in a normal distribution based on a **mean** and a
/// **standard deviation** provided by the `new` method
pub struct NormalRandomInitializer {
    /// The mean for the Normal distribution
    pub mean: f32,
    /// The standard deviation for the Normal distribution
    pub standard_deviation: f32,
}

impl NormalRandomInitializer  {
    /// Creates a new Normal Random initializer
    pub fn new(mean: f32, std_dev: f32) -> Self {
        NormalRandomInitializer { mean, standard_deviation: std_dev }
    }
}

impl InitializerTrait for NormalRandomInitializer {
    fn initialize_0d<'a>(&self, _layer: &dyn Layer<'a>) -> f32 {
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
    /// The interval that will be used to limit the uniform distribution
    pub interval: Range<f32>,
}

impl UniformRandomInitializer {
    /// Creates a new Uniform Random initializer
    pub fn new(interval: Range<f32>) -> Self {
        UniformRandomInitializer { interval }
    }
}

impl InitializerTrait for UniformRandomInitializer {
    fn initialize_0d<'a>(&self, _layer: &dyn Layer<'a>) -> f32 {
        let distribution = Uniform::new(self.interval.start, self.interval.end);
        let mut rng = thread_rng();
        distribution.sample(&mut rng) as f32
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers in a uniform distribution based on a range 
/// calculated using the inputs and outputs of the layer the initializer is being used on.
/// It is defined in a range of [-limit, limit] where 
/// limit = sqrt(6.0 / (inputs_amount + outputs_amount))
pub struct GlorotUniformInitializer();

impl GlorotUniformInitializer {
    /// Creates a new Glorot Uniform initializer
    pub fn new() -> Self { GlorotUniformInitializer() }
}

impl InitializerTrait for GlorotUniformInitializer {
    fn initialize_0d<'a>(&self, layer: &dyn Layer<'a>) -> f32 {
        let fan_in = layer.get_inputs_amount() as f32;
        let fan_out = layer.get_outputs_amount() as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();
        let distribution = Uniform::new(-limit, limit);
        let mut rng = thread_rng();
        distribution.sample(&mut rng) as f32
    }
}

#[derive(Debug, Clone, Savefile)]
/// A Initializer that generates random numbers in a normal distribution with a mmean of **0.0**
/// and a standard deviation calculated based on the inputs and outputs amount
/// The standard deviation is calculated as **sqrt(2.0 / (inputs_amount + outputs_amount))**
pub struct GlorotNormalInitializer();

impl GlorotNormalInitializer {
    /// Creates a new Glorot Normal initializer
    pub fn new() -> Self { GlorotNormalInitializer() }
}

impl InitializerTrait for GlorotNormalInitializer {
    fn initialize_0d<'a>(&self, layer: &dyn Layer<'a>) -> f32 {
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

#[derive(Debug, Clone, Savefile, FromForAllUnnamedVariants)]
/// The enum that contains all of the possible Initializers
pub enum Initializer {
    /// The Constant initalizer
    Constant(ConstantInitializer),
    /// The Limited Random initalizer
    LimitedRandom(LimitedRandomInitializer),
    /// The Uniform Random initalizer
    UniformRandom(UniformRandomInitializer),
    /// The Normal Random initalizer
    NormalRandom(NormalRandomInitializer),
    /// The Glorot Normal initalizer
    GlorotNormal(GlorotNormalInitializer),
    /// The Glorot Uniform initalizer
    GlorotUniform(GlorotUniformInitializer),
}

impl InitializerTrait for Initializer {
    fn initialize_0d<'a>(&self, layer: &dyn Layer<'a>) -> f32 {
        match self {
            Initializer::Constant(i) => i.initialize_0d(layer),
            Initializer::LimitedRandom(i) => i.initialize_0d(layer),
            Initializer::UniformRandom(i) => i.initialize_0d(layer),
            Initializer::NormalRandom(i) => i.initialize_0d(layer),
            Initializer::GlorotNormal(i) => i.initialize_0d(layer),
            Initializer::GlorotUniform(i) => i.initialize_0d(layer),
        }
    }
}

impl Default for Initializer {
    fn default() -> Self {
        Self::GlorotUniform(GlorotUniformInitializer::new())
    }
}