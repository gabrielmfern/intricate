use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};

pub trait VectorOperations {
    type Item;

    fn add(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn subtract(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn multiply_number(&self, factor: &Self::Item) -> Vec<Self::Item>;

    fn divide_number(&self, factor: &Self::Item) -> Vec<Self::Item>;

    fn multiply(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn divide(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn powf(&self, power: f64) -> Vec<Self::Item>;

    fn from_powf(&self, base: f64) -> Vec<Self::Item>;
}

impl VectorOperations for Vec<f64> {
    type Item = f64;

    fn add(&self, against: &Vec<f64>) -> Vec<f64> {
        self.par_iter().zip(against).map(|(a, b)| a + b).collect::<Vec<f64>>()
    }

    fn subtract(&self, against: &Vec<f64>) -> Vec<f64> {
        self.par_iter().zip(against).map(|(a,b)| a - b).collect::<Vec<f64>>()
    }

    fn powf(&self, power: f64) -> Vec<f64> {
        self.par_iter().map(|x| x.powf(power)).collect::<Vec<f64>>()
    }

    fn from_powf(&self, base: f64) -> Vec<Self::Item> {
        self.par_iter().map(|x| base.powf(*x)).collect::<Vec<f64>>()
    }

    fn divide(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.par_iter().zip(against).map(|(a,b)| a / b).collect::<Vec<f64>>()
    }

    fn multiply(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.par_iter().zip(against).map(|(a,b)| a * b).collect::<Vec<f64>>()
    }

    fn divide_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.par_iter().map(|x| x / factor).collect::<Vec<f64>>()
    }

    fn multiply_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.par_iter().map(|x| x * factor).collect::<Vec<f64>>()
    }
}
