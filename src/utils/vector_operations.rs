pub trait VectorOperations {
    type Item;

    fn add(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn subtract(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn subtract_number(&self, numb: Self::Item) -> Vec<Self::Item>;

    fn multiply_number(&self, factor: &Self::Item) -> Vec<Self::Item>;

    fn divide_number(&self, factor: &Self::Item) -> Vec<Self::Item>;

    fn multiply(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn divide(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn powf(&self, power: f64) -> Vec<Self::Item>;

    fn from_powf(&self, base: f64) -> Vec<Self::Item>;
}

impl VectorOperations for Vec<f32> {
    type Item = f32;

    fn add(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a + b)
            .collect::<Vec<Self::Item>>()
    }

    fn subtract(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a - b)
            .collect::<Vec<Self::Item>>()
    }

    fn subtract_number(&self, numb: Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x - numb).collect::<Vec<Self::Item>>()
    }

    fn powf(&self, power: f64) -> Vec<Self::Item> {
        self.iter().map(|x| x.powf(power as Self::Item)).collect::<Vec<Self::Item>>()
    }

    fn from_powf(&self, base: f64) -> Vec<Self::Item> {
        self.iter().map(|x| (base as f32).powf(*x)).collect::<Vec<Self::Item>>()
    }

    fn divide(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a / b)
            .collect::<Vec<Self::Item>>()
    }

    fn multiply(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a * b)
            .collect::<Vec<Self::Item>>()
    }

    fn divide_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x / factor).collect::<Vec<Self::Item>>()
    }

    fn multiply_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x * factor).collect::<Vec<Self::Item>>()
    }
}

impl VectorOperations for Vec<f64> {
    type Item = f64;

    fn add(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a + b)
            .collect::<Vec<Self::Item>>()
    }

    fn subtract(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a - b)
            .collect::<Vec<Self::Item>>()
    }

    fn subtract_number(&self, numb: Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x - numb).collect::<Vec<Self::Item>>()
    }

    fn powf(&self, power: f64) -> Vec<Self::Item> {
        self.iter().map(|x| x.powf(power)).collect::<Vec<Self::Item>>()
    }

    fn from_powf(&self, base: f64) -> Vec<Self::Item> {
        self.iter().map(|x| base.powf(*x)).collect::<Vec<Self::Item>>()
    }

    fn divide(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a / b)
            .collect::<Vec<Self::Item>>()
    }

    fn multiply(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        self.iter()
            .zip(against)
            .map(|(a, b)| a * b)
            .collect::<Vec<Self::Item>>()
    }

    fn divide_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x / factor).collect::<Vec<Self::Item>>()
    }

    fn multiply_number(&self, factor: &Self::Item) -> Vec<Self::Item> {
        self.iter().map(|x| x * factor).collect::<Vec<Self::Item>>()
    }
}