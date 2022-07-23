use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::loss_functions::loss_function::LossFunctionF64;
use crate::utils::vector_operations::VectorOperations;

pub struct MeanSquared;

impl LossFunctionF64 for MeanSquared {
    fn compute_loss(&self, outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> f64 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());

        expected_outputs
            .subtract(outputs)
            .powf(2.0)
            .par_iter()
            .sum::<f64>()
            / outputs_amount as f64
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f64,
        expected_output: f64,
    ) -> f64 {
        2.0 / ouputs_amount as f64 * (expected_output - output)
    }
}
