use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::loss_functions::LossFunction;
use crate::utils::vector_operations::VectorOperations;

#[derive(Debug)]
/// The Mean Squared loss function, good for some problem with
/// linear regression, because this error is quite free, in comparison
/// to the Categorical Cross Entropy loss function which restricts things
/// to be from 1.0 to 0.0 to work well
pub struct MeanSquared;

impl LossFunction for MeanSquared {
    fn compute_loss(&self, outputs: &Vec<f32>, expected_outputs: &Vec<f32>) -> f32 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());

        expected_outputs
            .subtract(outputs)
            .powf(2.0)
            .par_iter()
            .sum::<f32>()
            / outputs_amount as f32
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f32,
        expected_output: f32,
    ) -> f32 {
        2.0 / ouputs_amount as f32 * (expected_output - output)
    }
}