pub mod categorical_cross_entropy;
pub mod mean_squared;

use std::fmt::Debug;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A trait representing the definitions of a function
/// for computing the loss/cost/error of a Model
pub trait LossFunction
where 
    Self: Sync + Send + Debug
{
    /// Computes the loss based on the implementation in question
    ///
    /// dont recommend using any kind of parallel computing
    /// to compute this loss as well as rayon is used when averaging losses
    fn compute_loss(
        &self,
        outputs: &Vec<f32>,
        expected_outputs: &Vec<f32>,
    ) -> f32;
    
    /// Computes the derivative of the error with respect to the Model's outputs,
    /// 
    /// dE/dO
    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f32,
        expected_output: f32,
    ) -> f32;

    /// Computes the average of the loss for all of the samples using the
    /// current implementation
    fn average_loss_for_samples(
        &self,
        sample_outputs: &Vec<Vec<f32>>,
        sample_expected_outputs: &Vec<Vec<f32>>,
    ) -> f32 {
        let samples_amount = sample_outputs.len();
        assert_eq!(samples_amount, sample_expected_outputs.len());
        (0..samples_amount)
            .into_par_iter()
            // .zip(sample_expected_outputs)
            .map(|sample_index| {
                self.compute_loss(
                    &sample_outputs[sample_index],
                    &sample_expected_outputs[sample_index],
                )
            })
            .sum::<f32>()
            / sample_outputs.len() as f32
    }
}
