use std::fmt::Debug;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait LossFunctionF64
where 
    Self: Sync + Send + Debug
{
    /// dont recommend using any kind of parallel computing
    /// to compute this loss as well as rayon is used when averaging losses
    fn compute_loss(
        &self,
        outputs: &Vec<f64>,
        expected_outputs: &Vec<f64>,
    ) -> f64;
    
    fn compute_loss_derivative_with_respect_to_output(
        &self,
        ouputs_amount: usize,
        output: f64,
        expected_output: f64,
    ) -> f64;

    fn average_loss_for_samples(
        &self,
        sample_outputs: &Vec<Vec<f64>>,
        sample_expected_outputs: &Vec<Vec<f64>>,
    ) -> f64 {
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
            .sum::<f64>()
            / sample_outputs.len() as f64
    }
}
