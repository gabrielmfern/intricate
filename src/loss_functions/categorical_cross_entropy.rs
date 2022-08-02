use crate::loss_functions::LossFunction;

#[derive(Debug)]
/// The Categorical Cross Entropy loss function, as the name may suggest
/// it is very good for categorical problems because it 'punishes' wrong
/// values very much, and even being a bit far away from the proper output
/// will yield very large cost values
pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy {
    fn compute_loss(&self, outputs: &Vec<f32>, expected_outputs: &Vec<f32>) -> f32 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());
        -outputs
            .iter()
            .zip(expected_outputs)
            .map(|(output, expected_output)| expected_output * output.ln())
            .sum::<f32>()
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        _: usize,
        output: f32,
        expected_output: f32,
    ) -> f32 {
        expected_output / output
    }
}