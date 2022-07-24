use crate::loss_functions::loss_function::LossFunctionF64;

#[derive(Debug)]
pub struct CategoricalCrossEntropy;

impl LossFunctionF64 for CategoricalCrossEntropy {
    fn compute_loss(&self, outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> f64 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());
        -outputs
            .iter()
            .zip(expected_outputs)
            .map(|(output, expected_output)| expected_output * output.ln())
            .sum::<f64>()
    }

    fn compute_loss_derivative_with_respect_to_output(
        &self,
        _: usize,
        output: f64,
        expected_output: f64,
    ) -> f64 {
        -expected_output / output
    }
}
