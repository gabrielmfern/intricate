use crate::loss_functions::loss_function::LossFunctionF64;

pub struct CategoricalCrossEntropy;

impl LossFunctionF64 for CategoricalCrossEntropy {
    fn compute_loss(&self, outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> f64 {
        let outputs_amount = outputs.len();
        assert_eq!(outputs_amount, expected_outputs.len());
        -outputs
            .iter()
            .zip(expected_outputs)
            .map(|(output, expected_output)| expected_output * output.ln().max(0.00000001_f64))
            .sum::<f64>()
            / outputs_amount as f64
    }

    fn compute_loss_derivative_with_respect_to(&self, output: f64, expected_output: f64) -> f64 {
        -expected_output / output.max(0.00000001)
    }
}
