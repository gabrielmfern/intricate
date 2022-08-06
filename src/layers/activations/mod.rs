pub mod softmax;
pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub mod tanh_gpu;

mod tests;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::layers::Layer;

/// A type of layer that is used for some kind of function to be applied
/// to all outputs of the last layer, such as the SoftMax, ReLU and etc.
pub trait ActivationLayer: Layer
where Self: Sync + Send {
    /// The actual activation function calculation should go here,
    /// this function receives all of the inputs at once instead of just one x
    /// because of activations functions such as SoftMax
    ///
    /// dont recommend using rayon nor any type of multiprocessing because
    /// base_back_propagate already uses multiprocessing
    fn function(inputs: &Vec<f32>) -> Vec<f32>;

    /// The derivative of the activation with respect to the input
    /// useful so that, using the chain rule, the derivative of the loss
    /// with respect to the input of this activation layer can be calculated
    ///
    /// this function is so different from the 'function'
    /// mostly because of activations like softmax that the differential
    /// gets quite annoying to compute
    /// but usual functions such as ReLU are also computable here
    ///
    /// dont recommend using rayon nor any type of multiprocessing because
    /// base_back_propagate already uses multiprocessing
    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        output_index: usize,
    ) -> f32;

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f32>>);
    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f32>>);

    /// A base method made to be called at the implementation of Layer
    /// for the struct that implements Self trait so that there is not
    /// that much repeated code everywhere
    fn base_propagate(&mut self, input_samples: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        assert_eq!(self.get_inputs_amount(), self.get_outputs_amount());
        self.set_last_inputs(input_samples);
        
        self.set_last_outputs(
        &input_samples
            .par_iter()
            .map(|inputs| Self::function(inputs))
                .collect::<Vec<Vec<f32>>>(),
            );
            
        self.get_last_outputs().to_vec()
    }

    /// A base method made to be called at the implementation of Layer
    /// for the struct that implements Self trait so that there is not
    /// that much repeated code everywhere
    fn base_back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        _: f32,
    ) -> Option<Vec<Vec<f32>>> {
        assert_eq!(self.get_inputs_amount(), self.get_outputs_amount());

        if should_calculate_input_to_error_derivative {
            Some(
                layer_output_to_error_derivative
                    .par_iter()
                    .zip(self.get_last_inputs())
                    .enumerate()
                    .map(|(sample_index, (output_derivatives, inputs))| {
                        inputs
                            .iter()
                            .enumerate()
                            .map(|(l, _)| {
                                output_derivatives
                                    .iter()
                                    .enumerate()
                                    .map(|(j, output_derivative)| {
                                        self.differential_of_output_with_respect_to_input(
                                            sample_index,
                                            l,
                                            j,
                                        ) * output_derivative
                                    })
                                    .sum::<f32>()
                            })
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<Vec<f32>>>(),
            )
        } else {
            None
        }
    }
}