use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::layers::layer::Layer;

pub trait ActivationLayerF64: Layer<f64>
where Self: Sync + Send {
    fn function(inputs: &Vec<f64>) -> Vec<f64>;

    /// this function is so different from the normal activation
    /// mostly because of activations like softmax that the differential
    /// gets quite annoying to compute
    /// but usual functions such as ReLUF64 are also computable here
    ///
    /// dont recommend using rayon nor any type of multiprocessing because
    /// base_back_propagate already uses multiprocessing
    fn differential_of_output_with_respect_to_input(
        &self,
        sample_index: usize,
        input_index: usize,
        output_index: usize,
    ) -> f64;

    fn set_last_inputs(&mut self, input_samples: &Vec<Vec<f64>>);
    fn set_last_outputs(&mut self, output_samples: &Vec<Vec<f64>>);

    fn base_propagate(&mut self, input_samples: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        assert_eq!(self.get_inputs_amount(), self.get_outputs_amount());
        self.set_last_inputs(input_samples);
        
        self.set_last_outputs(
        &input_samples
        .par_iter()
        .map(|inputs| Self::function(inputs))
            .collect::<Vec<Vec<f64>>>(),
        );
            
        self.get_last_outputs()
    }

    fn base_back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f64>>,
        _: f64,
    ) -> Option<Vec<Vec<f64>>> {
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
                                    .sum::<f64>()
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>(),
            )
        } else {
            None
        }
    }
}

pub trait ActivationLayerF32: Layer<f32>
where Self: Sync + Send {
    fn function(inputs: &Vec<f32>) -> Vec<f32>;

    /// this function is so different from the normal activation
    /// mostly because of activations like softmax that the differential
    /// gets quite annoying to compute
    /// but usual functions such as ReLUF32 are also computable here
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

    fn base_propagate(&mut self, input_samples: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        assert_eq!(self.get_inputs_amount(), self.get_outputs_amount());
        self.set_last_inputs(input_samples);
        
        self.set_last_outputs(
        &input_samples
        .par_iter()
        .map(|inputs| Self::function(inputs))
            .collect::<Vec<Vec<f32>>>(),
        );
            
        self.get_last_outputs()
    }

    fn base_back_propagate(
        &mut self,
        should_calculate_input_to_error_derivative: bool,
        layer_output_to_error_derivative: &Vec<Vec<f32>>,
        _: f64,
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