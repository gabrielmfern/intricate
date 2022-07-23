use intricate::layers::activations::tanh::TanHF64;
use intricate::layers::dense::DenseF64;
use intricate::layers::layer::Layer;

use intricate::loss_functions::mean_squared::MeanSquared;
use intricate::model::{ModelF64, TrainingOptionsF64};

fn main() {
    let training_inputs = Vec::from([
        Vec::from([0.0, 0.0]),
        Vec::from([0.0, 1.0]),
        Vec::from([1.0, 0.0]),
        Vec::from([1.0, 1.0]),
    ]);

    let expected_outputs = Vec::from([
        Vec::from([0.0]),
        Vec::from([1.0]),
        Vec::from([1.0]),
        Vec::from([0.0]),
    ]);

    let mut layers: Vec<Box<dyn Layer<f64>>> = Vec::new();

    layers.push(Box::new(DenseF64::new(2, 10)));
    layers.push(Box::new(TanHF64::new()));
    layers.push(Box::new(DenseF64::new(10, 1)));
    layers.push(Box::new(TanHF64::new()));

    let mut xor_model = ModelF64::new(layers);

    let epoch_amount = 1000;

    for epoch_index in 0..epoch_amount {
        println!("epoch #{}", epoch_index + 1);
        
        xor_model.fit(
            &training_inputs, 
            &expected_outputs, 
            TrainingOptionsF64 {
                learning_rate: 0.1,
                loss_algorithm: Box::new(MeanSquared),
                should_print_information: true
            }
        );
    }
}
