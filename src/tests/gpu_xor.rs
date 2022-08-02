// use crate::layers::dense_gpu::DenseGPU;
// use crate::gpu;
// use crate::layers::activations::tanh::TanH;
// use crate::loss_functions::mean_squared::MeanSquared;
// use crate::model::{Model, ModelLayer, TrainingOptions};
//
// #[allow(dead_code)]
// async fn should_decerase_error_test() {
//     let mut layers: Vec<ModelLayer> = Vec::new();
//     layers.push(ModelLayer::DenseGPU(DenseGPU::new(2, 3)));
//     layers.push(ModelLayer::TanH(TanH::new())); 
//     layers.push(ModelLayer::DenseGPU(DenseGPU::new(3, 1))); 
//     layers.push(ModelLayer::TanH(TanH::new())); 
//
//     let mut model = Model::new(layers);
//
//     let training_input_samples = Vec::from([
//         Vec::from([0.0_f32, 0.0_f32]),
//         Vec::from([1.0_f32, 0.0_f32]),
//         Vec::from([1.0_f32, 1.0_f32]),
//         Vec::from([0.0_f32, 1.0_f32]),
//     ]);
//
//     let training_output_samples = Vec::from([
//         Vec::from([0.0_f32]),
//         Vec::from([1.0_f32]),
//         Vec::from([0.0_f32]),
//         Vec::from([1.0_f32]),
//     ]);
//     
//     let (actual_device, actual_queue) = gpu::setup_device_and_queue().await;
//     let device = &Some(actual_device);
//     let queue = &Some(actual_queue);
//
//     let epochs: usize = 3000;
//     let mut last_loss: f32 = 0.0;
//
//     for _ in 0..epochs {
//         last_loss = model.back_propagate(
//             &training_input_samples, 
//             &training_output_samples, 
//             &TrainingOptions {
//                 loss_algorithm: Box::new(MeanSquared),
//                 learning_rate: 0.1,
//                 should_print_information: true,
//                 instantiate_gpu: true,
//                 epochs: 0,
//             },
//             device,
//             queue,
//         ).await;
//     }
//
//     // a maximum value that the loss can be after training to
//     // be considered working
//     let loss_maximum_threshold = 0.1_f32;
//
//     assert!(last_loss <= loss_maximum_threshold);
// }
//
// #[test]
// fn should_decrease_error() {
//     pollster::block_on(should_decerase_error_test());
// }
//