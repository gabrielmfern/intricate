use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::gpu::{
    make_compute_storage_bind_group_layout_entry, make_compute_uniform_bind_group_layout_entry,
};

use crate::layers::dense_gpu::DenseGpuF32;
#[allow(unused_imports)]
use crate::layers::layer::Layer;
use crate::utils::vector_operations::VectorOperations;

#[allow(dead_code)]
pub async fn apply_gradients_to_f32_dense_weights(
    dense: &mut DenseGpuF32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layer_output_to_error_derivatives: &Vec<Vec<f32>>,
    learning_rate: f32,
) -> () {
    let flattened_layer_output_to_error_derivatives = layer_output_to_error_derivatives
        .par_iter()
        .map(|x| x.to_vec())
        .flatten()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let flattened_layer_weights = dense
        .weights
        .par_iter()
        .map(|x| x.to_vec())
        .flatten()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    
    let flattened_layer_inputs = dense
        .last_inputs
        .par_iter()
        .map(|x| x.to_vec())
        .flatten()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();

    let samples_amount = dense.last_inputs.len();

    dense.weights = execute_gpu_code(
        &device,
        &queue,
        learning_rate,
        samples_amount,
        dense.outputs_amount,
        dense.inputs_amount,
        flattened_layer_inputs.as_slice(),
        flattened_layer_output_to_error_derivatives.as_slice(),
        flattened_layer_weights.as_slice()
    )
    .await.expect("Some error happenned while trying to compute gradients for the weights in DenseGPUF32 layer");
}

async fn execute_gpu_code(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    learning_rate: f32,
    samples_amount: usize,
    outputs_amount: usize,
    inputs_amount: usize,
    flattened_layer_inputs: &[f32],
    flattened_layer_output_to_error_derivatives: &[f32],
    flattened_layer_weights: &[f32]
) -> Option<Vec<Vec<f32>>> {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/apply_gradients_to_dense_weights.wgsl"))),
    });

    let weights_size = inputs_amount * outputs_amount * std::mem::size_of::<f32>();
    let buffer_size = weights_size as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_WRITE
            | wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let flattened_layer_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flattened_layer_weights"),
        contents: bytemuck::cast_slice(&flattened_layer_weights),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let flattened_layer_output_to_error_derivatives_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flattened_layer_output_to_error_derivatives"),
        contents: bytemuck::cast_slice(&flattened_layer_output_to_error_derivatives),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let flattened_layer_inputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flattened_layer_inputs"),
        contents: bytemuck::cast_slice(&flattened_layer_inputs),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let learning_rate_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("learning_rate"),
        contents: &(learning_rate as f32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let inputs_amount_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("inputs_amount"),
        contents: &(inputs_amount as u32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let outputs_amount_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("outputs_amount"),
        contents: &(outputs_amount as u32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let samples_amount_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("samples_amount"),
        contents: &(samples_amount as u32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let samples_amount_float_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("samples_amount_float"),
        contents: &(samples_amount as f32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
        label: Some("Apply Gradients to Dense Weights Pipeline Bind Group Layout"), 
        entries: &[
            make_compute_storage_bind_group_layout_entry(0, true),
            make_compute_storage_bind_group_layout_entry(1, true),
            make_compute_storage_bind_group_layout_entry(2, false),
            make_compute_uniform_bind_group_layout_entry(3),
            make_compute_uniform_bind_group_layout_entry(4),
            make_compute_uniform_bind_group_layout_entry(5),
            make_compute_uniform_bind_group_layout_entry(6),
            make_compute_uniform_bind_group_layout_entry(7),
        ] 
    });

    let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Apply Gradients to Dense Weights Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[]
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&compute_layout),
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: flattened_layer_output_to_error_derivatives_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: flattened_layer_inputs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: flattened_layer_weights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: learning_rate_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: inputs_amount_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: outputs_amount_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: samples_amount_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: samples_amount_float_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute gradients iteration");
        cpass.dispatch_workgroups(
            inputs_amount as u32,
            outputs_amount as u32,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&flattened_layer_weights_buffer, 0, &staging_buffer, 0, buffer_size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);

        let flattened_new_weights: Vec<f32> = result.to_vec();

        drop(data);

        let new_weights: Vec<Vec<f32>> = (0..inputs_amount)
            .into_par_iter()
            .map(|input_index| {
                let row_part = input_index * outputs_amount;
                (0..outputs_amount)
                    .into_iter()
                    .map(|output_index| flattened_new_weights[row_part + output_index])
                    .collect()
            })
            .collect();
        
        drop(flattened_new_weights);

        staging_buffer.unmap();
        Some(new_weights)
    } else {
        panic!("failed to run compute gradients on gpu!")
    }
}