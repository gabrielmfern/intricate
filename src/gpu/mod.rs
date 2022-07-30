pub mod apply_gradients_to_dense_weights;
pub mod calculate_dense_input_to_error_derivatives;
pub mod propagate_through_weights_and_biases;

pub fn make_compute_uniform_bind_group_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn make_compute_storage_bind_group_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub async fn setup_device_and_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .unwrap();

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = 1024 << 20;
    limits.max_dynamic_storage_buffers_per_pipeline_layout = 8;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::default(),
                limits,
            },
            None,
        )
        .await
        .unwrap();

    let info = adapter.get_info();

    if info.vendor == 0x10005 {
        panic!();
    }

    (device, queue)
}