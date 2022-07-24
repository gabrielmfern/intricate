pub mod apply_gradients_to_dense_weights;

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