#![allow(dead_code)]

use std::sync::Arc;

use crate::*;

/// Returns noop adapter.
pub(crate) fn noop_adapter() -> Adapter {
    Instance::new_noop()
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("Noop adapter must exist")
}

/// Returns noop device.
pub(crate) fn noop_device() -> Device {
    let instance = Instance::new_noop();
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("Noop adapter must exist");
    adapter
        .create_device(None, &[], "", "")
        .expect("Noop device creation")
}

/// Returns HAL noop adapter.
pub(crate) fn hal_noop_adapter() -> yawgpu_hal::HalAdapter {
    yawgpu_hal::HalInstance::new_noop()
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("Noop HAL adapter must exist")
}

/// Returns HAL noop device.
pub(crate) fn hal_noop_device() -> yawgpu_hal::HalDevice {
    hal_noop_adapter()
        .create_device()
        .expect("Noop HAL device creation")
}

/// Returns HAL noop queue.
pub(crate) fn hal_noop_queue() -> yawgpu_hal::HalQueue {
    hal_noop_device().queue()
}

/// Returns rgba8 unorm.
pub(crate) fn rgba8_unorm() -> TextureFormat {
    TextureFormat::from_raw(0x0000_0016)
}

/// Returns valid texture descriptor.
pub(crate) fn valid_texture_descriptor() -> TextureDescriptor {
    TextureDescriptor {
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        format: rgba8_unorm(),
        mip_level_count: 1,
        sample_count: 1,
        view_formats: Vec::new(),
    }
}

/// Returns texture descriptor 4x4.
pub(crate) fn texture_descriptor_4x4() -> TextureDescriptor {
    TextureDescriptor {
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        format: rgba8_unorm(),
        mip_level_count: 1,
        sample_count: 1,
        view_formats: Vec::new(),
    }
}

/// Returns layered mipped texture descriptor.
pub(crate) fn layered_mipped_texture_descriptor() -> TextureDescriptor {
    TextureDescriptor {
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 3,
        },
        format: rgba8_unorm(),
        mip_level_count: 3,
        sample_count: 1,
        view_formats: Vec::new(),
    }
}

/// Returns noop texture.
pub(crate) fn noop_texture() -> Texture {
    noop_device().create_texture(texture_descriptor_4x4())
}

/// Returns noop buffer.
pub(crate) fn noop_buffer(size: u64, usage: BufferUsage) -> Buffer {
    noop_device().create_buffer(BufferDescriptor {
        usage,
        size,
        mapped_at_creation: false,
    })
}

/// Returns noop render attachment.
pub(crate) fn noop_render_attachment(device: &Device) -> Arc<TextureView> {
    let texture = device.create_texture(TextureDescriptor {
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        format: rgba8_unorm(),
        mip_level_count: 1,
        sample_count: 1,
        view_formats: Vec::new(),
    });
    let (view, error) = texture.create_view(TextureViewDescriptor {
        format: None,
        dimension: None,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        aspect: None,
    });
    assert_eq!(error, None);
    Arc::new(view)
}

/// Returns noop render pass descriptor.
pub(crate) fn noop_render_pass_descriptor(
    view: Arc<TextureView>,
    occlusion_query_set: Option<QuerySet>,
) -> RenderPassDescriptor {
    RenderPassDescriptor {
        max_color_attachments: Limits::DEFAULT.max_color_attachments,
        color_attachments: vec![Some(RenderPassColorAttachment {
            view,
            resolve_target: None,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_value: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set,
        timestamp_writes: None,
    }
}

/// Returns noop compute pipeline.
pub(crate) fn noop_compute_pipeline(device: &Device) -> Arc<ComputePipeline> {
    Arc::new(
        device.create_compute_pipeline(compute_pipeline_descriptor(compute_shader_module(device))),
    )
}

/// Returns noop render pipeline.
pub(crate) fn noop_render_pipeline(device: &Device) -> Arc<RenderPipeline> {
    Arc::new(
        device.create_render_pipeline(render_pipeline_descriptor(render_shader_module(device))),
    )
}

/// Returns empty bind group.
pub(crate) fn empty_bind_group(device: &Device) -> Arc<BindGroup> {
    let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
        entries: Vec::new(),
        error: None,
    }));
    Arc::new(device.create_bind_group(layout, Vec::new()))
}

/// Returns noop indirect buffer.
pub(crate) fn noop_indirect_buffer(device: &Device) -> Arc<Buffer> {
    Arc::new(device.create_buffer(BufferDescriptor {
        usage: BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        size: 20,
        mapped_at_creation: false,
    }))
}

/// Returns render bundle encoder descriptor.
pub(crate) fn render_bundle_encoder_descriptor() -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor {
        max_color_attachments: Limits::DEFAULT.max_color_attachments,
        color_formats: vec![Some(rgba8_unorm())],
        depth_stencil_format: None,
        sample_count: 1,
        depth_read_only: false,
        stencil_read_only: false,
    }
}

/// Returns compute shader module.
pub(crate) fn compute_shader_module(device: &Device) -> Arc<ShaderModule> {
    Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
        "@compute @workgroup_size(1) fn cs() {}".to_owned(),
    )))
}

/// Returns compute pipeline descriptor.
pub(crate) fn compute_pipeline_descriptor(module: Arc<ShaderModule>) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        layout: ComputePipelineLayout::Auto,
        shader_module: module,
        entry_point: Some("cs".to_owned()),
        constants: Vec::new(),
        error: None,
    }
}

/// Returns render shader module.
pub(crate) fn render_shader_module(device: &Device) -> Arc<ShaderModule> {
    Arc::new(
        device.create_shader_module(ShaderModuleSource::Wgsl(
            r"
@vertex
fn vs() -> @builtin(position) vec4<f32> {
return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"
            .to_owned(),
        )),
    )
}

/// Returns render pipeline descriptor.
pub(crate) fn render_pipeline_descriptor(module: Arc<ShaderModule>) -> RenderPipelineDescriptor {
    RenderPipelineDescriptor {
        layout: RenderPipelineLayout::Auto,
        vertex: RenderPipelineVertexState {
            shader: RenderPipelineShaderStage {
                module: module.clone(),
                entry_point: Some("vs".to_owned()),
                constants: Vec::new(),
            },
            buffer_count: 0,
            buffers: Vec::new(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u32::MAX,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(RenderPipelineFragmentState {
            shader: RenderPipelineShaderStage {
                module,
                entry_point: Some("fs".to_owned()),
                constants: Vec::new(),
            },
            target_count: 1,
            targets: vec![ColorTargetState {
                format: rgba8_unorm(),
                blend: false,
                write_mask: 0xF,
            }],
        }),
        error: None,
    }
}
