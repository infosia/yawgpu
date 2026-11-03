//! Shared body of the Block 132 e2e (`specs/blocks/132-render-bind-groups-at-draw.md`),
//! included by `e2e_{metal,vulkan}_render_bind_groups_at_draw.rs` through `#[path]`.
//!
//! A bind group may be set while a pipeline it does not match is bound, as long as no
//! draw uses that pairing. The pass binds pipeline A, sets a group made for pipeline B,
//! then binds B and draws. The left half is drawn that way in the pass, and the right
//! half the same way inside a render bundle. Each half must show its group's colour.

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const SIZE: u32 = 4;
const FORMAT: native::WGPUTextureFormat = native::WGPUTextureFormat_RGBA8Unorm;

const SHADER: &str = r#"
@group(0) @binding(0) var<uniform> color_a: vec4<f32>;
@group(0) @binding(1) var<uniform> color_b: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs_a() -> @location(0) vec4<f32> {
    return color_a;
}

@fragment
fn fs_b() -> @location(0) vec4<f32> {
    return color_b;
}
"#;

/// Order 1 of the spec's Problem: `setPipeline(A)`, `setBindGroup(0, groupForB)`,
/// `setPipeline(B)`, `draw`, in a pass and in a render bundle. Order 2 (an
/// incompatible pairing with no draw) is the first half of each sequence.
pub fn group_set_under_another_pipeline_draws_with_the_next(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let module = create_wgsl_module(device, SHADER);

        // Layout A declares only binding 0, layout B only binding 1, so a group made
        // for B lacks the binding pipeline A declares.
        let layout_a = uniform_layout(device, 0);
        let layout_b = uniform_layout(device, 1);
        let pipeline_a = create_pipeline(device, module, layout_a, "fs_a");
        let pipeline_b = create_pipeline(device, module, layout_b, "fs_b");

        let green = uniform_buffer(device, queue, [0.0, 1.0, 0.0, 1.0]);
        let blue = uniform_buffer(device, queue, [0.0, 0.0, 1.0, 1.0]);
        let group_green = uniform_group(device, layout_b, 1, green);
        let group_blue = uniform_group(device, layout_b, 1, blue);

        let target = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
                dimension: native::WGPUTextureDimension_2D,
                size: native::WGPUExtent3D {
                    width: SIZE,
                    height: SIZE,
                    depthOrArrayLayers: 1,
                },
                format: FORMAT,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        assert!(!target.is_null());
        let view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());

        let bundle_encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(
            device,
            &native::WGPURenderBundleEncoderDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorFormatCount: 1,
                colorFormats: &FORMAT,
                depthStencilFormat: native::WGPUTextureFormat_Undefined,
                sampleCount: 1,
                depthReadOnly: 0,
                stencilReadOnly: 0,
            },
        );
        assert!(!bundle_encoder.is_null());
        yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline_a);
        yawgpu::wgpuRenderBundleEncoderSetBindGroup(
            bundle_encoder,
            0,
            group_blue,
            0,
            std::ptr::null(),
        );
        yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline_b);
        yawgpu::wgpuRenderBundleEncoderDraw(bundle_encoder, 3, 1, 0, 0);
        let bundle = yawgpu::wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null());
        assert!(!bundle.is_null());
        yawgpu::wgpuRenderBundleEncoderRelease(bundle_encoder);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null_mut(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
            encoder,
            &native::WGPURenderPassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorAttachmentCount: 1,
                colorAttachments: &color_attachment,
                depthStencilAttachment: std::ptr::null(),
                occlusionQuerySet: std::ptr::null_mut(),
                timestampWrites: std::ptr::null(),
            },
        );
        let half = SIZE as f32 / 2.0;
        // Left half, in the pass.
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, half, SIZE as f32, 0.0, 1.0);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_a);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group_green, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_b);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        // Right half, from the bundle.
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, half, 0.0, half, SIZE as f32, 0.0, 1.0);
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let bytes_per_row = 256;
        let readback = create_buffer(
            device,
            u64::from(bytes_per_row * SIZE),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &native::WGPUTexelCopyTextureInfo {
                texture: target,
                mipLevel: 0,
                origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                aspect: native::WGPUTextureAspect_All,
            },
            &native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: bytes_per_row,
                    rowsPerImage: SIZE,
                },
                buffer: readback,
            },
            &native::WGPUExtent3D {
                width: SIZE,
                height: SIZE,
                depthOrArrayLayers: 1,
            },
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, (bytes_per_row * SIZE) as usize);
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");
        for y in 0..SIZE as usize {
            let row = &bytes[y * bytes_per_row as usize..][..SIZE as usize * 4];
            for (x, texel) in row.as_chunks::<4>().0.iter().enumerate() {
                let expected = if (x as f32) < half {
                    [0, 255, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                assert_eq!(*texel, expected, "texel ({x}, {y})");
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBindGroupRelease(group_green);
        yawgpu::wgpuBindGroupRelease(group_blue);
        yawgpu::wgpuBufferRelease(green);
        yawgpu::wgpuBufferRelease(blue);
        yawgpu::wgpuRenderPipelineRelease(pipeline_a);
        yawgpu::wgpuRenderPipelineRelease(pipeline_b);
        yawgpu::wgpuPipelineLayoutRelease(layout_a.1);
        yawgpu::wgpuPipelineLayoutRelease(layout_b.1);
        yawgpu::wgpuBindGroupLayoutRelease(layout_a.0);
        yawgpu::wgpuBindGroupLayoutRelease(layout_b.0);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// A bind group layout with one fragment-visible uniform buffer at `binding`, and a
/// pipeline layout with it as group 0.
unsafe fn uniform_layout(
    device: native::WGPUDevice,
    binding: u32,
) -> (native::WGPUBindGroupLayout, native::WGPUPipelineLayout) {
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Fragment,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
            hasDynamicOffset: 0,
            minBindingSize: 16,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    };
    let group_layout = yawgpu::wgpuDeviceCreateBindGroupLayout(
        device,
        &native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 1,
            entries: &entry,
        },
    );
    assert!(!group_layout.is_null());
    let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(
        device,
        &native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &group_layout,
            immediateSize: 0,
        },
    );
    assert!(!pipeline_layout.is_null());
    (group_layout, pipeline_layout)
}

unsafe fn create_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: (native::WGPUBindGroupLayout, native::WGPUPipelineLayout),
    fragment_entry: &str,
) -> native::WGPURenderPipeline {
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: FORMAT,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view(fragment_entry),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &target,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
        device,
        &native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: layout.1,
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: string_view("vs"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: native::WGPUPrimitiveState {
                nextInChain: std::ptr::null_mut(),
                topology: native::WGPUPrimitiveTopology_TriangleList,
                stripIndexFormat: native::WGPUIndexFormat_Undefined,
                frontFace: native::WGPUFrontFace_CCW,
                cullMode: native::WGPUCullMode_None,
                unclippedDepth: 0,
            },
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: u32::MAX,
                alphaToCoverageEnabled: 0,
            },
            fragment: &fragment,
        },
    );
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn uniform_buffer(
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    color: [f32; 4],
) -> native::WGPUBuffer {
    let buffer = create_buffer(
        device,
        16,
        native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
    );
    let bytes: Vec<u8> = color.iter().flat_map(|c| c.to_le_bytes()).collect();
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, bytes.as_ptr().cast(), bytes.len());
    buffer
}

unsafe fn uniform_group(
    device: native::WGPUDevice,
    layout: (native::WGPUBindGroupLayout, native::WGPUPipelineLayout),
    binding: u32,
    buffer: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset: 0,
        size: 16,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(
        device,
        &native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: layout.0,
            entryCount: 1,
            entries: &entry,
        },
    );
    assert!(!group.is_null());
    group
}
