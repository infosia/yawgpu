//! Shared body of the Block 148 R2 e2e (`specs/blocks/148-shader-passthrough-review.md`),
//! included by `e2e_metal_compute_binding_slots.rs` and
//! `e2e_vulkan_compute_binding_slots.rs` through `#[path]`. A compute pipeline's
//! explicit layout may also make entries visible to the vertex or fragment stage. Each
//! binding must still reach its own resource.

use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> out0: array<u32, 2>;
@group(0) @binding(1) var<storage, read_write> out1: array<u32, 2>;
@group(0) @binding(2) var<storage, read_write> out2: array<u32, 2>;
@group(0) @binding(3) var<storage, read> in3: array<u32, 2>;

@compute @workgroup_size(1)
fn main() {
    out0[0] = 11u;
    out1[0] = 22u;
    out2[0] = 33u;
    out0[1] = in3[0];
    out1[1] = in3[1];
    out2[1] = in3[0] + in3[1];
}
"#;

/// Bindings 0 and 2 are compute-only, binding 1 is also fragment-visible, and the
/// read-only binding 3 is also vertex-visible. The shader writes a distinct value to
/// each output and copies the input; every buffer reads back its own values.
pub fn compute_bindings_with_render_visibility_reach_their_buffers(
    backend: RealBackend,
    select: u32,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let visibilities = [
            native::WGPUShaderStage_Compute,
            native::WGPUShaderStage_Compute | native::WGPUShaderStage_Fragment,
            native::WGPUShaderStage_Compute,
            native::WGPUShaderStage_Compute | native::WGPUShaderStage_Vertex,
        ];
        let mut entries: Vec<native::WGPUBindGroupLayoutEntry> = Vec::new();
        for (binding, visibility) in visibilities.iter().enumerate() {
            let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
            entry.binding = binding as u32;
            entry.visibility = *visibility;
            entry.buffer.type_ = if binding == 3 {
                native::WGPUBufferBindingType_ReadOnlyStorage
            } else {
                native::WGPUBufferBindingType_Storage
            };
            entries.push(entry);
        }
        let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(
            device,
            &native::WGPUBindGroupLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                entryCount: entries.len(),
                entries: entries.as_ptr(),
            },
        );
        assert!(!bgl.is_null());
        let layout = yawgpu::wgpuDeviceCreatePipelineLayout(
            device,
            &native::WGPUPipelineLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                bindGroupLayoutCount: 1,
                bindGroupLayouts: &bgl,
                immediateSize: 0,
            },
        );
        assert!(!layout.is_null());

        let module = create_wgsl_module(device, SHADER);
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            device,
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout,
                compute: native::WGPUComputeState {
                    nextInChain: std::ptr::null_mut(),
                    module,
                    entryPoint: string_view("main"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                },
            },
        );
        assert!(!pipeline.is_null());

        let storage = native::WGPUBufferUsage_Storage
            | native::WGPUBufferUsage_CopySrc
            | native::WGPUBufferUsage_CopyDst;
        let buffers: Vec<native::WGPUBuffer> =
            (0..4).map(|_| create_buffer(device, 8, storage)).collect();
        let input = [5u32, 7u32];
        yawgpu::wgpuQueueWriteBuffer(queue, buffers[3], 0, input.as_ptr().cast(), 8);

        let bind_entries: Vec<native::WGPUBindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, &buffer)| native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: binding as u32,
                buffer,
                offset: 0,
                size: 8,
                sampler: std::ptr::null(),
                textureView: std::ptr::null(),
            })
            .collect();
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: bgl,
                entryCount: bind_entries.len(),
                entries: bind_entries.as_ptr(),
            },
        );
        assert!(!bind_group.is_null());

        let readbacks: Vec<native::WGPUBuffer> = (0..3)
            .map(|_| {
                create_buffer(
                    device,
                    8,
                    native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
                )
            })
            .collect();
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        for (index, &readback) in readbacks.iter().enumerate() {
            yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
                encoder,
                buffers[index],
                0,
                readback,
                0,
                8,
            );
        }
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let actual: Vec<[u32; 2]> = readbacks
            .iter()
            .map(|&readback| {
                let bytes = read_buffer(instance, readback, 8);
                [
                    u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                ]
            })
            .collect();
        assert_eq!(actual, vec![[11, 5], [22, 7], [33, 12]]);
        assert_no_errors(&errors);

        for buffer in buffers.into_iter().chain(readbacks) {
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn assert_no_errors(errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>) {
    let errors = errors.lock().expect("error lock");
    assert!(errors.is_empty(), "unexpected device errors: {errors:?}");
}

const RENDER_SHADER: &str = r#"
@group(0) @binding(1) var<uniform> color: vec4<f32>;

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
fn fs() -> @location(0) vec4<f32> {
    return color;
}
"#;

/// A render pipeline whose layout also has compute-only uniform buffers. Bindings 0
/// and 2 (compute-only) hold red and binding 1 (fragment-visible) holds green; the
/// fragment shader reads binding 1. The left half is drawn in the pass and the right
/// half through a render bundle; the whole target is green.
pub fn render_ignores_compute_only_bindings(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        const SIZE: u32 = 4;
        const FORMAT: native::WGPUTextureFormat = native::WGPUTextureFormat_RGBA8Unorm;
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let mut entries: Vec<native::WGPUBindGroupLayoutEntry> = Vec::new();
        for (binding, visibility) in [
            native::WGPUShaderStage_Compute,
            native::WGPUShaderStage_Fragment,
            native::WGPUShaderStage_Compute,
        ]
        .into_iter()
        .enumerate()
        {
            let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
            entry.binding = binding as u32;
            entry.visibility = visibility;
            entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
            entries.push(entry);
        }
        let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(
            device,
            &native::WGPUBindGroupLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                entryCount: entries.len(),
                entries: entries.as_ptr(),
            },
        );
        assert!(!bgl.is_null());
        let layout = yawgpu::wgpuDeviceCreatePipelineLayout(
            device,
            &native::WGPUPipelineLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                bindGroupLayoutCount: 1,
                bindGroupLayouts: &bgl,
                immediateSize: 0,
            },
        );
        assert!(!layout.is_null());

        let module = create_wgsl_module(device, RENDER_SHADER);
        let target_state = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: FORMAT,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("fs"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: &target_state,
        };
        let mut descriptor: native::WGPURenderPipelineDescriptor = std::mem::zeroed();
        descriptor.label = empty_string_view();
        descriptor.layout = layout;
        descriptor.vertex.module = module;
        descriptor.vertex.entryPoint = string_view("vs");
        descriptor.primitive.topology = native::WGPUPrimitiveTopology_TriangleList;
        descriptor.primitive.frontFace = native::WGPUFrontFace_CCW;
        descriptor.primitive.cullMode = native::WGPUCullMode_None;
        descriptor.multisample.count = 1;
        descriptor.multisample.mask = u32::MAX;
        descriptor.fragment = &fragment;
        let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
        assert!(!pipeline.is_null());

        let colors = [
            [1.0f32, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let buffers: Vec<native::WGPUBuffer> = colors
            .iter()
            .map(|color| {
                let buffer = create_buffer(
                    device,
                    16,
                    native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
                );
                yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, color.as_ptr().cast(), 16);
                buffer
            })
            .collect();
        let bind_entries: Vec<native::WGPUBindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, &buffer)| native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: binding as u32,
                buffer,
                offset: 0,
                size: 16,
                sampler: std::ptr::null(),
                textureView: std::ptr::null(),
            })
            .collect();
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: bgl,
                entryCount: bind_entries.len(),
                entries: bind_entries.as_ptr(),
            },
        );
        assert!(!bind_group.is_null());

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
        yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline);
        yawgpu::wgpuRenderBundleEncoderSetBindGroup(
            bundle_encoder,
            0,
            bind_group,
            0,
            std::ptr::null(),
        );
        yawgpu::wgpuRenderBundleEncoderDraw(bundle_encoder, 3, 1, 0, 0);
        let bundle = yawgpu::wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null());
        assert!(!bundle.is_null());
        yawgpu::wgpuRenderBundleEncoderRelease(bundle_encoder);

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
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null_mut(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
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
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, half, SIZE as f32, 0.0, 1.0);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, half, 0.0, half, SIZE as f32, 0.0, 1.0);
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let bytes_per_row = 256u32;
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
        assert_no_errors(&errors);
        for y in 0..SIZE as usize {
            let row = &bytes[y * bytes_per_row as usize..][..SIZE as usize * 4];
            for (x, texel) in row.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(*texel, [0, 255, 0, 255], "texel ({x}, {y})");
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBindGroupRelease(bind_group);
        for buffer in buffers {
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
