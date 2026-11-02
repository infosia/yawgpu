//! Shared bodies of the Block 125 and 126 e2e (`specs/blocks/125-tint-option-parity.md`,
//! `specs/blocks/126-vulkan-frag-depth-clamp.md`), included by
//! `e2e_{metal,vulkan}_shader_parity.rs` through `#[path]`. Each scenario takes the
//! real backend to probe and the `YAWGPU_INSTANCE_BACKEND_*` selector.

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const DOT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32, 4>;

@compute @workgroup_size(1)
fn main() {
    let a = data[0];
    let b = data[1];
    data[2] = dot4U8Packed(a, b);
    data[3] = bitcast<u32>(dot4I8Packed(a, b));
}
"#;

/// `dot4U8Packed` / `dot4I8Packed` give the WGSL results, whichever way the backend
/// lowers them (Block 125 R2: native instructions or Tint's polyfill).
pub fn packed_dot_products(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // a bytes, low to high: 4, 3, 2, 0xFF; b bytes: 8, 7, 6, 5.
        let words: [u32; 4] = [0xFF02_0304, 0x0506_0708, 0, 0];
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let data = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopyDst
                | native::WGPUBufferUsage_CopySrc,
        );
        yawgpu::wgpuQueueWriteBuffer(queue, data, 0, bytes.as_ptr().cast(), 16);
        let readback = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );

        let module = create_wgsl_module(device, DOT_SHADER);
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            device,
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null_mut(),
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
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: data,
            offset: 0,
            size: 16,
            sampler: std::ptr::null_mut(),
            textureView: std::ptr::null_mut(),
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout,
                entryCount: 1,
                entries: &entry,
            },
        );
        assert!(!bind_group.is_null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, data, 0, readback, 0, 16);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let out = read_buffer(instance, readback, 16);
        let words: Vec<u32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        // Unsigned: 4*8 + 3*7 + 2*6 + 255*5. Signed: 4*8 + 3*7 + 2*6 + (-1)*5.
        assert_eq!(words[2], 1340, "dot4U8Packed");
        assert_eq!(words[3] as i32, 60, "dot4I8Packed");
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(data);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

const DEPTH_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(corners[index], 0.25, 1.0);
}

@fragment
fn fs() -> @builtin(frag_depth) f32 {
    return 0.9;
}
"#;

const DEPTH_SIZE: u32 = 4;

/// A shader-written `frag_depth` above the viewport's `maxDepth` is stored as
/// `maxDepth` (WebGPU clamps it unless `unclippedDepth`; Block 126). The viewport is
/// set once before and once after the pipeline is bound; the second draw comes from a
/// render bundle. On a Vulkan host with `VK_EXT_depth_clip_enable` the hardware clamp
/// also gives these values, so the test only discriminates on MoltenVK and on hosts
/// without that extension.
pub fn frag_depth_is_clamped_to_viewport(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let module = create_wgsl_module(device, DEPTH_SHADER);

        let depth = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
                dimension: native::WGPUTextureDimension_2D,
                size: native::WGPUExtent3D {
                    width: DEPTH_SIZE,
                    height: DEPTH_SIZE,
                    depthOrArrayLayers: 1,
                },
                format: native::WGPUTextureFormat_Depth32Float,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        assert!(!depth.is_null());
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let stencil_face = native::WGPUStencilFaceState {
            compare: native::WGPUCompareFunction_Always,
            failOp: native::WGPUStencilOperation_Keep,
            depthFailOp: native::WGPUStencilOperation_Keep,
            passOp: native::WGPUStencilOperation_Keep,
        };
        let depth_stencil = native::WGPUDepthStencilState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Depth32Float,
            depthWriteEnabled: native::WGPUOptionalBool_True,
            depthCompare: native::WGPUCompareFunction_Always,
            stencilFront: stencil_face,
            stencilBack: stencil_face,
            stencilReadMask: 0xff,
            stencilWriteMask: 0xff,
            depthBias: 0,
            depthBiasSlopeScale: 0.0,
            depthBiasClamp: 0.0,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("fs"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 0,
            targets: std::ptr::null(),
        };
        let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
            device,
            &native::WGPURenderPipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null_mut(),
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
                depthStencil: &depth_stencil,
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

        let half = DEPTH_SIZE as f32 / 2.0;
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: 0,
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: 0,
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
            encoder,
            &native::WGPURenderPassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorAttachmentCount: 0,
                colorAttachments: std::ptr::null(),
                depthStencilAttachment: &depth_attachment,
                occlusionQuerySet: std::ptr::null_mut(),
                timestampWrites: std::ptr::null(),
            },
        );
        // Left half: viewport set before the pipeline, depth range [0, 0.5].
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, half, DEPTH_SIZE as f32, 0.0, 0.5);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        // Right half: viewport changed after the pipeline, depth range [0.1, 0.3],
        // drawn from a render bundle so the bundle replay path is covered too.
        yawgpu::wgpuRenderPassEncoderSetViewport(
            pass,
            half,
            0.0,
            half,
            DEPTH_SIZE as f32,
            0.1,
            0.3,
        );
        let depth_format = native::WGPUTextureFormat_Depth32Float;
        let bundle_encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(
            device,
            &native::WGPURenderBundleEncoderDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorFormatCount: 0,
                colorFormats: std::ptr::null(),
                depthStencilFormat: depth_format,
                sampleCount: 1,
                depthReadOnly: 0,
                stencilReadOnly: 0,
            },
        );
        assert!(!bundle_encoder.is_null());
        yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline);
        yawgpu::wgpuRenderBundleEncoderDraw(bundle_encoder, 3, 1, 0, 0);
        let bundle = yawgpu::wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null());
        yawgpu::wgpuRenderBundleEncoderRelease(bundle_encoder);
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let bytes_per_row = 256;
        let readback = create_buffer(
            device,
            u64::from(bytes_per_row * DEPTH_SIZE),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &native::WGPUTexelCopyTextureInfo {
                texture: depth,
                mipLevel: 0,
                origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                aspect: native::WGPUTextureAspect_DepthOnly,
            },
            &native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: bytes_per_row,
                    rowsPerImage: DEPTH_SIZE,
                },
                buffer: readback,
            },
            &native::WGPUExtent3D {
                width: DEPTH_SIZE,
                height: DEPTH_SIZE,
                depthOrArrayLayers: 1,
            },
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, (bytes_per_row * DEPTH_SIZE) as usize);
        for y in 0..DEPTH_SIZE as usize {
            let row = &bytes[y * bytes_per_row as usize..][..DEPTH_SIZE as usize * 4];
            for (x, texel) in row.as_chunks::<4>().0.iter().enumerate() {
                let value = f32::from_le_bytes(*texel);
                let expected = if (x as f32) < half { 0.5 } else { 0.3 };
                assert!(
                    (value - expected).abs() < 1e-6,
                    "texel ({x}, {y}) = {value}, expected {expected}"
                );
            }
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
