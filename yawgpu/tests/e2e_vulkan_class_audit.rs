#![cfg(feature = "vulkan")]
//! Block 114 S1 (`specs/blocks/114-class-audit-vulkan.md`): read-after-write
//! and write-after-read buffer hazards around uniform buffers (S1, S2) and
//! 2D-array <-> 3D texture copies (F1). Run under the Khronos layer with
//! `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`;
//! the hazards show up as `SYNC-HAZARD-*` lines, F1 also as wrong bytes.
#[path = "common/vulkan.rs"]
mod vulkan;
use vulkan::*;
use yawgpu::native;

const SIZE: u32 = 4;
const LAYERS: u32 = 3;
const BYTES_PER_ROW: u32 = 256;
const LAYER_BYTES: usize = BYTES_PER_ROW as usize * SIZE as usize;

/// S2 + S1: `writeBuffer(uniform)` then a dispatch that reads it, twice in a
/// row. The transfer write must be visible to the uniform read (S2) and the
/// second write must wait for the first dispatch's read (S1).
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_uniform_write_then_dispatch_twice_reads_each_value() {
    with_device(&[], |ctx| unsafe {
        let shader = r#"
struct Params { value: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> out: array<u32, 2>;
override slot: u32 = 0u;
@compute @workgroup_size(1)
fn main() { out[slot] = params.value; }
"#;
        let module = create_wgsl_module(ctx.device, shader);
        let uniform = create_buffer(
            ctx.device,
            16,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let storage = create_buffer(
            ctx.device,
            8,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            ctx.device,
            8,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        // Both updates and both dispatches in ONE command buffer, so the
        // copy -> uniform read (RAW) and uniform read -> next copy (WAR)
        // dependencies are not masked by CPU waits between submissions.
        let mut staging = Vec::new();
        for value in [0x1111_1111u32, 0x2222_2222] {
            let buffer = create_buffer(
                ctx.device,
                16,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            let bytes = [value, 0, 0, 0];
            yawgpu::wgpuQueueWriteBuffer(ctx.queue, buffer, 0, bytes.as_ptr().cast(), 16);
            staging.push(buffer);
        }
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let mut keep = Vec::new();
        for (slot, source) in [0u32, 1].into_iter().zip(staging.iter().copied()) {
            let pipeline = compute_pipeline(ctx.device, module, slot);
            let group = uniform_storage_group(ctx.device, pipeline, uniform, storage);
            yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, uniform, 0, 16);
            let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
            yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            yawgpu::wgpuComputePassEncoderEnd(pass);
            yawgpu::wgpuComputePassEncoderRelease(pass);
            keep.push((pipeline, group));
        }
        ctx.submit(encoder);
        for (pipeline, group) in keep {
            yawgpu::wgpuBindGroupRelease(group);
            yawgpu::wgpuComputePipelineRelease(pipeline);
        }
        for buffer in staging {
            yawgpu::wgpuBufferRelease(buffer);
        }
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 8);
        ctx.submit(encoder);
        let bytes = read_buffer(ctx.instance, readback, 8);
        let words = [
            u32::from_le_bytes(bytes[0..4].try_into().expect("word")),
            u32::from_le_bytes(bytes[4..8].try_into().expect("word")),
        ];
        assert_eq!(words, [0x1111_1111, 0x2222_2222]);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuBufferRelease(uniform);
        yawgpu::wgpuShaderModuleRelease(module);
    });
}

/// S1 + S2 on the render path: a fragment shader reads a uniform, then the
/// next submission overwrites the uniform with `writeBuffer` (write after
/// the render-pass read) and draws again.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_fragment_uniform_read_then_write_buffer_then_draw() {
    with_device(&[], |ctx| unsafe {
        let shader = r#"
struct Params { color: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> { return params.color; }
"#;
        let module = create_wgsl_module(ctx.device, shader);
        let pipeline = create_raster_pipeline(
            ctx.device,
            module,
            native::WGPUFrontFace_CCW,
            native::WGPUCullMode_None,
        );
        let uniform = create_buffer(
            ctx.device,
            16,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        let entry = native::WGPUBindGroupEntry {
            binding: 0,
            buffer: uniform,
            offset: 0,
            size: 16,
            ..std::mem::zeroed()
        };
        let group_descriptor = native::WGPUBindGroupDescriptor {
            layout,
            entryCount: 1,
            entries: &entry,
            ..std::mem::zeroed()
        };
        let group = yawgpu::wgpuDeviceCreateBindGroup(ctx.device, &group_descriptor);
        let texture = create_texture(
            ctx.device,
            native::WGPUTextureDimension_2D,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let readback = create_buffer(
            ctx.device,
            LAYER_BYTES as u64,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let mut staging = Vec::new();
        for color in [[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
            let buffer = create_buffer(
                ctx.device,
                16,
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            );
            yawgpu::wgpuQueueWriteBuffer(ctx.queue, buffer, 0, color.as_ptr().cast(), 16);
            staging.push(buffer);
        }
        // One command buffer: copy -> draw (uniform read) -> copy -> draw.
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        for &source in &staging {
            yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, uniform, 0, 16);
            let pass = begin_color_pass(encoder, view);
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
        ctx.submit(encoder);
        for buffer in staging {
            yawgpu::wgpuBufferRelease(buffer);
        }
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        record_t2b(encoder, texture, readback, 1);
        ctx.submit(encoder);
        let bytes = read_buffer(ctx.instance, readback, 4);
        assert_eq!(
            bytes,
            vec![0, 255, 0, 255],
            "second draw must see the second uniform"
        );
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(uniform);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
    });
}

/// F1: a 3-layer 2D-array texture copied into a 3D texture (depth 3) and
/// back; every layer must survive.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_2d_array_to_3d_and_back_copies_every_layer() {
    with_device(&[], |ctx| unsafe {
        let usage = native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst;
        let source = create_texture(ctx.device, native::WGPUTextureDimension_2D, LAYERS, usage);
        let volume = create_texture(ctx.device, native::WGPUTextureDimension_3D, LAYERS, usage);
        let destination =
            create_texture(ctx.device, native::WGPUTextureDimension_2D, LAYERS, usage);
        let total = LAYER_BYTES * LAYERS as usize;
        let expected: Vec<u8> = (0..total)
            .map(|i| u8::try_from((i * 7 + 1) % 251).expect("byte"))
            .collect();
        let info = native::WGPUTexelCopyTextureInfo {
            texture: source,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: SIZE,
        };
        yawgpu::wgpuQueueWriteTexture(
            ctx.queue,
            &info,
            expected.as_ptr().cast(),
            expected.len(),
            &layout,
            &extent(LAYERS),
        );
        let readback = create_buffer(
            ctx.device,
            total as u64,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        record_t2t(encoder, source, volume);
        record_t2t(encoder, volume, destination);
        record_t2b(encoder, destination, readback, LAYERS);
        ctx.submit(encoder);
        let bytes = read_buffer(ctx.instance, readback, total);
        for layer in 0..LAYERS as usize {
            for row in 0..SIZE as usize {
                let start = layer * LAYER_BYTES + row * BYTES_PER_ROW as usize;
                let end = start + SIZE as usize * 4;
                assert_eq!(
                    &bytes[start..end],
                    &expected[start..end],
                    "layer {layer} row {row} must survive 2D-array -> 3D -> 2D-array"
                );
            }
        }
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(destination);
        yawgpu::wgpuTextureRelease(volume);
        yawgpu::wgpuTextureRelease(source);
    });
}

/// S1: in one command buffer, copy buffer U into a texture (transfer read of
/// U) and then overwrite U with another copy (transfer write). Without a
/// read -> write dependency the second copy may overtake the first.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_buffer_read_by_copy_then_overwritten_in_same_command_buffer() {
    with_device(&[], |ctx| unsafe {
        let total = LAYER_BYTES;
        let upload = create_buffer(
            ctx.device,
            total as u64,
            native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
        );
        let other = create_buffer(
            ctx.device,
            total as u64,
            native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
        );
        let first: Vec<u8> = (0..total)
            .map(|i| u8::try_from(i % 251).expect("byte"))
            .collect();
        let second = vec![0xEEu8; total];
        yawgpu::wgpuQueueWriteBuffer(ctx.queue, upload, 0, first.as_ptr().cast(), total);
        yawgpu::wgpuQueueWriteBuffer(ctx.queue, other, 0, second.as_ptr().cast(), total);
        let texture = create_texture(
            ctx.device,
            native::WGPUTextureDimension_2D,
            1,
            native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
        );
        let readback = create_buffer(
            ctx.device,
            total as u64,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let source = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: SIZE,
            },
            buffer: upload,
        };
        let destination = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &extent(1));
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, other, 0, upload, 0, total as u64);
        record_t2b(encoder, texture, readback, 1);
        ctx.submit(encoder);
        let bytes = read_buffer(ctx.instance, readback, total);
        for row in 0..SIZE as usize {
            let start = row * BYTES_PER_ROW as usize;
            let end = start + SIZE as usize * 4;
            assert_eq!(
                &bytes[start..end],
                &first[start..end],
                "row {row} must hold the first upload"
            );
        }
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuBufferRelease(other);
        yawgpu::wgpuBufferRelease(upload);
    });
}

unsafe fn compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    slot: u32,
) -> native::WGPUComputePipeline {
    let constant = native::WGPUConstantEntry {
        nextInChain: std::ptr::null_mut(),
        key: string_view("slot"),
        value: f64::from(slot),
    };
    let descriptor = native::WGPUComputePipelineDescriptor {
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("main"),
            constantCount: 1,
            constants: &constant,
        },
        ..std::mem::zeroed()
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn uniform_storage_group(
    device: native::WGPUDevice,
    pipeline: native::WGPUComputePipeline,
    uniform: native::WGPUBuffer,
    storage: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let entries = [
        native::WGPUBindGroupEntry {
            binding: 0,
            buffer: uniform,
            offset: 0,
            size: 16,
            ..std::mem::zeroed()
        },
        native::WGPUBindGroupEntry {
            binding: 1,
            buffer: storage,
            offset: 0,
            size: 8,
            ..std::mem::zeroed()
        },
    ];
    let descriptor = native::WGPUBindGroupDescriptor {
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
        ..std::mem::zeroed()
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    assert!(!group.is_null());
    group
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    dimension: native::WGPUTextureDimension,
    depth_or_layers: u32,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        usage,
        dimension,
        size: extent(depth_or_layers),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        ..std::mem::zeroed()
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn record_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let info = |texture| native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(
        encoder,
        &info(source),
        &info(destination),
        &extent(LAYERS),
    );
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
    layers: u32,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: SIZE,
        },
        buffer,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent(layers));
}

fn extent(depth_or_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: SIZE,
        height: SIZE,
        depthOrArrayLayers: depth_or_layers,
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
