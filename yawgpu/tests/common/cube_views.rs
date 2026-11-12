//! Shared body of the Block 155 e2e (`specs/blocks/155-vulkan-cube-compatible-images.md`),
//! included by `e2e_metal_cube_views.rs` and `e2e_vulkan_cube_views.rs` through `#[path]`.
//! A 2D array texture is sampled through a `cube` or `cube-array` view; each face must
//! return the value written to its own array layer.

use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const SIZE: u32 = 4;

const CUBE_SHADER: &str = r#"
@group(0) @binding(0) var cubeTex: texture_cube<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<u32, 6>;

@compute @workgroup_size(1)
fn main() {
    var dirs = array<vec3f, 6>(
        vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
    for (var i = 0u; i < 6u; i++) {
        let c = textureSampleLevel(cubeTex, samp, dirs[i], 0.0);
        out[i] = u32(round(c.r * 255.0));
    }
}
"#;

const CUBE_ARRAY_SHADER: &str = r#"
@group(0) @binding(0) var cubeTex: texture_cube_array<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<u32, 6>;

@compute @workgroup_size(1)
fn main() {
    var dirs = array<vec3f, 6>(
        vec3f(1.0, 0.0, 0.0), vec3f(-1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0),
        vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0));
    for (var i = 0u; i < 6u; i++) {
        let c = textureSampleLevel(cubeTex, samp, dirs[i], 1, 0.0);
        out[i] = u32(round(c.r * 255.0));
    }
}
"#;

/// Layer `i` holds red = `(i + 1) * 10`. Faces follow the WebGPU/Vulkan layer order
/// +X, -X, +Y, -Y, +Z, -Z.
fn layer_value(layer: u32) -> u8 {
    u8::try_from((layer + 1) * 10).expect("layer value fits u8")
}

/// A six-layer square texture sampled through a `cube` view returns each face's layer.
pub fn cube_view_samples_each_face(backend: RealBackend, select: u32) {
    run(backend, select, 6, native::WGPUTextureViewDimension_Cube, CUBE_SHADER, 0);
}

/// A twelve-layer square texture sampled through a `cube-array` view at cube index 1
/// returns layers 6..12.
pub fn cube_array_view_samples_each_face(backend: RealBackend, select: u32) {
    run(
        backend,
        select,
        12,
        native::WGPUTextureViewDimension_CubeArray,
        CUBE_ARRAY_SHADER,
        6,
    );
}

fn run(
    backend: RealBackend,
    select: u32,
    layers: u32,
    view_dimension: native::WGPUTextureViewDimension,
    shader: &str,
    first_expected_layer: u32,
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

        let texture = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
                dimension: native::WGPUTextureDimension_2D,
                size: native::WGPUExtent3D {
                    width: SIZE,
                    height: SIZE,
                    depthOrArrayLayers: layers,
                },
                format: native::WGPUTextureFormat_RGBA8Unorm,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        assert!(!texture.is_null());
        let texels = (SIZE * SIZE) as usize;
        let mut data = Vec::with_capacity(texels * 4 * layers as usize);
        for layer in 0..layers {
            for _ in 0..texels {
                data.extend_from_slice(&[layer_value(layer), 0, 0, 255]);
            }
        }
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &native::WGPUTexelCopyTextureInfo {
                texture,
                mipLevel: 0,
                origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                aspect: native::WGPUTextureAspect_All,
            },
            data.as_ptr().cast(),
            data.len(),
            &native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: SIZE * 4,
                rowsPerImage: SIZE,
            },
            &native::WGPUExtent3D {
                width: SIZE,
                height: SIZE,
                depthOrArrayLayers: layers,
            },
        );
        let view = yawgpu::wgpuTextureCreateView(
            texture,
            &native::WGPUTextureViewDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                format: native::WGPUTextureFormat_Undefined,
                dimension: view_dimension,
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: layers,
                aspect: native::WGPUTextureAspect_All,
                usage: native::WGPUTextureUsage_None,
            },
        );
        assert!(!view.is_null());
        let sampler = yawgpu::wgpuDeviceCreateSampler(device, std::ptr::null());
        assert!(!sampler.is_null());

        let module = create_wgsl_module(device, shader);
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            device,
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null(),
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
        let bgl = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        let output = create_buffer(
            device,
            24,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            device,
            24,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let mut entries: [native::WGPUBindGroupEntry; 3] = std::mem::zeroed();
        entries[0].binding = 0;
        entries[0].textureView = view;
        entries[1].binding = 1;
        entries[1].sampler = sampler;
        entries[2].binding = 2;
        entries[2].buffer = output;
        entries[2].size = 24;
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: bgl,
                entryCount: entries.len(),
                entries: entries.as_ptr(),
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
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 24);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, 24);
        let actual: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect();
        let expected: Vec<u32> = (first_expected_layer..first_expected_layer + 6)
            .map(|layer| u32::from(layer_value(layer)))
            .collect();
        assert_eq!(actual, expected);
        assert_no_errors(&errors);

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
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
