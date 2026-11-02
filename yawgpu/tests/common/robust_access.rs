//! Shared bodies of the Block 128 e2e (`specs/blocks/128-vulkan-robustness2.md`),
//! included by `e2e_{metal,vulkan}_robust_access.rs` through `#[path]`.
//!
//! Each scenario makes out-of-bounds accesses that WGSL defines loosely, then checks
//! that the results are among the values WGSL allows. On Vulkan the bounds come from
//! either Tint's clamps or the driver's robustness2 guarantees, depending on the
//! device (Block 128 R2), so the same checks must hold on both paths.

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const STORAGE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32, 8>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 4u; i++) {
        out[i] = data[params[i]];
    }
    out[4] = arrayLength(&data);
    for (var i = 0u; i < 4u; i++) {
        data[params[i]] = 0xDEAD0000u + i;
    }
}
"#;

/// Words in the whole `data` buffer.
const DATA_WORDS: u32 = 128;
/// The binding covers words 64..68 (byte offset 256, 16 bytes).
const BINDING_FIRST_WORD: u32 = 64;
const BINDING_WORDS: u32 = 4;
/// Out-of-bounds indices, read from a uniform so the compiler cannot fold them.
const OOB_INDICES: [u32; 4] = [4, 100, 1_000_000, 0x7FFF_FFFF];

fn sentinel(word: u32) -> u32 {
    0x1000 + word
}

/// Out-of-bounds reads and writes of a runtime-sized storage array stay inside the
/// binding: reads return a value from the binding or zero, and writes either drop
/// or land inside the binding. Nothing outside the bound range changes.
pub fn storage_runtime_array_out_of_bounds(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let data_bytes: Vec<u8> = (0..DATA_WORDS)
            .flat_map(|word| sentinel(word).to_le_bytes())
            .collect();
        let data_size = u64::from(DATA_WORDS) * 4;
        let data = create_buffer(
            device,
            data_size,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopyDst
                | native::WGPUBufferUsage_CopySrc,
        );
        yawgpu::wgpuQueueWriteBuffer(queue, data, 0, data_bytes.as_ptr().cast(), data_bytes.len());
        let out = create_buffer(
            device,
            32,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let params = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let params_bytes: Vec<u8> = OOB_INDICES.iter().flat_map(|i| i.to_le_bytes()).collect();
        yawgpu::wgpuQueueWriteBuffer(queue, params, 0, params_bytes.as_ptr().cast(), 16);

        let entries = [
            buffer_entry(
                0,
                data,
                u64::from(BINDING_FIRST_WORD) * 4,
                u64::from(BINDING_WORDS) * 4,
            ),
            buffer_entry(1, out, 0, 32),
            buffer_entry(2, params, 0, 16),
        ];
        dispatch_once(device, queue, STORAGE_SHADER, &entries);

        let data_readback = copy_to_readback(device, queue, data, data_size);
        let out_readback = copy_to_readback(device, queue, out, 32);
        let words_after = words(&read_buffer(instance, data_readback, data_size as usize));
        let out_words = words(&read_buffer(instance, out_readback, 32));

        let binding = BINDING_FIRST_WORD..BINDING_FIRST_WORD + BINDING_WORDS;
        let allowed_reads: Vec<u32> = binding.clone().map(sentinel).chain([0]).collect();
        eprintln!("OOB storage reads: {:x?}", &out_words[..4]);
        eprintln!(
            "binding words after OOB writes: {:x?}",
            &words_after
                [BINDING_FIRST_WORD as usize..(BINDING_FIRST_WORD + BINDING_WORDS) as usize]
        );
        for (i, value) in out_words[..4].iter().enumerate() {
            assert!(
                allowed_reads.contains(value),
                "OOB read {i} (index {}) returned {value:#x}; allowed {allowed_reads:x?}",
                OOB_INDICES[i]
            );
        }
        assert_eq!(out_words[4], BINDING_WORDS, "arrayLength");
        let written: Vec<u32> = (0..4).map(|i| 0xDEAD_0000 + i).collect();
        for (word, value) in words_after.iter().enumerate() {
            let word = word as u32;
            if binding.contains(&word) {
                assert!(
                    *value == sentinel(word) || written.contains(value),
                    "word {word} inside the binding holds {value:#x}"
                );
            } else {
                assert_eq!(
                    *value,
                    sentinel(word),
                    "word {word} outside the binding changed"
                );
            }
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBufferRelease(out_readback);
        yawgpu::wgpuBufferRelease(data_readback);
        yawgpu::wgpuBufferRelease(params);
        yawgpu::wgpuBufferRelease(out);
        yawgpu::wgpuBufferRelease(data);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

const TEXTURE_SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_2d_array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>, 6>;
@group(0) @binding(2) var<uniform> params: array<vec4<i32>, 6>;

@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 6u; i++) {
        let p = params[i];
        out[i] = textureLoad(tex, p.xy, p.z, p.w);
    }
}
"#;

const TEX_SIZE: u32 = 4;
const TEX_LAYERS: u32 = 2;
/// `(x, y, layer, level)`. Case 0 is in bounds; every other case is out of bounds
/// in one coordinate.
const TEXTURE_CASES: [[i32; 4]; 6] = [
    [2, 1, 1, 0],
    [4, 0, 0, 0],
    [0, 100, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 5, 0],
    [0, 0, 0, 3],
];

fn texel(x: u32, y: u32, layer: u32) -> [u32; 4] {
    [x + 1, y + 1, layer + 1, 7]
}

/// Out-of-bounds `textureLoad` returns an in-bounds texel, `(0, 0, 0, 0)` or
/// `(0, 0, 0, 1)`, and the in-bounds control load returns its exact texel.
pub fn texture_load_out_of_bounds(backend: RealBackend, select: u32) {
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
                    width: TEX_SIZE,
                    height: TEX_SIZE,
                    depthOrArrayLayers: TEX_LAYERS,
                },
                format: native::WGPUTextureFormat_RGBA8Uint,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        assert!(!texture.is_null());
        let mut texels = Vec::new();
        let mut bytes = Vec::new();
        for layer in 0..TEX_LAYERS {
            for y in 0..TEX_SIZE {
                for x in 0..TEX_SIZE {
                    let value = texel(x, y, layer);
                    texels.push(value);
                    bytes.extend(value.iter().map(|channel| *channel as u8));
                }
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
            bytes.as_ptr().cast(),
            bytes.len(),
            &native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: TEX_SIZE * 4,
                rowsPerImage: TEX_SIZE,
            },
            &native::WGPUExtent3D {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depthOrArrayLayers: TEX_LAYERS,
            },
        );
        let view = yawgpu::wgpuTextureCreateView(
            texture,
            &native::WGPUTextureViewDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                format: native::WGPUTextureFormat_Undefined,
                dimension: native::WGPUTextureViewDimension_2DArray,
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: TEX_LAYERS,
                aspect: native::WGPUTextureAspect_All,
                usage: native::WGPUTextureUsage_None,
            },
        );
        assert!(!view.is_null());

        let out_size = 6 * 16;
        let out = create_buffer(
            device,
            out_size,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let params = create_buffer(
            device,
            out_size,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let params_bytes: Vec<u8> = TEXTURE_CASES
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            params,
            0,
            params_bytes.as_ptr().cast(),
            params_bytes.len(),
        );

        let entries = [
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 0,
                buffer: std::ptr::null_mut(),
                offset: 0,
                size: 0,
                sampler: std::ptr::null_mut(),
                textureView: view,
            },
            buffer_entry(1, out, 0, out_size),
            buffer_entry(2, params, 0, out_size),
        ];
        dispatch_once(device, queue, TEXTURE_SHADER, &entries);

        let readback = copy_to_readback(device, queue, out, out_size);
        let loaded: Vec<[u32; 4]> = words(&read_buffer(instance, readback, out_size as usize))
            .as_chunks::<4>()
            .0
            .to_vec();
        eprintln!("textureLoad results: {loaded:?}");

        let [x, y, layer, _] = TEXTURE_CASES[0];
        assert_eq!(
            loaded[0],
            texel(x as u32, y as u32, layer as u32),
            "in-bounds control load"
        );
        for (case, value) in loaded.iter().enumerate().skip(1) {
            assert!(
                texels.contains(value) || *value == [0, 0, 0, 0] || *value == [0, 0, 0, 1],
                "OOB textureLoad {:?} returned {value:?}",
                TEXTURE_CASES[case]
            );
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(params);
        yawgpu::wgpuBufferRelease(out);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

const STORAGE_TEXTURE_SHADER: &str = r#"
@group(0) @binding(0) var st: texture_storage_2d<r32uint, read_write>;
@group(0) @binding(1) var<storage, read_write> out: array<u32, 4>;
@group(0) @binding(2) var<uniform> params: array<vec4<i32>, 4>;

@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 4u; i++) {
        out[i] = textureLoad(st, params[i].xy).x;
    }
    for (var i = 0u; i < 4u; i++) {
        textureStore(st, params[i].xy, vec4<u32>(0xBEEF0000u + i));
    }
}
"#;

/// Out-of-bounds coordinates for the storage-texture scenario.
const STORAGE_TEXTURE_CASES: [[i32; 2]; 4] = [[4, 0], [0, 4], [-1, -1], [100, 100]];

fn storage_texel(x: u32, y: u32) -> u32 {
    0x100 + y * TEX_SIZE + x
}

/// Out-of-bounds read-write storage-texture loads return an in-bounds texel or zero,
/// and out-of-bounds stores either drop or land on an in-bounds texel.
pub fn storage_texture_out_of_bounds(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let extent = native::WGPUExtent3D {
            width: TEX_SIZE,
            height: TEX_SIZE,
            depthOrArrayLayers: 1,
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_StorageBinding
                    | native::WGPUTextureUsage_CopyDst
                    | native::WGPUTextureUsage_CopySrc,
                dimension: native::WGPUTextureDimension_2D,
                size: extent,
                format: native::WGPUTextureFormat_R32Uint,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        assert!(!texture.is_null());
        let originals: Vec<u32> = (0..TEX_SIZE)
            .flat_map(|y| (0..TEX_SIZE).map(move |x| storage_texel(x, y)))
            .collect();
        let bytes: Vec<u8> = originals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let texture_copy = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &texture_copy,
            bytes.as_ptr().cast(),
            bytes.len(),
            &native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: TEX_SIZE * 4,
                rowsPerImage: TEX_SIZE,
            },
            &extent,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());

        let out = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let params = create_buffer(
            device,
            64,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let params_bytes: Vec<u8> = STORAGE_TEXTURE_CASES
            .iter()
            .flat_map(|[x, y]| [*x, *y, 0, 0])
            .flat_map(|value| value.to_le_bytes())
            .collect();
        yawgpu::wgpuQueueWriteBuffer(queue, params, 0, params_bytes.as_ptr().cast(), 64);

        let entries = [
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 0,
                buffer: std::ptr::null_mut(),
                offset: 0,
                size: 0,
                sampler: std::ptr::null_mut(),
                textureView: view,
            },
            buffer_entry(1, out, 0, 16),
            buffer_entry(2, params, 0, 64),
        ];
        dispatch_once(device, queue, STORAGE_TEXTURE_SHADER, &entries);

        // Texture rows are copied at the 256-byte row pitch copies require.
        let row_pitch = 256u32;
        let texture_readback = create_buffer(
            device,
            u64::from(row_pitch * TEX_SIZE),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &texture_copy,
            &native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: row_pitch,
                    rowsPerImage: TEX_SIZE,
                },
                buffer: texture_readback,
            },
            &extent,
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let out_readback = copy_to_readback(device, queue, out, 16);

        let texture_words = words(&read_buffer(
            instance,
            texture_readback,
            (row_pitch * TEX_SIZE) as usize,
        ));
        let loaded = words(&read_buffer(instance, out_readback, 16));
        let after: Vec<u32> = (0..TEX_SIZE)
            .flat_map(|y| {
                let row = (y * row_pitch / 4) as usize;
                texture_words[row..row + TEX_SIZE as usize].to_vec()
            })
            .collect();
        eprintln!("OOB storage-texture loads: {loaded:x?}");
        eprintln!("texels after OOB stores: {after:x?}");

        for (case, value) in loaded.iter().enumerate() {
            assert!(
                originals.contains(value) || *value == 0,
                "OOB storage textureLoad {:?} returned {value:#x}",
                STORAGE_TEXTURE_CASES[case]
            );
        }
        let stored: Vec<u32> = (0..4).map(|i| 0xBEEF_0000 + i).collect();
        for (index, value) in after.iter().enumerate() {
            assert!(
                *value == originals[index] || stored.contains(value),
                "texel {index} holds {value:#x} after OOB stores"
            );
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBufferRelease(out_readback);
        yawgpu::wgpuBufferRelease(texture_readback);
        yawgpu::wgpuBufferRelease(params);
        yawgpu::wgpuBufferRelease(out);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn buffer_entry(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Builds an auto-layout compute pipeline from `source`, binds `entries` at group 0,
/// and submits one 1x1x1 dispatch.
unsafe fn dispatch_once(
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    source: &str,
    entries: &[native::WGPUBindGroupEntry],
) {
    let module = create_wgsl_module(device, source);
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
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
        device,
        &native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
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
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
}

/// Copies `size` bytes of `source` into a new map-readable buffer and submits it.
unsafe fn copy_to_readback(
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    source: native::WGPUBuffer,
    size: u64,
) -> native::WGPUBuffer {
    let readback = create_buffer(
        device,
        size,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
    );
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, readback, 0, size);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    readback
}
