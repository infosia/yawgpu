//! Shared body of the Block 141 feature e2e, included by `e2e_{metal,vulkan}_dawn_features.rs`
//! through `#[path]`.
//!
//! Proves on hardware (not just Noop validation):
//! - `atomic-vec2u-min-max`: `atomicStoreMax` and `atomicStoreMin` on `atomic<vec2u>` from
//!   many invocations leave the 64-bit maximum and minimum, with `y` as the high word, as
//!   the CTS reads them;
//! - `texture-compression-unaligned`: a `bc1-rgba-unorm` texture whose level 0 is 5×7 (not
//!   a multiple of the 4×4 block) is created, every mip level is written with
//!   `writeTexture` at its physical size, copied texture-to-texture, copied to a buffer, and
//!   read back unchanged.
//!
//! On Vulkan, run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; the expected
//! validation output is empty.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{YaWGPUInstanceBackendSelect, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT};
use yawgpu_test::wait;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const WORKGROUP_SIZE: u32 = 64;
const WORKGROUPS: u32 = 16;

/// `require_unaligned` / `require_atomic`: the adapter must advertise the feature;
/// otherwise a missing feature is only reported.
pub fn adapter_advertises_features(
    backend: RealBackend,
    select: u32,
    require_unaligned: bool,
    require_atomic: bool,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        let unaligned = yawgpu::wgpuAdapterHasFeature(
            adapter,
            native::WGPUFeatureName_TextureCompressionUnaligned,
        ) != 0;
        let atomic =
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_AtomicVec2uMinMax) != 0;
        eprintln!("texture-compression-unaligned: {unaligned}, atomic-vec2u-min-max: {atomic}");
        assert!(
            unaligned || !require_unaligned,
            "the adapter must advertise texture-compression-unaligned"
        );
        assert!(
            atomic || !require_atomic,
            "the adapter must advertise atomic-vec2u-min-max"
        );

        let mut features = Vec::new();
        if unaligned {
            features.push(native::WGPUFeatureName_TextureCompressionUnaligned);
        }
        if atomic {
            features.push(native::WGPUFeatureName_AtomicVec2uMinMax);
        }
        let device = request_device_with(instance, adapter, &features);
        for feature in features {
            assert!(yawgpu::wgpuDeviceHasFeature(device, feature) != 0);
        }
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// 64-bit `atomicStoreMax` / `atomicStoreMin` over `WORKGROUP_SIZE * WORKGROUPS`
/// invocations. The values are chosen so that comparing either word alone gives a
/// different answer from the 64-bit comparison.
pub fn atomic_vec2u_min_max_readback(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        if yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_AtomicVec2uMinMax) == 0 {
            eprintln!("skip: adapter does not expose atomic-vec2u-min-max");
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
            return;
        }
        let device = request_device_with(
            instance,
            adapter,
            &[native::WGPUFeatureName_AtomicVec2uMinMax],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // value(i).y = i * 2654435761 >> 28 (few distinct high words, many ties);
        // value(i).x = i * 40503 ^ 0x9e3779b9 (low word, breaks the ties).
        let shader = format!(
            "enable atomic_vec2u_min_max;\n\
             @group(0) @binding(0) var<storage, read_write> out: array<atomic<vec2u>, 2>;\n\
             @compute @workgroup_size({WORKGROUP_SIZE})\n\
             fn main(@builtin(global_invocation_id) gid: vec3u) {{\n\
                 let i = gid.x;\n\
                 let v = vec2u((i * 40503u) ^ 0x9e3779b9u, (i * 2654435761u) >> 28u);\n\
                 atomicStoreMax(&out[0], v);\n\
                 atomicStoreMin(&out[1], v);\n\
             }}\n"
        );
        let size = 16u64;
        let output = create_buffer_sized(
            device,
            size,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst,
        );
        let readback = create_buffer_sized(
            device,
            size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        // out[0] starts at the 64-bit minimum, out[1] at the maximum.
        let init: [u32; 4] = [0, 0, u32::MAX, u32::MAX];
        yawgpu::wgpuQueueWriteBuffer(queue, output, 0, init.as_ptr().cast(), 16);

        let module = create_wgsl_module(device, &shader);
        let pipeline = create_compute_pipeline(device, module);
        assert!(!pipeline.is_null());
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let bind_group = create_single_bind_group(device, layout, output, size);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, WORKGROUPS, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, size);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let words = read_words(instance, readback, size as usize);
        let values: Vec<u64> = (0..WORKGROUP_SIZE * WORKGROUPS)
            .map(|i| {
                let x = i.wrapping_mul(40503) ^ 0x9e37_79b9;
                let y = i.wrapping_mul(2_654_435_761) >> 28;
                (u64::from(y) << 32) | u64::from(x)
            })
            .collect();
        let max = *values.iter().max().expect("values");
        let min = *values.iter().min().expect("values");
        let got_max = (u64::from(words[1]) << 32) | u64::from(words[0]);
        let got_min = (u64::from(words[3]) << 32) | u64::from(words[2]);
        assert_eq!(
            got_max, max,
            "atomicStoreMax: got {got_max:#x}, expected {max:#x}"
        );
        assert_eq!(
            got_min, min,
            "atomicStoreMin: got {got_min:#x}, expected {min:#x}"
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// A 5×7 `bc1-rgba-unorm` texture with three mip levels (logical 5×7, 2×3, 1×1; physical
/// 8×8, 4×4, 4×4). Each level is written at its physical size, copied to a second texture,
/// then copied to a buffer and compared.
pub fn unaligned_bc_texture_round_trip(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        let needed = [
            native::WGPUFeatureName_TextureCompressionUnaligned,
            native::WGPUFeatureName_TextureCompressionBC,
        ];
        if needed
            .iter()
            .any(|&feature| yawgpu::wgpuAdapterHasFeature(adapter, feature) == 0)
        {
            eprintln!(
                "skip: adapter lacks texture-compression-unaligned or texture-compression-bc"
            );
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
            return;
        }
        let device = request_device_with(instance, adapter, &needed);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        const LEVELS: u32 = 3;
        const BLOCK_BYTES: u32 = 8;
        let size = native::WGPUExtent3D {
            width: 5,
            height: 7,
            depthOrArrayLayers: 1,
        };
        let source = create_bc1_texture(
            device,
            size,
            LEVELS,
            native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
        );
        let copy = create_bc1_texture(
            device,
            size,
            LEVELS,
            native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "creating an unaligned BC texture must succeed with the feature: {:?}",
            errors.lock().expect("error lock")
        );

        let mut expected = Vec::new();
        let mut readbacks = Vec::new();
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        for level in 0..LEVELS {
            let (blocks_w, blocks_h) = physical_blocks(size, level);
            let physical = native::WGPUExtent3D {
                width: blocks_w * 4,
                height: blocks_h * 4,
                depthOrArrayLayers: 1,
            };
            let row_bytes = blocks_w * BLOCK_BYTES;
            let data: Vec<u8> = (0..row_bytes * blocks_h)
                .map(|byte| (byte * 7 + level * 31 + 1) as u8)
                .collect();
            let layout = native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: row_bytes,
                rowsPerImage: blocks_h,
            };
            yawgpu::wgpuQueueWriteTexture(
                queue,
                &copy_texture(source, level),
                data.as_ptr().cast(),
                data.len(),
                &layout,
                &physical,
            );
            yawgpu::wgpuCommandEncoderCopyTextureToTexture(
                encoder,
                &copy_texture(source, level),
                &copy_texture(copy, level),
                &physical,
            );
            let readback_size = u64::from(256 * blocks_h);
            let readback = create_buffer_sized(
                device,
                readback_size,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            );
            let dst = native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: 256,
                    rowsPerImage: blocks_h,
                },
                buffer: readback,
            };
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
                encoder,
                &copy_texture(copy, level),
                &dst,
                &physical,
            );
            expected.push((data, row_bytes, blocks_h));
            readbacks.push((readback, readback_size));
        }
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        for (level, ((data, row_bytes, rows), (readback, readback_size))) in
            expected.iter().zip(&readbacks).enumerate()
        {
            let bytes = read_bytes(instance, *readback, *readback_size as usize);
            for row in 0..*rows as usize {
                let got = &bytes[row * 256..row * 256 + *row_bytes as usize];
                let want = &data[row * *row_bytes as usize..(row + 1) * *row_bytes as usize];
                assert_eq!(got, want, "mip level {level}, block row {row}");
            }
            yawgpu::wgpuBufferRelease(*readback);
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuTextureRelease(copy);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// A 5×7 `bc1-rgba-unorm` texture with two mip levels. Only block (0, 0) of level 0 is
/// written, so the copy to a buffer lazily zero-initializes the rest of level 0 and all
/// of level 1; every unwritten block must read back as zero.
pub fn unaligned_bc_texture_lazy_zero_init(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        let needed = [
            native::WGPUFeatureName_TextureCompressionUnaligned,
            native::WGPUFeatureName_TextureCompressionBC,
        ];
        if needed
            .iter()
            .any(|&feature| yawgpu::wgpuAdapterHasFeature(adapter, feature) == 0)
        {
            eprintln!(
                "skip: adapter lacks texture-compression-unaligned or texture-compression-bc"
            );
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
            return;
        }
        let device = request_device_with(instance, adapter, &needed);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        const LEVELS: u32 = 2;
        const BLOCK_BYTES: usize = 8;
        let size = native::WGPUExtent3D {
            width: 5,
            height: 7,
            depthOrArrayLayers: 1,
        };
        let texture = create_bc1_texture(
            device,
            size,
            LEVELS,
            native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
        );
        let block: [u8; BLOCK_BYTES] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let one_block = native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &copy_texture(texture, 0),
            block.as_ptr().cast(),
            BLOCK_BYTES,
            &native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BLOCK_BYTES as u32,
                rowsPerImage: 1,
            },
            &one_block,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let mut readbacks = Vec::new();
        for level in 0..LEVELS {
            let (blocks_w, blocks_h) = physical_blocks(size, level);
            let readback_size = u64::from(256 * blocks_h);
            let readback = create_buffer_sized(
                device,
                readback_size,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            );
            let dst = native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: 256,
                    rowsPerImage: blocks_h,
                },
                buffer: readback,
            };
            let physical = native::WGPUExtent3D {
                width: blocks_w * 4,
                height: blocks_h * 4,
                depthOrArrayLayers: 1,
            };
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
                encoder,
                &copy_texture(texture, level),
                &dst,
                &physical,
            );
            readbacks.push((readback, readback_size, blocks_w, blocks_h));
        }
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        for (level, (readback, readback_size, blocks_w, blocks_h)) in readbacks.iter().enumerate() {
            let bytes = read_bytes(instance, *readback, *readback_size as usize);
            for by in 0..*blocks_h as usize {
                for bx in 0..*blocks_w as usize {
                    let start = by * 256 + bx * BLOCK_BYTES;
                    let got = &bytes[start..start + BLOCK_BYTES];
                    if level == 0 && bx == 0 && by == 0 {
                        assert_eq!(
                            got, block,
                            "level 0 block (0, 0) must keep the written data"
                        );
                    } else {
                        assert!(
                            got.iter().all(|&b| b == 0),
                            "level {level} block ({bx}, {by}) must read back as zero: {got:?}"
                        );
                    }
                }
            }
            yawgpu::wgpuBufferRelease(*readback);
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn physical_blocks(size: native::WGPUExtent3D, level: u32) -> (u32, u32) {
    let width = (size.width >> level).max(1);
    let height = (size.height >> level).max(1);
    (width.div_ceil(4), height.div_ceil(4))
}

// ---- helpers ----

fn copy_texture(texture: native::WGPUTexture, level: u32) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: level,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

unsafe fn create_bc1_texture(
    device: native::WGPUDevice,
    size: native::WGPUExtent3D,
    levels: u32,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size,
        format: native::WGPUTextureFormat_BC1RGBAUnorm,
        mipLevelCount: levels,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_buffer_sized(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
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
    };
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
}

unsafe fn create_single_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
    size: u64,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset: 0,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    bind_group
}

unsafe fn read_bytes(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn read_words(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u32> {
    read_bytes(instance, buffer, len)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_ne_bytes(*c))
        .collect()
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn create_backend_instance(select: u32) -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: select,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    instance
}

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!adapter.is_null());
    adapter
}

unsafe fn request_device_with(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured_errors = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured_errors.lock().expect("error lock").push(error)),
    );
    errors
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    *(userdata1 as *mut native::WGPUAdapter) = adapter;
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
