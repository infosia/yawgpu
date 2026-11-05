//! Shared body of the Block 143 e2e, included by `e2e_{metal,vulkan}_destroy_memory.rs`
//! through `#[path]`.
//!
//! Proves on hardware that `destroy()` releases the backend allocation once the GPU is done
//! with it while handles stay alive (where the backend can report its allocated size), and
//! checks that destroying resources right after a submit does not disturb that submission.
//!
//! On Vulkan, run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; the expected
//! validation output is empty.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{YaWGPUInstanceBackendSelect, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT};
use yawgpu_test::wait;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const MIB: u64 = 1024 * 1024;
const BUFFER_SIZE: u64 = 64 * MIB;
const TEXTURE_EXTENT: u32 = 2048;
const TEXTURE_BYTES: u64 = (TEXTURE_EXTENT as u64) * (TEXTURE_EXTENT as u64) * 4;
const ITERATIONS: u64 = 16;

/// Each iteration creates a 64 MiB buffer and a 16 MiB texture, uses both in a submission,
/// destroys them without releasing the handles, and waits for the queue. With
/// `allocated_size`, the device's allocated size must stay within a few iterations' worth,
/// where keeping every allocation would grow it by `ITERATIONS` times that.
pub fn destroy_releases_memory(
    backend: RealBackend,
    select: u32,
    allocated_size: Option<fn() -> u64>,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        let per_iteration = BUFFER_SIZE + TEXTURE_BYTES;
        let start = allocated_size.map(|size| size());
        let mut buffers = Vec::new();
        let mut textures = Vec::new();
        for iteration in 0..ITERATIONS {
            let buffer = create_buffer(
                session.device,
                BUFFER_SIZE,
                native::WGPUBufferUsage_Storage
                    | native::WGPUBufferUsage_CopySrc
                    | native::WGPUBufferUsage_CopyDst,
            );
            let texture = create_texture(
                session.device,
                TEXTURE_EXTENT,
                TEXTURE_EXTENT,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
            );
            if iteration == 0 {
                if let (Some(size), Some(start)) = (allocated_size, start) {
                    // The probe must see this device's allocations, or the bound below
                    // proves nothing.
                    let grown = size().saturating_sub(start);
                    eprintln!("first iteration allocated {} MiB", grown / MIB);
                    assert!(
                        grown >= BUFFER_SIZE,
                        "allocated size grew by {grown} bytes after creating a {BUFFER_SIZE}-byte buffer"
                    );
                }
            }

            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
            yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, BUFFER_SIZE);
            yawgpu::wgpuCommandEncoderCopyBufferToTexture(
                encoder,
                &native::WGPUTexelCopyBufferInfo {
                    layout: native::WGPUTexelCopyBufferLayout {
                        offset: 0,
                        bytesPerRow: TEXTURE_EXTENT * 4,
                        rowsPerImage: TEXTURE_EXTENT,
                    },
                    buffer,
                },
                &copy_texture(texture),
                &native::WGPUExtent3D {
                    width: TEXTURE_EXTENT,
                    height: TEXTURE_EXTENT,
                    depthOrArrayLayers: 1,
                },
            );
            session.submit(encoder);
            yawgpu::wgpuBufferDestroy(buffer);
            yawgpu::wgpuTextureDestroy(texture);
            session.wait_work_done();
            buffers.push(buffer);
            textures.push(texture);
        }
        yawgpu::wgpuInstanceProcessEvents(session.instance);

        if let (Some(size), Some(start)) = (allocated_size, start) {
            let grown = size().saturating_sub(start);
            eprintln!(
                "after {ITERATIONS} destroyed iterations: {} MiB still allocated ({} MiB each)",
                grown / MIB,
                per_iteration / MIB
            );
            assert!(
                grown < 3 * per_iteration,
                "destroyed resources kept {grown} bytes allocated over {ITERATIONS} iterations of {per_iteration} bytes"
            );
        }

        for buffer in buffers {
            yawgpu::wgpuBufferRelease(buffer);
        }
        for texture in textures {
            yawgpu::wgpuTextureRelease(texture);
        }
        session.finish();
    }
}

/// One submission writes a storage buffer from a compute pass, copies it to a readback
/// buffer and to a texture, and copies the texture to a second readback buffer. The
/// storage buffer and the texture are destroyed and released right after the submit;
/// both readbacks must still hold the computed values. This is a safety regression test:
/// it cannot prove the submission was still running at the destroy, but on Vulkan the
/// validation layer reports any object destroyed while in use.
pub fn destroy_while_in_flight(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    const WORDS: u32 = 1024;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = WORDS / WIDTH;
    let size = u64::from(WORDS) * 4;
    unsafe {
        let session = Session::new(select);
        let device = session.device;
        let storage = create_buffer(
            device,
            size,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let texture = create_texture(
            device,
            WIDTH,
            HEIGHT,
            native::WGPUTextureFormat_R32Uint,
            native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
        );
        let readback_buffer = create_buffer(
            device,
            size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let readback_texture = create_buffer(
            device,
            size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let module = create_wgsl_module(
            device,
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
             @compute @workgroup_size(64)\n\
             fn main(@builtin(global_invocation_id) gid: vec3u) {\n\
                 out[gid.x] = gid.x * 3u + 1u;\n\
             }\n",
        );
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let bind_group = create_single_bind_group(device, layout, storage, size);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, WORDS / 64, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback_buffer, 0, size);
        let layout_256 = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: WIDTH * 4,
            rowsPerImage: HEIGHT,
        };
        let extent = native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(
            encoder,
            &native::WGPUTexelCopyBufferInfo {
                layout: layout_256,
                buffer: storage,
            },
            &copy_texture(texture),
            &extent,
        );
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &copy_texture(texture),
            &native::WGPUTexelCopyBufferInfo {
                layout: layout_256,
                buffer: readback_texture,
            },
            &extent,
        );
        session.submit(encoder);

        // The submission may still be running: destroy and release everything it writes
        // through except the readback buffers.
        yawgpu::wgpuBufferDestroy(storage);
        yawgpu::wgpuTextureDestroy(texture);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuTextureRelease(texture);

        let expected: Vec<u32> = (0..WORDS).map(|i| i * 3 + 1).collect();
        assert_eq!(
            read_words(session.instance, readback_buffer, size as usize),
            expected,
            "buffer readback"
        );
        assert_eq!(
            read_words(session.instance, readback_texture, size as usize),
            expected,
            "texture readback"
        );

        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback_buffer);
        yawgpu::wgpuBufferRelease(readback_texture);
        session.finish();
    }
}

struct Session {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl Session {
    unsafe fn new(select: u32) -> Self {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        Self {
            instance,
            adapter,
            device,
            queue,
            errors,
        }
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }

    unsafe fn wait_work_done(&self) {
        let mut status = native::WGPUQueueWorkDoneStatus_Error;
        let callback_info = native::WGPUQueueWorkDoneCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(work_done_callback),
            userdata1: (&mut status as *mut native::WGPUQueueWorkDoneStatus).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = yawgpu::wgpuQueueOnSubmittedWorkDone(self.queue, callback_info);
        wait(self.instance, future);
        assert_eq!(status, native::WGPUQueueWorkDoneStatus_Success);
    }

    unsafe fn finish(self) {
        assert!(
            self.errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            self.errors.lock().expect("error lock")
        );
        yawgpu::wgpuQueueRelease(self.queue);
        yawgpu::wgpuDeviceRelease(self.device);
        yawgpu::wgpuAdapterRelease(self.adapter);
        yawgpu::wgpuInstanceRelease(self.instance);
    }
}

fn copy_texture(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

unsafe fn create_buffer(
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

unsafe fn create_texture(
    device: native::WGPUDevice,
    width: u32,
    height: u32,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width,
            height,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
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
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
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

unsafe fn read_words(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u32> {
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
    let words = std::slice::from_raw_parts(ptr.cast::<u32>(), len / 4).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    words
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

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
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

unsafe extern "C" fn work_done_callback(
    status: native::WGPUQueueWorkDoneStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUQueueWorkDoneStatus) = status;
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
