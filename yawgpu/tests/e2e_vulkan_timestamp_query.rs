//! Real-Vulkan e2e for Block 102 (`specs/blocks/102-timestamp-query.md`):
//! `timestamp-query` executes end to end through the C ABI — encoder-level
//! `writeTimestamp`, compute/render pass `timestampWrites`, and
//! `resolveQuerySet` producing **nanoseconds** (the conversion pass ran),
//! with never-written slots resolving to zero.

#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const BACKEND: RealBackend = RealBackend::Vulkan;
const ONE_SECOND_NS: u64 = 1_000_000_000;

/// A deliberately busy kernel so that the two stamps around it are
/// measurably apart: every invocation loops over its own storage slot.
const BUSY_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var acc = id.x;
    for (var i = 0u; i < 256u; i = i + 1u) {
        acc = acc * 1664525u + 1013904223u;
    }
    data[id.x % 1024u] = acc;
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_write_timestamp_around_a_dispatch_resolves_increasing_nanoseconds() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        if session.device.is_null() {
            eprintln!("skipping: timestamp-query is not advertised on this adapter");
            return;
        }
        let set = session.create_timestamp_set(2);
        let readback = session.create_resolve_buffer(256);
        let busy = session.busy_dispatch();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, set, 0);
        busy.record(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, set, 1);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set, 0, 2, readback, 0);
        session.submit(encoder);

        let stamps = session.read_u64s(readback, 0, 2);
        assert!(stamps[0] > 0, "begin stamp must be non-zero: {stamps:?}");
        assert!(stamps[1] > stamps[0], "end must follow begin: {stamps:?}");
        assert!(
            stamps[1] - stamps[0] < ONE_SECOND_NS,
            "elapsed must be a plausible nanosecond count: {stamps:?}"
        );
        session.assert_no_errors();

        busy.release();
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQuerySetRelease(set);
        session.release();
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_pass_timestamp_writes_with_only_an_end_index_leave_the_other_slot_zero() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        if session.device.is_null() {
            eprintln!("skipping: timestamp-query is not advertised on this adapter");
            return;
        }
        let set = session.create_timestamp_set(4);
        let readback = session.create_resolve_buffer(256);
        let busy = session.busy_dispatch();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        // Compute pass: only endOfPassWriteIndex = 1 (slot 0 stays unwritten).
        let compute_writes = native::WGPUPassTimestampWrites {
            nextInChain: std::ptr::null_mut(),
            querySet: set,
            beginningOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
            endOfPassWriteIndex: 1,
        };
        busy.record(encoder, &compute_writes);
        // Render pass (clear only): only endOfPassWriteIndex = 3 (slot 2 unwritten).
        let render_writes = native::WGPUPassTimestampWrites {
            nextInChain: std::ptr::null_mut(),
            querySet: set,
            beginningOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
            endOfPassWriteIndex: 3,
        };
        let target = session.create_color_target();
        record_clear_render_pass(encoder, target.view, &render_writes);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set, 0, 4, readback, 0);
        session.submit(encoder);

        let stamps = session.read_u64s(readback, 0, 4);
        assert_eq!(stamps[0], 0, "unwritten compute begin slot: {stamps:?}");
        assert!(stamps[1] > 0, "compute end stamp: {stamps:?}");
        assert_eq!(stamps[2], 0, "unwritten render begin slot: {stamps:?}");
        assert!(stamps[3] > 0, "render end stamp: {stamps:?}");
        assert!(
            stamps[3] >= stamps[1],
            "render pass ends after the compute pass: {stamps:?}"
        );
        assert!(
            stamps[3] - stamps[1] < ONE_SECOND_NS,
            "plausible nanoseconds: {stamps:?}"
        );
        session.assert_no_errors();

        target.release();
        busy.release();
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQuerySetRelease(set);
        session.release();
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_resolving_a_never_written_range_yields_zeros() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        if session.device.is_null() {
            eprintln!("skipping: timestamp-query is not advertised on this adapter");
            return;
        }
        let set = session.create_timestamp_set(8);
        let readback = session.create_resolve_buffer(256);
        // Pre-fill the destination so a "skipped" resolve would be visible.
        let queue = session.queue;
        let junk = [0xABu8; 64];
        yawgpu::wgpuQueueWriteBuffer(queue, readback, 0, junk.as_ptr().cast(), junk.len());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set, 0, 8, readback, 0);
        session.submit(encoder);

        let stamps = session.read_u64s(readback, 0, 8);
        assert_eq!(stamps, vec![0u64; 8], "never-written slots resolve to zero");
        session.assert_no_errors();

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQuerySetRelease(set);
        session.release();
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_two_sets_resolved_into_one_buffer_at_different_offsets_do_not_clobber() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        if session.device.is_null() {
            eprintln!("skipping: timestamp-query is not advertised on this adapter");
            return;
        }
        let set_a = session.create_timestamp_set(2);
        let set_b = session.create_timestamp_set(2);
        let readback = session.create_resolve_buffer(512);
        let busy = session.busy_dispatch();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, set_a, 0);
        busy.record(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, set_b, 1);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set_a, 0, 2, readback, 0);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set_b, 0, 2, readback, 256);
        session.submit(encoder);

        let a = session.read_u64s(readback, 0, 2);
        let b = session.read_u64s(readback, 256, 2);
        assert!(
            a[0] > 0 && a[1] == 0,
            "set A: slot 0 written, slot 1 not: {a:?}"
        );
        assert!(
            b[0] == 0 && b[1] > 0,
            "set B: slot 1 written, slot 0 not: {b:?}"
        );
        assert!(
            b[1] > a[0] && b[1] - a[0] < ONE_SECOND_NS,
            "ordered, plausible: {a:?} {b:?}"
        );
        session.assert_no_errors();

        busy.release();
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQuerySetRelease(set_a);
        yawgpu::wgpuQuerySetRelease(set_b);
        session.release();
    }
}

// --- session / helpers -------------------------------------------------------

struct Session {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    /// Null when the adapter does not advertise `timestamp-query`.
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

struct BusyDispatch {
    pipeline: native::WGPUComputePipeline,
    bind_group: native::WGPUBindGroup,
    layout: native::WGPUPipelineLayout,
    bgl: native::WGPUBindGroupLayout,
    module: native::WGPUShaderModule,
    buffer: native::WGPUBuffer,
}

struct ColorTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

impl Session {
    unsafe fn new() -> Self {
        let instance = create_instance();
        let adapter = request_adapter(instance);
        let advertised =
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_TimestampQuery) != 0;
        if !advertised {
            return Self {
                instance,
                adapter,
                device: std::ptr::null(),
                queue: std::ptr::null(),
                errors: Arc::new(Mutex::new(Vec::new())),
            };
        }
        let device = request_device(instance, adapter, &[native::WGPUFeatureName_TimestampQuery]);
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

    unsafe fn create_timestamp_set(&self, count: u32) -> native::WGPUQuerySet {
        let descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            type_: native::WGPUQueryType_Timestamp,
            count,
        };
        let set = yawgpu::wgpuDeviceCreateQuerySet(self.device, &descriptor);
        assert!(!set.is_null());
        set
    }

    /// A resolve destination. `MAP_READ` may only pair with `COPY_DST`, so
    /// results are copied into a separate readback buffer by `read_u64s`.
    unsafe fn create_resolve_buffer(&self, size: u64) -> native::WGPUBuffer {
        create_buffer(
            self.device,
            size,
            native::WGPUBufferUsage_QueryResolve
                | native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst,
        )
    }

    unsafe fn busy_dispatch(&self) -> BusyDispatch {
        let buffer = create_buffer(self.device, 4096, native::WGPUBufferUsage_Storage);
        let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
        entry.binding = 0;
        entry.visibility = native::WGPUShaderStage_Compute;
        entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
        let bgl_descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 1,
            entries: &entry,
        };
        let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(self.device, &bgl_descriptor);
        assert!(!bgl.is_null());
        let layout_descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let layout = yawgpu::wgpuDeviceCreatePipelineLayout(self.device, &layout_descriptor);
        assert!(!layout.is_null());
        let module = create_wgsl_module(self.device, BUSY_SHADER);
        let pipeline_descriptor = native::WGPUComputePipelineDescriptor {
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
        };
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(self.device, &pipeline_descriptor);
        assert!(!pipeline.is_null());
        let bind_entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer,
            offset: 0,
            size: 4096,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let bind_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: bgl,
            entryCount: 1,
            entries: &bind_entry,
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(self.device, &bind_descriptor);
        assert!(!bind_group.is_null());
        BusyDispatch {
            pipeline,
            bind_group,
            layout,
            bgl,
            module,
            buffer,
        }
    }

    unsafe fn create_color_target(&self) -> ColorTarget {
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: 4,
                height: 4,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(self.device, &descriptor);
        assert!(!texture.is_null());
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());
        ColorTarget { texture, view }
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }

    /// Copies `count` u64s at `offset` out of `buffer` into a fresh
    /// `COPY_DST | MAP_READ` readback buffer and maps them.
    unsafe fn read_u64s(
        &self,
        buffer: native::WGPUBuffer,
        offset: usize,
        count: usize,
    ) -> Vec<u64> {
        let len = count * 8;
        let readback = create_buffer(
            self.device,
            len as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(self.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            buffer,
            offset as u64,
            readback,
            0,
            len as u64,
        );
        self.submit(encoder);
        let values = read_buffer(self.instance, readback, 0, len)
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
            .collect();
        yawgpu::wgpuBufferRelease(readback);
        values
    }

    fn assert_no_errors(&self) {
        let errors = self.errors.lock().expect("error lock");
        assert!(errors.is_empty(), "unexpected device errors: {errors:?}");
    }

    unsafe fn release(self) {
        if !self.device.is_null() {
            yawgpu::wgpuQueueRelease(self.queue);
            yawgpu::wgpuDeviceRelease(self.device);
        }
        yawgpu::wgpuAdapterRelease(self.adapter);
        yawgpu::wgpuInstanceRelease(self.instance);
    }
}

impl BusyDispatch {
    /// Records one compute pass (optionally with `timestampWrites`) that
    /// dispatches 256 × 256 workgroups of the busy kernel.
    unsafe fn record(
        &self,
        encoder: native::WGPUCommandEncoder,
        timestamp_writes: *const native::WGPUPassTimestampWrites,
    ) {
        let descriptor = native::WGPUComputePassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            timestampWrites: timestamp_writes,
        };
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, self.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, self.bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 256, 256, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
    }

    unsafe fn release(self) {
        yawgpu::wgpuBindGroupRelease(self.bind_group);
        yawgpu::wgpuComputePipelineRelease(self.pipeline);
        yawgpu::wgpuShaderModuleRelease(self.module);
        yawgpu::wgpuPipelineLayoutRelease(self.layout);
        yawgpu::wgpuBindGroupLayoutRelease(self.bgl);
        yawgpu::wgpuBufferRelease(self.buffer);
    }
}

impl ColorTarget {
    unsafe fn release(self) {
        yawgpu::wgpuTextureViewRelease(self.view);
        yawgpu::wgpuTextureRelease(self.texture);
    }
}

unsafe fn record_clear_render_pass(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
    timestamp_writes: *const native::WGPUPassTimestampWrites,
) {
    let attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    };
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 1,
        colorAttachments: &attachment,
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: timestamp_writes,
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: usize,
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
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, offset, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, offset, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
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

unsafe fn create_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_VULKAN,
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

fn string_view(text: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: text.as_ptr().cast(),
        length: text.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
