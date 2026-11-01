//! Shared body of the Block 119 indirect dispatch e2e
//! (`specs/blocks/119-cts-d116bcc-findings.md` R5), included by
//! `e2e_metal_indirect_dispatch.rs` and `e2e_vulkan_indirect_dispatch.rs` through
//! `#[path]`. Noop cannot run the validation shader, so these scenarios pin what
//! only a GPU can show: invalid counts skip silently, and valid counts arrive
//! intact whichever way the arguments reach the buffer. Each scenario takes the real
//! backend to probe and the `YAWGPU_INSTANCE_BACKEND_*` selector.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{YaWGPUInstanceBackendSelect, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const COUNTER_SIZE: u64 = 16;

fn counting_shader(workgroup_size: u32, builtin: Option<&str>) -> String {
    let (parameter, read) = match builtin {
        Some(name) => (format!("@builtin({name}) input: u32"), "_ = input;\n"),
        None => (String::new(), ""),
    };
    format!(
        "@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;\n\
         @compute @workgroup_size({workgroup_size})\n\
         fn main({parameter}) {{\n{read}atomicAdd(&counter, 1u);\n}}\n"
    )
}

const WRITER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> args: array<u32, 3>;

@compute @workgroup_size(1)
fn main() {
    args[0] = 3u;
    args[1] = 2u;
    args[2] = 1u;
}
"#;

pub fn valid_counts_pass_through(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        assert_eq!(s.count_indirect(1, None, &[2, 3, 4], 0), 24);
        assert_eq!(s.count_indirect(2, None, &[3, 2, 1], 0), 12);
        s.finish();
    }
}

pub fn zero_counts_run_nothing_without_error(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        for builtin in [
            None,
            Some("global_invocation_index"),
            Some("workgroup_index"),
        ] {
            for args in [[0, 5, 1], [5, 0, 1], [5, 1, 0], [0, 0, 0]] {
                assert_eq!(
                    s.count_indirect(1, builtin, &args, 0),
                    0,
                    "{builtin:?} {args:?}"
                );
            }
        }
        s.finish();
    }
}

pub fn over_limit_counts_skip_on_every_pipeline(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let over = s.max_workgroups_per_dimension() + 1;
        for args in [[over, 1, 1], [1, over, 1], [1, 1, over], [u32::MAX, 1, 1]] {
            assert_eq!(s.count_indirect(1, None, &args, 0), 0, "{args:?}");
        }
        s.finish();
    }
}

pub fn linear_index_overflow_skips(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let max = s.max_workgroups_per_dimension();
        assert!(u64::from(max) * u64::from(max) * 2 > u64::from(u32::MAX));
        assert_eq!(
            s.count_indirect(2, Some("global_invocation_index"), &[max, max, 1], 0),
            0
        );
        assert_eq!(
            s.count_indirect(1, Some("workgroup_index"), &[max, max, 2], 0),
            0
        );
        assert_eq!(
            s.count_indirect(2, Some("global_invocation_index"), &[3, 2, 1], 0),
            12
        );
        assert_eq!(
            s.count_indirect(1, Some("workgroup_index"), &[3, 2, 1], 0),
            6
        );
        s.finish();
    }
}

pub fn unaligned_client_offsets_read_the_right_words(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let mut words = vec![999_999u32; 128];
        words[3..6].copy_from_slice(&[2, 3, 1]);
        assert_eq!(s.count_indirect(1, None, &words, 12), 6);
        words[64..67].copy_from_slice(&[2, 2, 1]);
        assert_eq!(s.count_indirect(1, None, &words, 256), 4);
        words[67..70].copy_from_slice(&[5, 1, 1]);
        assert_eq!(s.count_indirect(1, None, &words, 268), 5);
        s.finish();
    }
}

pub fn repeated_dispatches_in_one_pass_see_no_stale_value(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let over = s.max_workgroups_per_dimension() + 1;
        let valid = s.indirect_buffer(&[3, 2, 1]);
        let invalid = s.indirect_buffer(&[over, 1, 1]);
        let other = s.indirect_buffer(&[5, 1, 1]);
        let run = s.counting_run(1, None);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, run.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, run.bind_group, 0, std::ptr::null());
        for buffer in [valid, invalid, other, invalid, valid] {
            yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, buffer, 0);
        }
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        s.submit(encoder);
        assert_eq!(s.read_counter(&run), 6 + 5 + 6);
        for buffer in [valid, invalid, other] {
            yawgpu::wgpuBufferRelease(buffer);
        }
        run.release();
        s.finish();
    }
}

pub fn pass_state_survives_for_a_following_direct_dispatch(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let args = s.indirect_buffer(&[3, 2, 1]);
        let run = s.counting_run(1, None);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, run.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, run.bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args, 0);
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 4, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        s.submit(encoder);
        assert_eq!(s.read_counter(&run), 6 + 4);
        yawgpu::wgpuBufferRelease(args);
        run.release();
        s.finish();
    }
}

pub fn arguments_written_on_the_gpu_are_read_after_the_write(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let args = create_buffer(
            s.device,
            12,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_Indirect
                | native::WGPUBufferUsage_CopyDst,
        );
        let module = create_wgsl_module(s.device, WRITER_SHADER);
        let writer = s.create_pipeline(module);
        let writer_group = s.storage_bind_group(writer, args, 12);
        let run = s.counting_run(1, None);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, writer);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, writer_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, run.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, run.bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args, 0);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        s.submit(encoder);
        assert_eq!(s.read_counter(&run), 6);

        // `writeBuffer` then indirect, in a later submit on the same buffer.
        let bytes: Vec<u8> = [2u32, 2, 1].iter().flat_map(|w| w.to_le_bytes()).collect();
        yawgpu::wgpuQueueWriteBuffer(s.queue, args, 0, bytes.as_ptr().cast(), bytes.len());
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, run.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, run.bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args, 0);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        s.submit(encoder);
        assert_eq!(s.read_counter(&run), 6 + 4);

        yawgpu::wgpuBindGroupRelease(writer_group);
        yawgpu::wgpuComputePipelineRelease(writer);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(args);
        run.release();
        s.finish();
    }
}

struct CountingRun {
    module: native::WGPUShaderModule,
    pipeline: native::WGPUComputePipeline,
    counter: native::WGPUBuffer,
    readback: native::WGPUBuffer,
    bind_group: native::WGPUBindGroup,
}

impl CountingRun {
    unsafe fn release(self) {
        yawgpu::wgpuBindGroupRelease(self.bind_group);
        yawgpu::wgpuBufferRelease(self.readback);
        yawgpu::wgpuBufferRelease(self.counter);
        yawgpu::wgpuComputePipelineRelease(self.pipeline);
        yawgpu::wgpuShaderModuleRelease(self.module);
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
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
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

    /// Asserts no device error reached the application, then releases the session.
    unsafe fn finish(self) {
        assert!(
            self.errors.lock().expect("error lock").is_empty(),
            "{:?}",
            self.errors
        );
        yawgpu::wgpuQueueRelease(self.queue);
        yawgpu::wgpuDeviceRelease(self.device);
        yawgpu::wgpuAdapterRelease(self.adapter);
        yawgpu::wgpuInstanceRelease(self.instance);
    }

    unsafe fn max_workgroups_per_dimension(&self) -> u32 {
        let mut limits: native::WGPULimits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(self.device, &mut limits),
            native::WGPUStatus_Success
        );
        limits.maxComputeWorkgroupsPerDimension
    }

    unsafe fn indirect_buffer(&self, words: &[u32]) -> native::WGPUBuffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let buffer = create_buffer(
            self.device,
            bytes.len() as u64,
            native::WGPUBufferUsage_Indirect | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(self.queue, buffer, 0, bytes.as_ptr().cast(), bytes.len());
        buffer
    }

    unsafe fn create_pipeline(
        &self,
        module: native::WGPUShaderModule,
    ) -> native::WGPUComputePipeline {
        let descriptor = native::WGPUComputePipelineDescriptor {
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
        };
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(self.device, &descriptor);
        assert!(!pipeline.is_null());
        pipeline
    }

    unsafe fn storage_bind_group(
        &self,
        pipeline: native::WGPUComputePipeline,
        buffer: native::WGPUBuffer,
        size: u64,
    ) -> native::WGPUBindGroup {
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer,
            offset: 0,
            size,
            sampler: std::ptr::null_mut(),
            textureView: std::ptr::null_mut(),
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            self.device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout,
                entryCount: 1,
                entries: &entry,
            },
        );
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        assert!(!bind_group.is_null());
        bind_group
    }

    unsafe fn counting_run(&self, workgroup_size: u32, builtin: Option<&str>) -> CountingRun {
        let module = create_wgsl_module(self.device, &counting_shader(workgroup_size, builtin));
        let pipeline = self.create_pipeline(module);
        let counter = create_buffer(
            self.device,
            COUNTER_SIZE,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            self.device,
            COUNTER_SIZE,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let bind_group = self.storage_bind_group(pipeline, counter, COUNTER_SIZE);
        CountingRun {
            module,
            pipeline,
            counter,
            readback,
            bind_group,
        }
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }

    unsafe fn read_counter(&self, run: &CountingRun) -> u32 {
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(self.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            run.counter,
            0,
            run.readback,
            0,
            COUNTER_SIZE,
        );
        self.submit(encoder);
        let bytes = read_buffer(self.instance, run.readback, 4);
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    /// Runs one indirect dispatch and returns how many invocations executed.
    unsafe fn count_indirect(
        &self,
        workgroup_size: u32,
        builtin: Option<&str>,
        words: &[u32],
        offset: u64,
    ) -> u32 {
        let args = self.indirect_buffer(words);
        let run = self.counting_run(workgroup_size, builtin);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(self.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, run.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, run.bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, args, offset);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        self.submit(encoder);
        let count = self.read_counter(&run);
        yawgpu::wgpuBufferRelease(args);
        run.release();
        count
    }
}

unsafe fn read_buffer(
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

unsafe fn create_instance(select: u32) -> native::WGPUInstance {
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
