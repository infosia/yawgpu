//! Shared body of the Block 129 e2e (`specs/blocks/129-vulkan-buffer-zero-init.md`),
//! included by `e2e_metal_buffer_zero_init.rs` and `e2e_vulkan_buffer_zero_init.rs`
//! through `#[path]`. WebGPU requires every new buffer to read as zero. Each round
//! fills buffers with a non-zero pattern and releases them, so the driver can hand the
//! same memory back to the next allocation; the next round's fresh buffers must still
//! read as zero, both through `mapAsync` and through `mappedAtCreation`.
//!
//! Whether a driver reuses freed memory is its own heuristic, so this is a regression
//! guard, not a reliable red check: on Windows NVIDIA a heavier variant (20 rounds of
//! 16 buffers in 8 sizes) failed once in four runs before the fix. The reliable red
//! check is the CTS `copyTextureToTexture:*` query, recorded in the block.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, install_error_capture, read_buffer, request_adapter,
    request_device,
};

/// Sizes small enough to come from a driver's suballocated pool and large enough to
/// take a dedicated allocation, plus an odd size that leaves allocation padding.
const SIZES: [usize; 4] = [256, 4 * 1024, 260 * 1024, 4 * 1024 * 1024 + 4];
/// Buffers of each size per round.
const PER_SIZE: usize = 4;
/// Fill-and-release rounds. Each round after the first checks the previous one's memory.
const ROUNDS: usize = 6;
/// The pattern written into every buffer before it is released.
const PATTERN: u8 = 0xA5;

/// A fresh `MapRead` buffer reads as zero after earlier buffers held `PATTERN`.
pub fn fresh_buffers_map_as_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        for round in 0..ROUNDS {
            let mut buffers = Vec::new();
            for &size in &SIZES {
                for _ in 0..PER_SIZE {
                    let buffer = create_buffer(
                        session.device,
                        size as u64,
                        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
                    );
                    let bytes = read_buffer(session.instance, buffer, size);
                    assert_all_zero(&bytes, round, size);
                    buffers.push((buffer, size));
                }
            }
            session.fill_and_release(buffers);
        }
        session.finish();
    }
}

/// A fresh `mappedAtCreation` buffer reads as zero after earlier buffers held `PATTERN`.
pub fn fresh_buffers_mapped_at_creation_are_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        for round in 0..ROUNDS {
            let mut buffers = Vec::new();
            for &size in &SIZES {
                for _ in 0..PER_SIZE {
                    let descriptor = native::WGPUBufferDescriptor {
                        nextInChain: std::ptr::null_mut(),
                        label: native::WGPUStringView {
                            data: std::ptr::null(),
                            length: 0,
                        },
                        usage: native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
                        size: size as u64,
                        mappedAtCreation: 1,
                    };
                    let buffer = yawgpu::wgpuDeviceCreateBuffer(session.device, &descriptor);
                    assert!(!buffer.is_null());
                    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, size);
                    assert!(!ptr.is_null());
                    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), size);
                    assert_all_zero(bytes, round, size);
                    yawgpu::wgpuBufferUnmap(buffer);
                    buffers.push((buffer, size));
                }
            }
            session.fill_and_release(buffers);
        }
        session.finish();
    }
}

fn assert_all_zero(bytes: &[u8], round: usize, size: usize) {
    if let Some(index) = bytes.iter().position(|&byte| byte != 0) {
        panic!(
            "round {round}: fresh {size}-byte buffer has {:#04x} at byte {index}, expected 0",
            bytes[index]
        );
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

    /// Writes `PATTERN` over every buffer, waits for the writes, and releases them.
    unsafe fn fill_and_release(&self, buffers: Vec<(native::WGPUBuffer, usize)>) {
        for &(buffer, size) in &buffers {
            let data = vec![PATTERN; size];
            yawgpu::wgpuQueueWriteBuffer(self.queue, buffer, 0, data.as_ptr().cast(), size);
        }
        self.wait_work_done();
        for (buffer, _) in buffers {
            yawgpu::wgpuBufferRelease(buffer);
        }
        // Let deferred destruction run before the next round allocates.
        self.wait_work_done();
    }

    /// Waits for `onSubmittedWorkDone` and asserts it reported success.
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
}

unsafe extern "C" fn work_done_callback(
    status: native::WGPUQueueWorkDoneStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUQueueWorkDoneStatus) = status;
}
