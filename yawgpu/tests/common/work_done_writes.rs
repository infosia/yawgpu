//! Shared body of the Block 121 e2e (`specs/blocks/121-work-done-pending-writes.md`),
//! included by `e2e_metal_work_done_writes.rs` and `e2e_vulkan_work_done_writes.rs`
//! through `#[path]`. `onSubmittedWorkDone` no longer flushes pending queue writes
//! (R1), and writes flush themselves once 16 MiB is staged (R2). These scenarios pin
//! R4 on a real GPU: the written bytes still reach a following submit and a following
//! `mapAsync`, in queue order, including across a threshold flush.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

use crate::indirect_dispatch::{
    create_buffer, create_instance, install_error_capture, read_buffer, request_adapter,
    request_device,
};

/// Dawn's `kPendingMemorySubmitThreshold`, mirrored by Block 121 R2.
const THRESHOLD: usize = 16 * 1024 * 1024;
/// One staging chunk per write, so the threshold falls between two writes.
const WRITE_SIZE: usize = 64 * 1024;

/// A write followed by work-done, then a submitted copy, reads the written bytes.
pub fn write_then_work_done_then_copy(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        let source = create_buffer(
            session.device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            session.device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let data: Vec<u8> = (1..=16).collect();
        yawgpu::wgpuQueueWriteBuffer(session.queue, source, 0, data.as_ptr().cast(), 16);
        session.wait_work_done();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, readback, 0, 16);
        session.submit(encoder);
        assert_eq!(read_buffer(session.instance, readback, 16), data);

        yawgpu::wgpuBufferRelease(source);
        yawgpu::wgpuBufferRelease(readback);
        session.finish();
    }
}

/// A write followed by work-done, then `mapAsync` on the written buffer, maps the
/// written bytes.
pub fn write_then_work_done_then_map(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        let buffer = create_buffer(
            session.device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let data: Vec<u8> = (101..=116).collect();
        yawgpu::wgpuQueueWriteBuffer(session.queue, buffer, 0, data.as_ptr().cast(), 16);
        session.wait_work_done();
        assert_eq!(read_buffer(session.instance, buffer, 16), data);

        yawgpu::wgpuBufferRelease(buffer);
        session.finish();
    }
}

/// Writes past the threshold, all to a small buffer, keep queue order: every region
/// holds the value of the last write to it, though a threshold flush fell between
/// writes to the same region. Through the C ABI this cannot see whether the flush
/// happened; the core unit tests pin R2 itself, and this pins R4 around it.
pub fn writes_across_the_threshold_keep_queue_order(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let session = Session::new(select);
        const REGIONS: usize = 8;
        let size = REGIONS * WRITE_SIZE;
        let buffer = create_buffer(
            session.device,
            size as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let writes = THRESHOLD / WRITE_SIZE + REGIONS + 3;
        let mut expected = vec![0u8; size];
        for index in 0..writes {
            let region = index % REGIONS;
            let value = (index % 251) as u8 + 1;
            let data = vec![value; WRITE_SIZE];
            yawgpu::wgpuQueueWriteBuffer(
                session.queue,
                buffer,
                (region * WRITE_SIZE) as u64,
                data.as_ptr().cast(),
                WRITE_SIZE,
            );
            expected[region * WRITE_SIZE..(region + 1) * WRITE_SIZE].fill(value);
            if index % 64 == 0 {
                session.wait_work_done();
            }
        }
        session.wait_work_done();
        assert!(read_buffer(session.instance, buffer, size) == expected);

        yawgpu::wgpuBufferRelease(buffer);
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

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
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
