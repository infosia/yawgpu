#![cfg(feature = "vulkan")]
//! Block 117 (`specs/blocks/117-vulkan-submit-recycling.md`): many un-waited
//! submissions on one queue (non-blocking retirement, recycled command pools
//! and fences) and submissions from two threads at once. Each submission
//! updates a uniform with `writeBuffer` and accumulates it in a storage
//! buffer, so a recycled object reused too early, a missing dependency
//! between submissions, or a resource released before its submission retired
//! shows up as a wrong sum. Run under the Khronos layer with synchronization
//! validation as well.
#[path = "common/vulkan.rs"]
#[allow(dead_code)]
mod vulkan;
use vulkan::*;
use yawgpu::native;

const SHADER: &str = r#"
struct Params { value: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> acc: array<u32, 1>;
@compute @workgroup_size(1)
fn main() { acc[0] = acc[0] + params.value; }
"#;

/// More than the old 3-slot ring and the 64-frame high-water mark, so both
/// the harvest and the back-pressure paths run.
const SUBMISSIONS: u32 = 300;

#[derive(Clone, Copy)]
struct Handles {
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    pipeline: native::WGPUComputePipeline,
}

// SAFETY: webgpu.h objects are thread-safe; the test only shares handles that
// outlive the threads (joined inside `std::thread::scope`).
unsafe impl Send for Handles {}
unsafe impl Sync for Handles {}

unsafe fn accumulator_pipeline(device: native::WGPUDevice) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(device, SHADER);
    let descriptor = native::WGPUComputePipelineDescriptor {
        compute: native::WGPUComputeState {
            module,
            entryPoint: native::WGPUStringView {
                data: c"main".as_ptr(),
                length: 4,
            },
            ..std::mem::zeroed()
        },
        ..std::mem::zeroed()
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

/// Submits `count` dispatches that add `first..first + count` into `storage`,
/// one submission each, never waiting. Every bind group and uniform buffer is
/// released right after its submit, so only the queue's retention keeps them
/// alive until the GPU is done.
unsafe fn accumulate(handles: Handles, storage: native::WGPUBuffer, first: u32, count: u32) {
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(handles.pipeline, 0);
    for value in first..first + count {
        let uniform = create_buffer(
            handles.device,
            16,
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let bytes = [value, 0, 0, 0];
        yawgpu::wgpuQueueWriteBuffer(handles.queue, uniform, 0, bytes.as_ptr().cast(), 16);
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
                size: 4,
                ..std::mem::zeroed()
            },
        ];
        let group_descriptor = native::WGPUBindGroupDescriptor {
            layout,
            entryCount: entries.len(),
            entries: entries.as_ptr(),
            ..std::mem::zeroed()
        };
        let group = yawgpu::wgpuDeviceCreateBindGroup(handles.device, &group_descriptor);
        assert!(!group.is_null());
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(handles.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, handles.pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        let commands = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(handles.queue, 1, &commands);
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBufferRelease(uniform);
    }
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn zeroed_storage(ctx: &Ctx) -> native::WGPUBuffer {
    let storage = create_buffer(
        ctx.device,
        4,
        native::WGPUBufferUsage_Storage
            | native::WGPUBufferUsage_CopySrc
            | native::WGPUBufferUsage_CopyDst,
    );
    let zero = 0u32;
    yawgpu::wgpuQueueWriteBuffer(ctx.queue, storage, 0, (&zero as *const u32).cast(), 4);
    storage
}

unsafe fn read_u32(ctx: &Ctx, storage: native::WGPUBuffer) -> u32 {
    let readback = create_buffer(
        ctx.device,
        4,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
    );
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 4);
    ctx.submit(encoder);
    let bytes = read_buffer(ctx.instance, readback, 4);
    yawgpu::wgpuBufferRelease(readback);
    u32::from_le_bytes(bytes.try_into().expect("one word"))
}

fn sum(first: u32, count: u32) -> u32 {
    (first..first + count).sum()
}

/// Hundreds of un-waited submissions on one thread, then one readback.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_many_unwaited_submissions_accumulate_exactly() {
    with_device(&[], |ctx| unsafe {
        let pipeline = accumulator_pipeline(ctx.device);
        let storage = zeroed_storage(ctx);
        let handles = Handles {
            device: ctx.device,
            queue: ctx.queue,
            pipeline,
        };
        accumulate(handles, storage, 1, SUBMISSIONS);
        ctx.assert_clean();
        assert_eq!(read_u32(ctx, storage), sum(1, SUBMISSIONS));
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuComputePipelineRelease(pipeline);
    });
}

/// Two threads submit to the same queue at once, each into its own
/// accumulator, so one thread's retirement / back-pressure wait overlaps the
/// other thread's recording and submit.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_two_threads_submitting_concurrently_accumulate_exactly() {
    with_device(&[], |ctx| unsafe {
        let pipeline = accumulator_pipeline(ctx.device);
        let storages = [zeroed_storage(ctx), zeroed_storage(ctx)];
        let handles = Handles {
            device: ctx.device,
            queue: ctx.queue,
            pipeline,
        };
        struct SendBuffer(native::WGPUBuffer);
        // SAFETY: see `Handles`.
        unsafe impl Send for SendBuffer {}
        std::thread::scope(|scope| {
            for (thread, storage) in storages.iter().enumerate() {
                let storage = SendBuffer(*storage);
                let first = 1 + thread as u32 * 10_000;
                scope.spawn(move || {
                    let storage = storage;
                    accumulate(handles, storage.0, first, SUBMISSIONS / 2);
                });
            }
        });
        ctx.assert_clean();
        for (thread, storage) in storages.iter().enumerate() {
            let first = 1 + thread as u32 * 10_000;
            assert_eq!(
                read_u32(ctx, *storage),
                sum(first, SUBMISSIONS / 2),
                "thread {thread}"
            );
            yawgpu::wgpuBufferRelease(*storage);
        }
        yawgpu::wgpuComputePipelineRelease(pipeline);
    });
}
