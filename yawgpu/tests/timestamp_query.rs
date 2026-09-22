//! Timestamp recording, pass ordering and resolve readback on Noop.
use std::os::raw::c_void;
use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};
const EMPTY_LABEL: native::WGPUStringView = native::WGPUStringView {
    data: std::ptr::null(),
    length: 0,
};

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: EMPTY_LABEL,
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

/// Maps `buffer` for reading, copies out `size` bytes, and unmaps it.
unsafe fn read_back(test: &ValidationTest, buffer: native::WGPUBuffer, size: usize) -> Vec<u8> {
    let mut map_status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut map_status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, size, callback_info);
    wait(test.instance(), future);
    assert_eq!(map_status, native::WGPUMapAsyncStatus_Success);
    let mapped = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, size);
    assert!(!mapped.is_null());
    // Safety: the successful map exposes exactly `size` bytes until unmap.
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
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
        label: EMPTY_LABEL,
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

#[test]
fn timestamp_query_executes_and_resolves_zeroes_on_noop() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        let descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: EMPTY_LABEL,
            type_: native::WGPUQueryType_Timestamp,
            count: 4,
        };
        let set = yawgpu::wgpuDeviceCreateQuerySet(test.device(), &descriptor);
        let destination = create_buffer(
            test.device(),
            native::WGPUBufferUsage_QueryResolve | native::WGPUBufferUsage_CopySrc,
            32,
        );
        // MapRead cannot be combined with QueryResolve; copy into a readback buffer.
        let readback = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            32,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, set, 0);
        let writes = native::WGPUPassTimestampWrites {
            nextInChain: std::ptr::null_mut(),
            querySet: set,
            beginningOfPassWriteIndex: 1,
            endOfPassWriteIndex: 2,
        };
        let descriptor = native::WGPUComputePassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: EMPTY_LABEL,
            timestampWrites: &writes,
        };
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
        let source = "@compute @workgroup_size(1) fn main() {}";
        let mut wgsl = native::WGPUShaderSourceWGSL {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ShaderSourceWGSL,
            },
            code: native::WGPUStringView {
                data: source.as_ptr().cast(),
                length: source.len(),
            },
        };
        let module = yawgpu::wgpuDeviceCreateShaderModule(
            test.device(),
            &native::WGPUShaderModuleDescriptor {
                nextInChain: &mut wgsl.chain,
                label: EMPTY_LABEL,
            },
        );
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            test.device(),
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: EMPTY_LABEL,
                layout: std::ptr::null(),
                compute: native::WGPUComputeState {
                    nextInChain: std::ptr::null_mut(),
                    module,
                    entryPoint: EMPTY_LABEL,
                    constantCount: 0,
                    constants: std::ptr::null(),
                },
            },
        );
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: EMPTY_LABEL,
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
        let texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
        assert!(!texture.is_null());
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());

        let writes = native::WGPUPassTimestampWrites {
            nextInChain: std::ptr::null_mut(),
            querySet: set,
            beginningOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
            endOfPassWriteIndex: 3,
        };
        record_clear_render_pass(encoder, view, &writes);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, set, 0, 4, destination, 0);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, destination, 0, readback, 0, 32);
        let commands = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        yawgpu::wgpuQueueSubmit(queue, 1, &commands);
        assert_eq!(read_back(&test, readback, 32), [0; 32]);
        assert!(test.errors().is_empty(), "{:?}", test.errors());
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQuerySetRelease(set);
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQueueRelease(queue);
    }
}
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}
