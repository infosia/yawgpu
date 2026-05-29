use std::ffi::CStr;
use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

#[derive(Default)]
pub struct PopState {
    pub calls: Vec<PopCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopCall {
    pub status: native::WGPUPopErrorScopeStatus,
    pub error_type: native::WGPUErrorType,
    pub message: String,
}

#[derive(Default)]
pub struct ComputePipelineAsyncState {
    pub calls: usize,
    pub status: native::WGPUCreatePipelineAsyncStatus,
    pub pipeline: native::WGPUComputePipeline,
}

#[derive(Default)]
pub struct RenderPipelineAsyncState {
    pub calls: usize,
    pub status: native::WGPUCreatePipelineAsyncStatus,
    pub pipeline: native::WGPURenderPipeline,
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

pub fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

pub unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    if value.length == native::WGPU_STRLEN {
        unsafe { CStr::from_ptr(value.data) }
            .to_string_lossy()
            .into_owned()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) };
        String::from_utf8_lossy(bytes).into_owned()
    }
}

pub unsafe fn pop_error_scope(
    device: native::WGPUDevice,
    state: &mut PopState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUPopErrorScopeCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(pop_error_scope_callback),
        userdata1: (state as *mut PopState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuDevicePopErrorScope(device, callback_info) }
}

pub unsafe fn pop_and_wait(test: &ValidationTest, state: &mut PopState) -> PopCall {
    let future = unsafe { pop_error_scope(test.device(), state) };
    unsafe { wait(test.instance(), future) };
    state
        .calls
        .last()
        .cloned()
        .expect("popErrorScope callback must fire")
}

unsafe extern "C" fn pop_error_scope_callback(
    status: native::WGPUPopErrorScopeStatus,
    error_type: native::WGPUErrorType,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = unsafe { &mut *(userdata1 as *mut PopState) };
    state.calls.push(PopCall {
        status,
        error_type,
        message: unsafe { string_view_to_string(message) },
    });
}

pub unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: u32::from(mapped_at_creation),
    };
    let buffer = unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) };
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
    width: u32,
    height: u32,
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
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    texture
}

pub unsafe fn create_texture_view(texture: native::WGPUTexture) -> native::WGPUTextureView {
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    view
}

pub unsafe fn create_sampler(device: native::WGPUDevice) -> native::WGPUSampler {
    let sampler = unsafe { yawgpu::wgpuDeviceCreateSampler(device, std::ptr::null()) };
    assert!(!sampler.is_null());
    sampler
}

pub unsafe fn create_query_set(
    device: native::WGPUDevice,
    query_type: native::WGPUQueryType,
    count: u32,
) -> native::WGPUQuerySet {
    let descriptor = native::WGPUQuerySetDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        type_: query_type,
        count,
    };
    let query_set = unsafe { yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor) };
    assert!(!query_set.is_null());
    query_set
}

pub fn uniform_layout_entry(
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    }
}

pub unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let layout = unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) };
    assert!(!layout.is_null());
    layout
}

pub fn buffer_binding(binding: u32, buffer: native::WGPUBuffer) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset: 0,
        size: 16,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

pub unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let group = unsafe { yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor) };
    assert!(!group.is_null());
    group
}

pub unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: layouts.len(),
        bindGroupLayouts: layouts.as_ptr(),
        immediateSize: 0,
    };
    let layout = unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) };
    assert!(!layout.is_null());
    layout
}

pub unsafe fn create_wgsl_module(
    device: native::WGPUDevice,
    source: &str,
) -> native::WGPUShaderModule {
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
    let module = unsafe { yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor) };
    assert!(!module.is_null());
    module
}

pub unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    entry_point: &str,
) -> native::WGPUComputePipeline {
    let descriptor = compute_pipeline_descriptor(module, entry_point);
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor) };
    assert!(!pipeline.is_null());
    pipeline
}

pub fn compute_pipeline_descriptor(
    module: native::WGPUShaderModule,
    entry_point: &str,
) -> native::WGPUComputePipelineDescriptor {
    native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view(entry_point),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    }
}

pub unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    vertex: native::WGPUShaderModule,
    fragment: native::WGPUShaderModule,
    fragment_entry: &str,
) -> native::WGPURenderPipeline {
    let target = color_target();
    let fragment_state = fragment_state(fragment, fragment_entry, &target);
    let descriptor = render_pipeline_descriptor(vertex, &fragment_state);
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor) };
    assert!(!pipeline.is_null());
    pipeline
}

pub fn color_target() -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_None,
    }
}

pub fn fragment_state(
    module: native::WGPUShaderModule,
    entry_point: &str,
    target: &native::WGPUColorTargetState,
) -> native::WGPUFragmentState {
    native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view(entry_point),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: target,
    }
}

pub fn render_pipeline_descriptor(
    vertex: native::WGPUShaderModule,
    fragment: &native::WGPUFragmentState,
) -> native::WGPURenderPipelineDescriptor {
    native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex,
            entryPoint: string_view("main"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment,
    }
}

pub unsafe fn create_command_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null()) };
    assert!(!encoder.is_null());
    encoder
}

pub unsafe fn finish_command_encoder(
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let command_buffer = unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
    assert!(!command_buffer.is_null());
    command_buffer
}

pub unsafe fn create_render_bundle_encoder(
    device: native::WGPUDevice,
) -> native::WGPURenderBundleEncoder {
    let formats = [native::WGPUTextureFormat_RGBA8Unorm];
    let descriptor = native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: formats.len(),
        colorFormats: formats.as_ptr(),
        depthStencilFormat: native::WGPUTextureFormat_Undefined,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    };
    let encoder = unsafe { yawgpu::wgpuDeviceCreateRenderBundleEncoder(device, &descriptor) };
    assert!(!encoder.is_null());
    encoder
}

pub fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
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
            a: 0.0,
        },
    }
}

pub fn render_pass_descriptor(
    attachment: &native::WGPURenderPassColorAttachment,
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 1,
        colorAttachments: attachment,
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

pub unsafe fn write_texture(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_row: u32,
) {
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: bytes_per_row,
        rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
    };
    let size = native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: 1,
    };
    unsafe {
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &destination,
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &size,
        );
    }
}

pub unsafe fn create_compute_pipeline_async(
    device: native::WGPUDevice,
    descriptor: &native::WGPUComputePipelineDescriptor,
    state: &mut ComputePipelineAsyncState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUCreateComputePipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(compute_pipeline_async_callback),
        userdata1: (state as *mut ComputePipelineAsyncState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuDeviceCreateComputePipelineAsync(device, descriptor, callback_info) }
}

pub unsafe fn create_render_pipeline_async(
    device: native::WGPUDevice,
    descriptor: &native::WGPURenderPipelineDescriptor,
    state: &mut RenderPipelineAsyncState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUCreateRenderPipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(render_pipeline_async_callback),
        userdata1: (state as *mut RenderPipelineAsyncState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuDeviceCreateRenderPipelineAsync(device, descriptor, callback_info) }
}

unsafe extern "C" fn compute_pipeline_async_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPUComputePipeline,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = unsafe { &mut *(userdata1 as *mut ComputePipelineAsyncState) };
    state.calls += 1;
    state.status = status;
    state.pipeline = pipeline;
}

unsafe extern "C" fn render_pipeline_async_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPURenderPipeline,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = unsafe { &mut *(userdata1 as *mut RenderPipelineAsyncState) };
    state.calls += 1;
    state.status = status;
    state.pipeline = pipeline;
}
