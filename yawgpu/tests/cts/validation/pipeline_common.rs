use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[derive(Clone, Copy)]
pub struct PipelineConstantInput<'a> {
    pub key: &'a str,
    pub value: f64,
}

pub fn constant(key: &str, value: f64) -> PipelineConstantInput<'_> {
    PipelineConstantInput { key, value }
}

#[derive(Clone, Copy)]
pub struct FragmentInput<'a> {
    pub source: &'a str,
    pub entry: Option<&'a str>,
    pub target_count: usize,
    pub targets: Option<&'a [native::WGPUColorTargetState]>,
}

impl<'a> FragmentInput<'a> {
    pub fn new(source: &'a str, entry: Option<&'a str>, target_count: usize) -> Self {
        Self {
            source,
            entry,
            target_count,
            targets: None,
        }
    }
}

#[derive(Default)]
struct ComputePipelineAsyncState {
    calls: usize,
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPUComputePipeline,
}

#[derive(Default)]
struct RenderPipelineAsyncState {
    calls: usize,
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPURenderPipeline,
}

pub unsafe fn assert_compute_pipeline_ok(
    test: &ValidationTest,
    is_async: bool,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) {
    test.clear_errors();
    let pipeline = create_compute_pipeline(test, is_async, source, entry_point, constants, layout);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
}

pub unsafe fn assert_compute_pipeline_error(
    test: &ValidationTest,
    is_async: bool,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) {
    if is_async {
        let pipeline =
            unsafe { create_compute_pipeline(test, true, source, entry_point, constants, layout) };
        assert!(pipeline.is_null());
    } else {
        let mut pipeline = std::ptr::null();
        assert_device_error!({
            pipeline = unsafe {
                create_compute_pipeline(test, false, source, entry_point, constants, layout)
            };
        });
        assert!(!pipeline.is_null());
        unsafe {
            yawgpu::wgpuComputePipelineRelease(pipeline);
        }
    }
}

pub unsafe fn create_compute_pipeline(
    test: &ValidationTest,
    is_async: bool,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    let module = unsafe { create_wgsl_module(test.device(), source) };
    let pipeline = unsafe {
        create_compute_pipeline_with_module(
            test.instance(),
            test.device(),
            is_async,
            module,
            entry_point,
            constants,
            layout,
        )
    };
    unsafe {
        yawgpu::wgpuShaderModuleRelease(module);
    }
    pipeline
}

pub unsafe fn create_compute_pipeline_with_module(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    is_async: bool,
    module: native::WGPUShaderModule,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    let native_constants = constants
        .iter()
        .map(|constant| native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(constant.key),
            value: constant.value,
        })
        .collect::<Vec<_>>();
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: entry_point.map_or(empty_string_view(), string_view),
            constantCount: native_constants.len(),
            constants: native_constants.as_ptr(),
        },
    };
    if is_async {
        let mut state = ComputePipelineAsyncState::default();
        let callback_info = native::WGPUCreateComputePipelineAsyncCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(compute_pipeline_async_callback),
            userdata1: (&mut state as *mut ComputePipelineAsyncState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe {
            let future =
                yawgpu::wgpuDeviceCreateComputePipelineAsync(device, &descriptor, callback_info);
            wait(instance, future);
        }
        assert_eq!(state.calls, 1);
        match state.status {
            native::WGPUCreatePipelineAsyncStatus_Success => state.pipeline,
            native::WGPUCreatePipelineAsyncStatus_ValidationError => std::ptr::null(),
            status => panic!("unexpected async compute pipeline status {status}"),
        }
    } else {
        unsafe { yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor) }
    }
}

pub unsafe fn assert_render_pipeline_ok(
    test: &ValidationTest,
    is_async: bool,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
) {
    test.clear_errors();
    let pipeline = unsafe {
        create_render_pipeline(
            test,
            is_async,
            vertex_source,
            vertex_entry,
            fragment,
            layout,
            depth_stencil,
        )
    };
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

pub unsafe fn assert_render_pipeline_error(
    test: &ValidationTest,
    is_async: bool,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
) {
    if is_async {
        let pipeline = unsafe {
            create_render_pipeline(
                test,
                true,
                vertex_source,
                vertex_entry,
                fragment,
                layout,
                depth_stencil,
            )
        };
        assert!(pipeline.is_null());
    } else {
        let mut pipeline = std::ptr::null();
        assert_device_error!({
            pipeline = unsafe {
                create_render_pipeline(
                    test,
                    false,
                    vertex_source,
                    vertex_entry,
                    fragment,
                    layout,
                    depth_stencil,
                )
            };
        });
        assert!(!pipeline.is_null());
        unsafe {
            yawgpu::wgpuRenderPipelineRelease(pipeline);
        }
    }
}

pub unsafe fn assert_render_pipeline_descriptor(
    test: &ValidationTest,
    is_async: bool,
    success: bool,
    descriptor: &native::WGPURenderPipelineDescriptor,
) {
    if success {
        test.clear_errors();
        let pipeline =
            unsafe { create_render_pipeline_from_descriptor(test, is_async, descriptor) };
        assert!(!pipeline.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        unsafe {
            yawgpu::wgpuRenderPipelineRelease(pipeline);
        }
    } else if is_async {
        let pipeline = unsafe { create_render_pipeline_from_descriptor(test, true, descriptor) };
        assert!(pipeline.is_null());
    } else {
        let mut pipeline = std::ptr::null();
        assert_device_error!({
            pipeline = unsafe { create_render_pipeline_from_descriptor(test, false, descriptor) };
        });
        assert!(!pipeline.is_null());
        unsafe {
            yawgpu::wgpuRenderPipelineRelease(pipeline);
        }
    }
}

unsafe fn create_render_pipeline_from_descriptor(
    test: &ValidationTest,
    is_async: bool,
    descriptor: &native::WGPURenderPipelineDescriptor,
) -> native::WGPURenderPipeline {
    if is_async {
        let mut state = RenderPipelineAsyncState::default();
        let callback_info = native::WGPUCreateRenderPipelineAsyncCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(render_pipeline_async_callback),
            userdata1: (&mut state as *mut RenderPipelineAsyncState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe {
            let future = yawgpu::wgpuDeviceCreateRenderPipelineAsync(
                test.device(),
                descriptor,
                callback_info,
            );
            wait(test.instance(), future);
        }
        assert_eq!(state.calls, 1);
        match state.status {
            native::WGPUCreatePipelineAsyncStatus_Success => state.pipeline,
            native::WGPUCreatePipelineAsyncStatus_ValidationError => std::ptr::null(),
            status => panic!("unexpected async render pipeline status {status}"),
        }
    } else {
        unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), descriptor) }
    }
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    is_async: bool,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
) -> native::WGPURenderPipeline {
    let vertex_module = unsafe { create_wgsl_module(test.device(), vertex_source) };
    let fragment_module =
        fragment.map(|fragment| unsafe { create_wgsl_module(test.device(), fragment.source) });
    let default_color_targets = [color_target()];
    let fragment_state = fragment.map(|fragment| native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module.expect("fragment module exists"),
        entryPoint: fragment.entry.map_or(empty_string_view(), string_view),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: fragment.target_count,
        targets: if fragment.target_count == 0 {
            std::ptr::null()
        } else if let Some(targets) = fragment.targets {
            targets.as_ptr()
        } else {
            default_color_targets.as_ptr()
        },
    });
    let fragment_ptr = fragment_state
        .as_ref()
        .map_or(std::ptr::null(), |state| state as *const _);
    let depth_ptr = depth_stencil
        .as_ref()
        .map_or(std::ptr::null(), |state| state as *const _);
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: vertex_entry.map_or(empty_string_view(), string_view),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: depth_ptr,
        multisample: default_multisample(),
        fragment: fragment_ptr,
    };
    let pipeline = if is_async {
        let mut state = RenderPipelineAsyncState::default();
        let callback_info = native::WGPUCreateRenderPipelineAsyncCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(render_pipeline_async_callback),
            userdata1: (&mut state as *mut RenderPipelineAsyncState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe {
            let future = yawgpu::wgpuDeviceCreateRenderPipelineAsync(
                test.device(),
                &descriptor,
                callback_info,
            );
            wait(test.instance(), future);
        }
        assert_eq!(state.calls, 1);
        match state.status {
            native::WGPUCreatePipelineAsyncStatus_Success => state.pipeline,
            native::WGPUCreatePipelineAsyncStatus_ValidationError => std::ptr::null(),
            status => panic!("unexpected async render pipeline status {status}"),
        }
    } else {
        unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) }
    };
    if let Some(module) = fragment_module {
        unsafe {
            yawgpu::wgpuShaderModuleRelease(module);
        }
    }
    unsafe {
        yawgpu::wgpuShaderModuleRelease(vertex_module);
    }
    pipeline
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
    unsafe { yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor) }
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
    unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) }
}

pub unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: layouts.len(),
        bindGroupLayouts: layouts.as_ptr(),
        immediateSize: immediate_size,
    };
    unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) }
}

pub unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    #[derive(Default)]
    struct State {
        device: native::WGPUDevice,
    }
    unsafe extern "C" fn callback(
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestDeviceStatus_Success);
        unsafe {
            (*(userdata1 as *mut State)).device = device;
        }
    }

    let mut state = State::default();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(callback),
        userdata1: (&mut state as *mut State).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe {
        let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
        wait(instance, future);
    }
    assert!(!state.device.is_null());
    state.device
}

pub unsafe fn wait(instance: native::WGPUInstance, future: native::WGPUFuture) {
    let mut wait_info = native::WGPUFutureWaitInfo {
        future,
        completed: 0,
    };
    let status = unsafe { yawgpu::wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0) };
    assert_eq!(status, native::WGPUWaitStatus_Success);
    assert_eq!(wait_info.completed, 1);
}

pub fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut limits),
            native::WGPUStatus_Success
        );
        limits
    }
}

pub fn default_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
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

pub fn uniform_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    min_binding_size: u64,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, visibility);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = min_binding_size;
    entry
}

pub fn storage_texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    access: native::WGPUStorageTextureAccess,
    format: native::WGPUTextureFormat,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, visibility);
    entry.storageTexture.access = access;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = view_dimension;
    entry
}

pub fn depth_state() -> native::WGPUDepthStencilState {
    native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    }
}

pub fn color_target() -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    }
}

pub fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn default_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

fn stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Undefined,
        failOp: native::WGPUStencilOperation_Undefined,
        depthFailOp: native::WGPUStencilOperation_Undefined,
        passOp: native::WGPUStencilOperation_Undefined,
    }
}

unsafe extern "C" fn compute_pipeline_async_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPUComputePipeline,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    unsafe {
        let state = &mut *(userdata1 as *mut ComputePipelineAsyncState);
        state.calls += 1;
        state.status = status;
        state.pipeline = pipeline;
    }
}

unsafe extern "C" fn render_pipeline_async_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPURenderPipeline,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    unsafe {
        let state = &mut *(userdata1 as *mut RenderPipelineAsyncState);
        state.calls += 1;
        state.status = status;
        state.pipeline = pipeline;
    }
}
