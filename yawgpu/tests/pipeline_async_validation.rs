use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Default)]
struct PipelineAsyncState<T> {
    calls: u32,
    statuses: Vec<native::WGPUCreatePipelineAsyncStatus>,
    pipelines: Vec<T>,
    messages: Vec<String>,
}

#[test]
fn valid_compute_pipeline_async_succeeds_and_returns_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), compute_source());
        let descriptor = compute_pipeline_descriptor(module);
        let mut state = PipelineAsyncState::<native::WGPUComputePipeline>::default();

        let _future = create_compute_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        assert_eq!(state.calls, 0);
        yawgpu::wgpuInstanceProcessEvents(test.instance());

        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCreatePipelineAsyncStatus_Success]
        );
        assert!(state.messages[0].is_empty());
        let pipeline = state.pipelines[0];
        assert!(!pipeline.is_null());

        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        assert!(!layout.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn invalid_compute_pipeline_async_reports_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), vertex_source());
        let descriptor = compute_pipeline_descriptor(module);
        let mut state = PipelineAsyncState::<native::WGPUComputePipeline>::default();

        test.clear_errors();
        create_compute_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        assert!(test.errors().is_empty());
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert!(test.errors().is_empty());

        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCreatePipelineAsyncStatus_ValidationError]
        );
        assert!(state.pipelines[0].is_null());
        assert!(!state.messages[0].is_empty());

        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn valid_render_pipeline_async_succeeds_and_returns_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = create_wgsl_module(test.device(), vertex_source());
        let fragment = create_wgsl_module(test.device(), fragment_source());
        let descriptor = render_pipeline_descriptor(vertex, fragment);
        let mut state = PipelineAsyncState::<native::WGPURenderPipeline>::default();

        create_render_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());

        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCreatePipelineAsyncStatus_Success]
        );
        let pipeline = state.pipelines[0];
        assert!(!pipeline.is_null());

        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        assert!(!layout.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
}

#[test]
fn invalid_render_pipeline_async_reports_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = create_wgsl_module(test.device(), vertex_source());
        let fragment = create_wgsl_module(test.device(), vertex_source());
        let descriptor = render_pipeline_descriptor(vertex, fragment);
        let mut state = PipelineAsyncState::<native::WGPURenderPipeline>::default();

        test.clear_errors();
        create_render_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        assert!(test.errors().is_empty());
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert!(test.errors().is_empty());

        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCreatePipelineAsyncStatus_ValidationError]
        );
        assert!(state.pipelines[0].is_null());
        assert!(!state.messages[0].is_empty());

        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
}

#[test]
fn wait_any_only_pipeline_async_waits_for_wait_any() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), compute_source());
        let descriptor = compute_pipeline_descriptor(module);
        let mut state = PipelineAsyncState::<native::WGPUComputePipeline>::default();

        let future = create_compute_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut state,
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 0);

        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        assert_eq!(
            yawgpu::wgpuInstanceWaitAny(test.instance(), 1, &mut wait_info, 0),
            native::WGPUWaitStatus_Success
        );
        assert_eq!(wait_info.completed, 1);
        assert_eq!(state.calls, 1);

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);

        yawgpu::wgpuComputePipelineRelease(state.pipelines[0]);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn process_events_pipeline_async_fires_once() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), compute_source());
        let descriptor = compute_pipeline_descriptor(module);
        let mut state = PipelineAsyncState::<native::WGPUComputePipeline>::default();

        create_compute_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        yawgpu::wgpuInstanceProcessEvents(test.instance());

        assert_eq!(state.calls, 1);
        yawgpu::wgpuComputePipelineRelease(state.pipelines[0]);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn async_pipeline_reuses_sync_cache_handle() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), compute_source());
        let bind_group_layout = create_bind_group_layout(test.device(), &[uniform_layout(0)]);
        let pipeline_layout = create_pipeline_layout(test.device(), &[bind_group_layout]);
        let mut descriptor = compute_pipeline_descriptor(module);
        descriptor.layout = pipeline_layout;
        let sync_pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
        let mut state = PipelineAsyncState::<native::WGPUComputePipeline>::default();

        create_compute_pipeline_async(
            test.device(),
            &descriptor,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());

        assert_eq!(state.calls, 1);
        assert_eq!(sync_pipeline, state.pipelines[0]);

        yawgpu::wgpuComputePipelineRelease(state.pipelines[0]);
        yawgpu::wgpuComputePipelineRelease(sync_pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bind_group_layout);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

unsafe fn create_compute_pipeline_async(
    device: native::WGPUDevice,
    descriptor: &native::WGPUComputePipelineDescriptor,
    mode: native::WGPUCallbackMode,
    state: &mut PipelineAsyncState<native::WGPUComputePipeline>,
) -> native::WGPUFuture {
    let callback_info = native::WGPUCreateComputePipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode,
        callback: Some(compute_pipeline_callback),
        userdata1: (state as *mut PipelineAsyncState<native::WGPUComputePipeline>).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuDeviceCreateComputePipelineAsync(device, descriptor, callback_info)
}

unsafe fn create_render_pipeline_async(
    device: native::WGPUDevice,
    descriptor: &native::WGPURenderPipelineDescriptor,
    mode: native::WGPUCallbackMode,
    state: &mut PipelineAsyncState<native::WGPURenderPipeline>,
) -> native::WGPUFuture {
    let callback_info = native::WGPUCreateRenderPipelineAsyncCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode,
        callback: Some(render_pipeline_callback),
        userdata1: (state as *mut PipelineAsyncState<native::WGPURenderPipeline>).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuDeviceCreateRenderPipelineAsync(device, descriptor, callback_info)
}

unsafe extern "C" fn compute_pipeline_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPUComputePipeline,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut PipelineAsyncState<native::WGPUComputePipeline>);
    state.calls += 1;
    state.statuses.push(status);
    state.pipelines.push(pipeline);
    state.messages.push(string_view_to_string(message));
}

unsafe extern "C" fn render_pipeline_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPURenderPipeline,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut PipelineAsyncState<native::WGPURenderPipeline>);
    state.calls += 1;
    state.statuses.push(status);
    state.pipelines.push(pipeline);
    state.messages.push(string_view_to_string(message));
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
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
}

unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor)
}

unsafe fn create_pipeline_layout(
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
    yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor)
}

fn compute_pipeline_descriptor(
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipelineDescriptor {
    native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    }
}

fn render_pipeline_descriptor(
    vertex_module: native::WGPUShaderModule,
    fragment_module: native::WGPUShaderModule,
) -> native::WGPURenderPipelineDescriptor {
    let color_target = Box::leak(Box::new(native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    }));
    let fragment = Box::leak(Box::new(native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: color_target,
    }));
    native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
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

fn uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Compute,
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

fn compute_source() -> &'static str {
    "@group(0) @binding(0) var<uniform> u: vec4f;
     @compute @workgroup_size(1) fn main() { _ = u; }"
}

fn vertex_source() -> &'static str {
    "@group(0) @binding(0) var<uniform> u: vec4f;
     @vertex fn vs() -> @builtin(position) vec4f { return u; }"
}

fn fragment_source() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
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

fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.length == 0 || value.data.is_null() {
        return String::new();
    }
    unsafe {
        let bytes = std::slice::from_raw_parts(value.data.cast::<u8>(), value.length);
        String::from_utf8_lossy(bytes).into_owned()
    }
}
