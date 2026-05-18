use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn vertex_entry_point_resolution_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            fragment_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            "@vertex fn a() -> @builtin(position) vec4f { return vec4f(); }
             @vertex fn b() -> @builtin(position) vec4f { return vec4f(); }",
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            vertex_single(),
            Some("missing"),
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_ok(
            &test,
            "@vertex fn a() -> @builtin(position) vec4f { return vec4f(); }
             @vertex fn b() -> @builtin(position) vec4f { return vec4f(); }",
            Some("b"),
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn fragment_entry_point_resolution_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(fragment_single(), None, 1)),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(vertex_single(), None, 1)),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(
                "@fragment fn a() -> @location(0) vec4f { return vec4f(); }
                 @fragment fn b() -> @location(0) vec4f { return vec4f(); }",
                None,
                1,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(fragment_single(), Some("missing"), 1)),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn render_target_or_depth_stencil_presence_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(fragment_single(), None, 0)),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn primitive_strip_index_format_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut primitive = default_primitive();
        primitive.stripIndexFormat = native::WGPUIndexFormat_Uint16;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            primitive,
            default_multisample(),
        );

        primitive.topology = native::WGPUPrimitiveTopology_TriangleStrip;
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            primitive,
            default_multisample(),
        );
    }
}

#[test]
fn depth_bias_values_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut depth = depth_state();
        depth.depthBiasSlopeScale = f32::NAN;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth),
            default_primitive(),
            default_multisample(),
        );

        depth = depth_state();
        depth.depthBiasClamp = f32::INFINITY;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth),
            default_primitive(),
            default_multisample(),
        );

        depth = depth_state();
        depth.depthBias = 1;
        let mut primitive = default_primitive();
        primitive.topology = native::WGPUPrimitiveTopology_PointList;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth),
            primitive,
            default_multisample(),
        );

        primitive.topology = native::WGPUPrimitiveTopology_TriangleList;
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth),
            primitive,
            default_multisample(),
        );
    }
}

#[test]
fn multisample_state_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut multisample = default_multisample();
        multisample.count = 2;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            multisample,
        );

        multisample.count = 1;
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            multisample,
        );

        multisample.count = 4;
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            multisample,
        );

        multisample.count = 1;
        multisample.alphaToCoverageEnabled = 1;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(fragment_single(), None, 1)),
            None,
            None,
            default_primitive(),
            multisample,
        );

        multisample.count = 4;
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(fragment_single(), None, 1)),
            None,
            None,
            default_primitive(),
            multisample,
        );
    }
}

#[test]
fn alpha_to_coverage_rejects_fragment_sample_mask_output() {
    let test = ValidationTest::new();
    unsafe {
        let mut multisample = default_multisample();
        multisample.count = 4;
        multisample.alphaToCoverageEnabled = 1;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(
                "@fragment fn fs() -> @builtin(sample_mask) u32 { return 1u; }",
                None,
                1,
            )),
            None,
            None,
            default_primitive(),
            multisample,
        );
    }
}

#[test]
fn render_pipeline_release_is_safe_for_valid_and_error_pipelines() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_pipeline(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(depth_state()),
            default_primitive(),
            default_multisample(),
        );
        yawgpu::wgpuRenderPipelineAddRef(pipeline);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuRenderPipelineRelease(pipeline);

        let mut error_pipeline = std::ptr::null();
        assert_device_error!({
            error_pipeline = create_pipeline(
                &test,
                vertex_single(),
                Some("missing"),
                None,
                None,
                Some(depth_state()),
                default_primitive(),
                default_multisample(),
            );
        });
        assert!(!error_pipeline.is_null());
        yawgpu::wgpuRenderPipelineRelease(error_pipeline);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn assert_pipeline_ok(
    test: &ValidationTest,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
    primitive: native::WGPUPrimitiveState,
    multisample: native::WGPUMultisampleState,
) {
    test.clear_errors();
    let pipeline = create_pipeline(
        test,
        vertex_source,
        vertex_entry,
        fragment,
        layout,
        depth_stencil,
        primitive,
        multisample,
    );
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuRenderPipelineRelease(pipeline);
}

#[allow(clippy::too_many_arguments)]
unsafe fn assert_pipeline_error(
    test: &ValidationTest,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
    primitive: native::WGPUPrimitiveState,
    multisample: native::WGPUMultisampleState,
) {
    let mut pipeline = std::ptr::null();
    assert_device_error!({
        pipeline = create_pipeline(
            test,
            vertex_source,
            vertex_entry,
            fragment,
            layout,
            depth_stencil,
            primitive,
            multisample,
        );
    });
    assert!(!pipeline.is_null());
    yawgpu::wgpuRenderPipelineRelease(pipeline);
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    vertex_entry: Option<&str>,
    fragment: Option<FragmentInput<'_>>,
    layout: Option<native::WGPUPipelineLayout>,
    depth_stencil: Option<native::WGPUDepthStencilState>,
    primitive: native::WGPUPrimitiveState,
    multisample: native::WGPUMultisampleState,
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module =
        fragment.map(|fragment| create_wgsl_module(test.device(), fragment.source));
    let color_targets = [color_target()];
    let fragment_state = fragment.map(|fragment| native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module.expect("fragment module exists"),
        entryPoint: fragment.entry.map_or(empty_string_view(), string_view),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: fragment.target_count,
        targets: if fragment.target_count == 0 {
            std::ptr::null()
        } else {
            color_targets.as_ptr()
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
        primitive,
        depthStencil: depth_ptr,
        multisample,
        fragment: fragment_ptr,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    if let Some(module) = fragment_module {
        yawgpu::wgpuShaderModuleRelease(module);
    }
    yawgpu::wgpuShaderModuleRelease(vertex_module);
    pipeline
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

fn vertex_single() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn fragment_single() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
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

fn depth_state() -> native::WGPUDepthStencilState {
    native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth24Plus,
        depthWriteEnabled: native::WGPUOptionalBool_Undefined,
        depthCompare: native::WGPUCompareFunction_Undefined,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
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

fn color_target() -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    }
}

#[derive(Clone, Copy)]
struct FragmentInput<'a> {
    source: &'a str,
    entry: Option<&'a str>,
    target_count: usize,
}

impl<'a> FragmentInput<'a> {
    fn new(source: &'a str, entry: Option<&'a str>, target_count: usize) -> Self {
        Self {
            source,
            entry,
            target_count,
        }
    }
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
