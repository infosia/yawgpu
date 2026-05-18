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
fn depth_stencil_aspects_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut color_depth = depth_state();
        color_depth.format = native::WGPUTextureFormat_RGBA8Unorm;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(color_depth),
            default_primitive(),
            default_multisample(),
        );

        let mut stencil_on_depth = depth_state();
        stencil_on_depth.stencilFront.failOp = native::WGPUStencilOperation_Replace;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(stencil_on_depth),
            default_primitive(),
            default_multisample(),
        );

        let mut missing_depth_settings = depth_state();
        missing_depth_settings.depthCompare = native::WGPUCompareFunction_Undefined;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            None,
            None,
            Some(missing_depth_settings),
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn fragment_depth_output_requires_depth_attachment() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(
                "@fragment fn fs() -> @builtin(frag_depth) f32 { return 0.5; }",
                None,
                1,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn color_target_formats_outputs_and_blending_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let no_alpha = [color_target_format(native::WGPUTextureFormat_R8Unorm)];
        let mut multisample = default_multisample();
        multisample.count = 4;
        multisample.alphaToCoverageEnabled = 1;
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                fragment_single(),
                None,
                &no_alpha,
            )),
            None,
            None,
            default_primitive(),
            multisample,
        );

        let non_renderable = [color_target_format(native::WGPUTextureFormat_RGBA8Snorm)];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                fragment_single(),
                None,
                &non_renderable,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );

        let hole = [color_target_with_write_mask(
            native::WGPUTextureFormat_Undefined,
            native::WGPUColorWriteMask_None,
        )];
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                "@fragment fn fs() {}",
                None,
                &hole,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );

        let blend = blend_state();
        let non_blendable = [color_target_with_blend(
            native::WGPUTextureFormat_RGBA8Uint,
            &blend,
        )];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                "@fragment fn fs() -> @location(0) vec4u { return vec4u(); }",
                None,
                &non_blendable,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );

        let no_output = [color_target()];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                "@fragment fn fs() {}",
                None,
                &no_output,
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
            Some(FragmentInput::new(
                "@fragment fn fs() -> @location(0) vec4u { return vec4u(); }",
                None,
                1,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );

        let undefined_with_blend = [color_target_with_blend(
            native::WGPUTextureFormat_Undefined,
            &blend,
        )];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                "@fragment fn fs() {}",
                None,
                &undefined_with_blend,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn color_target_bytes_per_sample_limit_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let targets = [
            color_target_with_write_mask(
                native::WGPUTextureFormat_RGBA32Float,
                native::WGPUColorWriteMask_None,
            ),
            color_target_with_write_mask(
                native::WGPUTextureFormat_RGBA32Float,
                native::WGPUColorWriteMask_None,
            ),
            color_target_with_write_mask(
                native::WGPUTextureFormat_RGBA32Float,
                native::WGPUColorWriteMask_None,
            ),
        ];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_targets(
                "@fragment fn fs() {}",
                None,
                &targets,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
        );
    }
}

#[test]
fn explicit_render_pipeline_layout_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let source = "struct U { value: vec4<f32> }
             @group(0) @binding(0) var<uniform> u: U;
             @fragment fn fs() -> @location(0) vec4f { return u.value; }";

        let empty_bgl = create_bind_group_layout(test.device(), &[]);
        let empty_layout = create_pipeline_layout(test.device(), &[empty_bgl]);
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(source, None, 1)),
            Some(empty_layout),
            None,
            default_primitive(),
            default_multisample(),
        );

        let wrong_visibility_bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Vertex, 16)],
        );
        let wrong_visibility_layout =
            create_pipeline_layout(test.device(), &[wrong_visibility_bgl]);
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(source, None, 1)),
            Some(wrong_visibility_layout),
            None,
            default_primitive(),
            default_multisample(),
        );

        let matching_bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Fragment, 16)],
        );
        let matching_layout = create_pipeline_layout(test.device(), &[matching_bgl]);
        assert_pipeline_ok(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::new(source, None, 1)),
            Some(matching_layout),
            None,
            default_primitive(),
            default_multisample(),
        );

        yawgpu::wgpuPipelineLayoutRelease(matching_layout);
        yawgpu::wgpuBindGroupLayoutRelease(matching_bgl);
        yawgpu::wgpuPipelineLayoutRelease(wrong_visibility_layout);
        yawgpu::wgpuBindGroupLayoutRelease(wrong_visibility_bgl);
        yawgpu::wgpuPipelineLayoutRelease(empty_layout);
        yawgpu::wgpuBindGroupLayoutRelease(empty_bgl);
    }
}

#[test]
fn render_pipeline_fragment_constants_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let constants = [constant("value", f64::NAN)];
        assert_pipeline_error(
            &test,
            vertex_single(),
            None,
            Some(FragmentInput::with_constants(
                "override value: f32; @fragment fn fs() -> @location(0) vec4f { return vec4f(value); }",
                None,
                1,
                &constants,
            )),
            None,
            None,
            default_primitive(),
            default_multisample(),
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
    let default_color_targets = [color_target()];
    let fragment_constants = fragment
        .map(|fragment| {
            fragment
                .constants
                .iter()
                .map(|constant| native::WGPUConstantEntry {
                    nextInChain: std::ptr::null_mut(),
                    key: string_view(constant.key),
                    value: constant.value,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let fragment_state = fragment.map(|fragment| native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module.expect("fragment module exists"),
        entryPoint: fragment.entry.map_or(empty_string_view(), string_view),
        constantCount: fragment_constants.len(),
        constants: fragment_constants.as_ptr(),
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
        depthWriteEnabled: native::WGPUOptionalBool_False,
        depthCompare: native::WGPUCompareFunction_Less,
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
    color_target_format(native::WGPUTextureFormat_RGBA8Unorm)
}

fn color_target_format(format: native::WGPUTextureFormat) -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    }
}

fn color_target_with_write_mask(
    format: native::WGPUTextureFormat,
    write_mask: native::WGPUColorWriteMask,
) -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format,
        blend: std::ptr::null(),
        writeMask: write_mask,
    }
}

fn color_target_with_blend(
    format: native::WGPUTextureFormat,
    blend: &native::WGPUBlendState,
) -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format,
        blend,
        writeMask: native::WGPUColorWriteMask_All,
    }
}

fn blend_state() -> native::WGPUBlendState {
    native::WGPUBlendState {
        color: blend_component(),
        alpha: blend_component(),
    }
}

fn blend_component() -> native::WGPUBlendComponent {
    native::WGPUBlendComponent {
        operation: native::WGPUBlendOperation_Add,
        srcFactor: native::WGPUBlendFactor_One,
        dstFactor: native::WGPUBlendFactor_Zero,
    }
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

fn uniform_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    min_binding_size: u64,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, visibility);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = min_binding_size;
    entry
}

fn default_layout(
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

#[derive(Clone, Copy)]
struct FragmentInput<'a> {
    source: &'a str,
    entry: Option<&'a str>,
    target_count: usize,
    targets: Option<&'a [native::WGPUColorTargetState]>,
    constants: &'a [PipelineConstantInput<'a>],
}

impl<'a> FragmentInput<'a> {
    fn new(source: &'a str, entry: Option<&'a str>, target_count: usize) -> Self {
        Self {
            source,
            entry,
            target_count,
            targets: None,
            constants: &[],
        }
    }

    fn with_targets(
        source: &'a str,
        entry: Option<&'a str>,
        targets: &'a [native::WGPUColorTargetState],
    ) -> Self {
        Self {
            source,
            entry,
            target_count: targets.len(),
            targets: Some(targets),
            constants: &[],
        }
    }

    fn with_constants(
        source: &'a str,
        entry: Option<&'a str>,
        target_count: usize,
        constants: &'a [PipelineConstantInput<'a>],
    ) -> Self {
        Self {
            source,
            entry,
            target_count,
            targets: None,
            constants,
        }
    }
}

#[derive(Clone, Copy)]
struct PipelineConstantInput<'a> {
    key: &'a str,
    value: f64,
}

fn constant(key: &str, value: f64) -> PipelineConstantInput<'_> {
    PipelineConstantInput { key, value }
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
