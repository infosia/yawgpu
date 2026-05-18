use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn vertex_buffer_count_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let buffers = vec![empty_vertex_buffer(); limits.maxVertexBuffers as usize + 1];
        assert_pipeline_error(&test, vertex_no_input(), &buffers);
    }
}

#[test]
fn vertex_attribute_count_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let attributes = (0..=limits.maxVertexAttributes)
            .map(|location| vertex_attribute(native::WGPUVertexFormat_Float32, 0, location))
            .collect::<Vec<_>>();
        let buffers = [vertex_buffer(4, &attributes)];
        assert_pipeline_error(&test, vertex_no_input(), &buffers);
    }
}

#[test]
fn vertex_buffer_stride_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let attributes = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];

        let zero_stride = [vertex_buffer(0, &attributes)];
        assert_pipeline_ok(&test, vertex_f32(), &zero_stride);

        let misaligned_stride = [vertex_buffer(6, &attributes)];
        assert_pipeline_error(&test, vertex_f32(), &misaligned_stride);

        let too_large_stride = [vertex_buffer(
            u64::from(limits.maxVertexBufferArrayStride) + 4,
            &attributes,
        )];
        assert_pipeline_error(&test, vertex_f32(), &too_large_stride);
    }
}

#[test]
fn vertex_attribute_ranges_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let misaligned = [vertex_attribute(native::WGPUVertexFormat_Float32, 2, 0)];
        let buffers = [vertex_buffer(8, &misaligned)];
        assert_pipeline_error(&test, vertex_f32(), &buffers);

        let out_of_stride = [vertex_attribute(native::WGPUVertexFormat_Float32x3, 0, 0)];
        let buffers = [vertex_buffer(8, &out_of_stride)];
        assert_pipeline_error(&test, vertex_f32(), &buffers);
    }
}

#[test]
fn vertex_attribute_locations_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let duplicate_a = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        let duplicate_b = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        let duplicate_buffers = [
            vertex_buffer(4, &duplicate_a),
            vertex_buffer(4, &duplicate_b),
        ];
        assert_pipeline_error(&test, vertex_f32(), &duplicate_buffers);

        let limits = device_limits(test.device());
        let too_high = [vertex_attribute(
            native::WGPUVertexFormat_Float32,
            0,
            limits.maxVertexAttributes,
        )];
        let buffers = [vertex_buffer(4, &too_high)];
        assert_pipeline_error(&test, vertex_f32(), &buffers);
    }
}

#[test]
fn vertex_shader_input_types_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let scalar_feeds_vector = [
            vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0),
            vertex_attribute(native::WGPUVertexFormat_Float32, 4, 1),
        ];
        let buffers = [vertex_buffer(8, &scalar_feeds_vector)];
        assert_pipeline_ok(&test, vertex_f32_and_vec3(), &buffers);

        let wrong_class = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        let buffers = [vertex_buffer(4, &wrong_class)];
        assert_pipeline_error(&test, vertex_i32(), &buffers);
    }
}

#[test]
fn vertex_shader_inputs_must_have_matching_attributes() {
    let test = ValidationTest::new();
    unsafe {
        let extra = [
            vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0),
            vertex_attribute(native::WGPUVertexFormat_Float32, 4, 1),
        ];
        let buffers = [vertex_buffer(8, &extra)];
        assert_pipeline_ok(&test, vertex_f32(), &buffers);

        let missing = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        let buffers = [vertex_buffer(4, &missing)];
        assert_pipeline_error(&test, vertex_location_two(), &buffers);
    }
}

unsafe fn assert_pipeline_ok(
    test: &ValidationTest,
    vertex_source: &str,
    buffers: &[native::WGPUVertexBufferLayout],
) {
    test.clear_errors();
    let pipeline = create_pipeline(test, vertex_source, buffers);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuRenderPipelineRelease(pipeline);
}

unsafe fn assert_pipeline_error(
    test: &ValidationTest,
    vertex_source: &str,
    buffers: &[native::WGPUVertexBufferLayout],
) {
    let mut pipeline = std::ptr::null();
    assert_device_error!({
        pipeline = create_pipeline(test, vertex_source, buffers);
    });
    assert!(!pipeline.is_null());
    yawgpu::wgpuRenderPipelineRelease(pipeline);
}

unsafe fn create_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    buffers: &[native::WGPUVertexBufferLayout],
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), fragment_single());
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment_state = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: buffers.len(),
            buffers: buffers.as_ptr(),
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
        fragment: &fragment_state,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    yawgpu::wgpuShaderModuleRelease(fragment_module);
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

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuDeviceGetLimits(device, &mut limits),
        native::WGPUStatus_Success
    );
    limits
}

fn vertex_buffer(
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Vertex,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

fn empty_vertex_buffer() -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Undefined,
        arrayStride: 0,
        attributeCount: 0,
        attributes: std::ptr::null(),
    }
}

fn vertex_attribute(
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
) -> native::WGPUVertexAttribute {
    native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format,
        offset,
        shaderLocation: shader_location,
    }
}

fn vertex_no_input() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn vertex_f32() -> &'static str {
    "@vertex fn vs(@location(0) a: f32) -> @builtin(position) vec4f { return vec4f(a); }"
}

fn vertex_i32() -> &'static str {
    "@vertex fn vs(@location(0) a: i32) -> @builtin(position) vec4f {
        return vec4f(f32(a));
    }"
}

fn vertex_f32_and_vec3() -> &'static str {
    "@vertex fn vs(@location(0) a: f32, @location(1) b: vec3f) -> @builtin(position) vec4f {
        return vec4f(a + b.x, b.yz, 1.0);
    }"
}

fn vertex_location_two() -> &'static str {
    "@vertex fn vs(@location(2) a: f32) -> @builtin(position) vec4f { return vec4f(a); }"
}

fn fragment_single() -> &'static str {
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
