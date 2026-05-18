use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn shader_modules_are_cached_by_source() {
    let test = ValidationTest::new();
    unsafe {
        let first = create_wgsl_module(test.device(), compute_source());
        let second = create_wgsl_module(test.device(), compute_source());
        let different = create_wgsl_module(test.device(), compute_source_different());

        assert_eq!(first, second);
        assert_ne!(first, different);
        assert!(test.errors().is_empty());

        yawgpu::wgpuShaderModuleRelease(different);
        yawgpu::wgpuShaderModuleRelease(second);
        yawgpu::wgpuShaderModuleRelease(first);
    }
}

#[test]
fn pipeline_layouts_are_cached_by_bind_group_layout_identity() {
    let test = ValidationTest::new();
    unsafe {
        let bgl_a = create_bind_group_layout(test.device(), &[uniform_layout(0)]);
        let bgl_b = create_bind_group_layout(test.device(), &[uniform_layout(0)]);
        let bgl_c = create_bind_group_layout(test.device(), &[uniform_layout(1)]);

        let first = create_pipeline_layout(test.device(), &[bgl_a, bgl_c]);
        let second = create_pipeline_layout(test.device(), &[bgl_a, bgl_c]);
        let different_element = create_pipeline_layout(test.device(), &[bgl_b, bgl_c]);
        let different_order = create_pipeline_layout(test.device(), &[bgl_c, bgl_a]);

        assert_eq!(first, second);
        assert_ne!(first, different_element);
        assert_ne!(first, different_order);
        assert!(test.errors().is_empty());

        yawgpu::wgpuPipelineLayoutRelease(different_order);
        yawgpu::wgpuPipelineLayoutRelease(different_element);
        yawgpu::wgpuPipelineLayoutRelease(second);
        yawgpu::wgpuPipelineLayoutRelease(first);
        yawgpu::wgpuBindGroupLayoutRelease(bgl_c);
        yawgpu::wgpuBindGroupLayoutRelease(bgl_b);
        yawgpu::wgpuBindGroupLayoutRelease(bgl_a);
    }
}

#[test]
fn compute_pipelines_are_cached_by_module_layout_and_constants() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(test.device(), compute_source());
        let different_module = create_wgsl_module(test.device(), compute_source_different());
        let layout_a = create_pipeline_layout(test.device(), &[]);
        let bgl = create_bind_group_layout(test.device(), &[uniform_layout(0)]);
        let layout_b = create_pipeline_layout(test.device(), &[bgl]);

        let first = create_compute_pipeline(test.device(), module, Some(layout_a), &[]);
        let second = create_compute_pipeline(test.device(), module, Some(layout_a), &[]);
        let different_module_pipeline =
            create_compute_pipeline(test.device(), different_module, Some(layout_a), &[]);
        let different_layout = create_compute_pipeline(test.device(), module, Some(layout_b), &[]);

        assert_eq!(first, second);
        assert_ne!(first, different_module_pipeline);
        assert_ne!(first, different_layout);

        let constants_module = create_wgsl_module(test.device(), compute_constant_source());
        let constants_a = [constant("value", 1.0)];
        let constants_a_again = [constant("value", 1.0)];
        let constants_b = [constant("value", 2.0)];
        let constant_pipeline =
            create_compute_pipeline(test.device(), constants_module, None, &constants_a);
        let constant_pipeline_again =
            create_compute_pipeline(test.device(), constants_module, None, &constants_a_again);
        let different_constant_pipeline =
            create_compute_pipeline(test.device(), constants_module, None, &constants_b);

        assert_eq!(constant_pipeline, constant_pipeline_again);
        assert_ne!(constant_pipeline, different_constant_pipeline);
        assert!(test.errors().is_empty());

        yawgpu::wgpuComputePipelineRelease(different_constant_pipeline);
        yawgpu::wgpuComputePipelineRelease(constant_pipeline_again);
        yawgpu::wgpuComputePipelineRelease(constant_pipeline);
        yawgpu::wgpuShaderModuleRelease(constants_module);
        yawgpu::wgpuComputePipelineRelease(different_layout);
        yawgpu::wgpuComputePipelineRelease(different_module_pipeline);
        yawgpu::wgpuComputePipelineRelease(second);
        yawgpu::wgpuComputePipelineRelease(first);
        yawgpu::wgpuPipelineLayoutRelease(layout_b);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuPipelineLayoutRelease(layout_a);
        yawgpu::wgpuShaderModuleRelease(different_module);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn render_pipelines_are_cached_by_layout_modules_and_constants() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = create_wgsl_module(test.device(), vertex_source());
        let different_vertex = create_wgsl_module(test.device(), vertex_source_different());
        let fragment = create_wgsl_module(test.device(), fragment_source());
        let layout_a = create_pipeline_layout(test.device(), &[]);
        let bgl = create_bind_group_layout(test.device(), &[uniform_layout(0)]);
        let layout_b = create_pipeline_layout(test.device(), &[bgl]);

        let first = create_render_pipeline(test.device(), vertex, fragment, Some(layout_a), &[]);
        let second = create_render_pipeline(test.device(), vertex, fragment, Some(layout_a), &[]);
        let different_module = create_render_pipeline(
            test.device(),
            different_vertex,
            fragment,
            Some(layout_a),
            &[],
        );
        let different_layout =
            create_render_pipeline(test.device(), vertex, fragment, Some(layout_b), &[]);

        assert_eq!(first, second);
        assert_ne!(first, different_module);
        assert_ne!(first, different_layout);

        let constant_vertex = create_wgsl_module(test.device(), vertex_constant_source());
        let constants_a = [constant("value", 1.0)];
        let constants_a_again = [constant("value", 1.0)];
        let constants_b = [constant("value", 2.0)];
        let constant_pipeline =
            create_render_pipeline(test.device(), constant_vertex, fragment, None, &constants_a);
        let constant_pipeline_again = create_render_pipeline(
            test.device(),
            constant_vertex,
            fragment,
            None,
            &constants_a_again,
        );
        let different_constant_pipeline =
            create_render_pipeline(test.device(), constant_vertex, fragment, None, &constants_b);

        assert_eq!(constant_pipeline, constant_pipeline_again);
        assert_ne!(constant_pipeline, different_constant_pipeline);
        assert!(test.errors().is_empty());

        yawgpu::wgpuRenderPipelineRelease(different_constant_pipeline);
        yawgpu::wgpuRenderPipelineRelease(constant_pipeline_again);
        yawgpu::wgpuRenderPipelineRelease(constant_pipeline);
        yawgpu::wgpuShaderModuleRelease(constant_vertex);
        yawgpu::wgpuRenderPipelineRelease(different_layout);
        yawgpu::wgpuRenderPipelineRelease(different_module);
        yawgpu::wgpuRenderPipelineRelease(second);
        yawgpu::wgpuRenderPipelineRelease(first);
        yawgpu::wgpuPipelineLayoutRelease(layout_b);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuPipelineLayoutRelease(layout_a);
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(different_vertex);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
}

#[test]
fn error_objects_are_not_cached() {
    let test = ValidationTest::new();
    unsafe {
        let mut first = std::ptr::null();
        test.assert_device_error_after(
            || {
                first = create_wgsl_module(test.device(), "this is not wgsl");
            },
            None,
        );
        let mut second = std::ptr::null();
        test.assert_device_error_after(
            || {
                second = create_wgsl_module(test.device(), "this is not wgsl");
            },
            None,
        );

        assert!(!first.is_null());
        assert!(!second.is_null());
        assert_ne!(first, second);

        yawgpu::wgpuShaderModuleRelease(second);
        yawgpu::wgpuShaderModuleRelease(first);
    }
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

unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: Option<native::WGPUPipelineLayout>,
    constants: &[PipelineConstantInput<'_>],
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
            entryPoint: empty_string_view(),
            constantCount: native_constants.len(),
            constants: native_constants.as_ptr(),
        },
    };
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
}

unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    vertex_module: native::WGPUShaderModule,
    fragment_module: native::WGPUShaderModule,
    layout: Option<native::WGPUPipelineLayout>,
    vertex_constants: &[PipelineConstantInput<'_>],
) -> native::WGPURenderPipeline {
    let native_vertex_constants = vertex_constants
        .iter()
        .map(|constant| native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(constant.key),
            value: constant.value,
        })
        .collect::<Vec<_>>();
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
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
        layout: layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: native_vertex_constants.len(),
            constants: native_vertex_constants.as_ptr(),
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
        fragment: &fragment,
    };
    yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor)
}

fn uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry
}

fn default_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Compute,
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
struct PipelineConstantInput<'a> {
    key: &'a str,
    value: f64,
}

fn constant(key: &str, value: f64) -> PipelineConstantInput<'_> {
    PipelineConstantInput { key, value }
}

fn compute_source() -> &'static str {
    "@compute @workgroup_size(1) fn main() {}"
}

fn compute_source_different() -> &'static str {
    "@compute @workgroup_size(2) fn main() {}"
}

fn compute_constant_source() -> &'static str {
    "override value: f32;
     @compute @workgroup_size(1) fn main() { _ = value; }"
}

fn vertex_source() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn vertex_source_different() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(1.0); }"
}

fn vertex_constant_source() -> &'static str {
    "override value: f32;
     @vertex fn vs() -> @builtin(position) vec4f { return vec4f(value); }"
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
