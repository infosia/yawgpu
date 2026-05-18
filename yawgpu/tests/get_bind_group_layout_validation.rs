use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn get_bind_group_layout_index_range_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_compute_pipeline(&test, compute_uniform(), None);

        test.clear_errors();
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        assert!(!layout.is_null());
        assert!(test.errors().is_empty());

        let mut error_layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 1);
            },
            None,
        );
        assert!(!error_layout.is_null());

        yawgpu::wgpuBindGroupLayoutRelease(error_layout);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
}

#[test]
fn render_auto_layout_aggregates_bindings_across_stages() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline =
            create_render_pipeline(&test, vertex_small_uniform(), fragment_big_uniform());
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);

        let small_buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 16);
        let big_buffer = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 128);
        assert_bind_group_error(
            &test,
            layout,
            &[buffer_binding(0, small_buffer, 0, u64::MAX)],
        );
        assert_bind_group_ok(&test, layout, &[buffer_binding(0, big_buffer, 0, u64::MAX)]);

        yawgpu::wgpuBufferRelease(big_buffer);
        yawgpu::wgpuBufferRelease(small_buffer);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn default_bind_group_layouts_are_rejected_by_create_pipeline_layout() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_compute_pipeline(&test, compute_uniform(), None);
        let default_layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        assert_pipeline_layout_error(&test, &[default_layout]);

        let user_entry = uniform_layout(0, native::WGPUShaderStage_Compute, 16);
        let user_layout = create_bind_group_layout(test.device(), &[user_entry]);
        assert_pipeline_layout_ok(&test, &[user_layout]);

        yawgpu::wgpuBindGroupLayoutRelease(user_layout);
        yawgpu::wgpuBindGroupLayoutRelease(default_layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
}

#[test]
fn default_bind_group_layout_identity_is_pipeline_bound() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline_a = create_compute_pipeline(&test, compute_uniform(), None);
        let pipeline_b = create_compute_pipeline(&test, compute_uniform_different(), None);

        let a0 = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline_a, 0);
        let a0_again = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline_a, 0);
        let b0 = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline_b, 0);

        assert_eq!(a0, a0_again);
        assert_ne!(a0, b0);

        yawgpu::wgpuBindGroupLayoutRelease(b0);
        yawgpu::wgpuBindGroupLayoutRelease(a0_again);
        yawgpu::wgpuBindGroupLayoutRelease(a0);
        yawgpu::wgpuComputePipelineRelease(pipeline_b);
        yawgpu::wgpuComputePipelineRelease(pipeline_a);
    }
}

#[test]
fn texture_sample_type_is_derived_from_usage() {
    let test = ValidationTest::new();
    unsafe {
        let sampled_pipeline = create_compute_pipeline(&test, compute_texture_sample(), None);
        let sampled_layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(sampled_pipeline, 0);
        let load_pipeline = create_compute_pipeline(&test, compute_texture_load(), None);
        let load_layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(load_pipeline, 0);

        let sampler = create_sampler(test.device());
        let depth_texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24Plus,
        );
        let depth_view = yawgpu::wgpuTextureCreateView(depth_texture, std::ptr::null());

        assert_bind_group_error(
            &test,
            sampled_layout,
            &[sampler_binding(0, sampler), texture_binding(1, depth_view)],
        );
        assert_bind_group_ok(&test, load_layout, &[texture_binding(0, depth_view)]);

        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth_texture);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuBindGroupLayoutRelease(load_layout);
        yawgpu::wgpuComputePipelineRelease(load_pipeline);
        yawgpu::wgpuBindGroupLayoutRelease(sampled_layout);
        yawgpu::wgpuComputePipelineRelease(sampled_pipeline);
    }
}

#[test]
fn returned_and_error_bind_group_layouts_are_release_safe() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_compute_pipeline(&test, compute_uniform(), None);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        yawgpu::wgpuBindGroupLayoutAddRef(layout);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(layout);

        let mut error_layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 5);
            },
            None,
        );
        yawgpu::wgpuBindGroupLayoutRelease(error_layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
}

unsafe fn assert_pipeline_layout_ok(
    test: &ValidationTest,
    layouts: &[native::WGPUBindGroupLayout],
) {
    test.clear_errors();
    let pipeline_layout = create_pipeline_layout(test.device(), layouts);
    assert!(!pipeline_layout.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
}

unsafe fn assert_pipeline_layout_error(
    test: &ValidationTest,
    layouts: &[native::WGPUBindGroupLayout],
) {
    let mut pipeline_layout = std::ptr::null();
    test.assert_device_error_after(
        || {
            pipeline_layout = create_pipeline_layout(test.device(), layouts);
        },
        None,
    );
    assert!(!pipeline_layout.is_null());
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
}

unsafe fn assert_bind_group_ok(
    test: &ValidationTest,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) {
    test.clear_errors();
    let bind_group = create_bind_group(test.device(), layout, entries);
    assert!(!bind_group.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuBindGroupRelease(bind_group);
}

unsafe fn assert_bind_group_error(
    test: &ValidationTest,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) {
    let mut bind_group = std::ptr::null();
    test.assert_device_error_after(
        || {
            bind_group = create_bind_group(test.device(), layout, entries);
        },
        None,
    );
    assert!(!bind_group.is_null());
    yawgpu::wgpuBindGroupRelease(bind_group);
}

unsafe fn create_compute_pipeline(
    test: &ValidationTest,
    source: &str,
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(test.device(), source);
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    fragment_source: &str,
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), fragment_source);
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
        fragment: &fragment,
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

unsafe fn create_bind_group(
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
    yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor)
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}

unsafe fn create_sampler(device: native::WGPUDevice) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_Undefined,
        addressModeV: native::WGPUAddressMode_Undefined,
        addressModeW: native::WGPUAddressMode_Undefined,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Undefined,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    };
    yawgpu::wgpuDeviceCreateSampler(device, &descriptor)
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    yawgpu::wgpuDeviceCreateTexture(device, &descriptor)
}

fn buffer_binding(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn sampler_binding(binding: u32, sampler: native::WGPUSampler) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler,
        textureView: std::ptr::null(),
    }
}

fn texture_binding(
    binding: u32,
    texture_view: native::WGPUTextureView,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler: std::ptr::null(),
        textureView: texture_view,
    }
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

fn compute_uniform() -> &'static str {
    "struct U { values: array<vec4f, 4> }
     @group(0) @binding(0) var<uniform> u: U;
     @compute @workgroup_size(1) fn main() { _ = u.values[0]; }"
}

fn compute_uniform_different() -> &'static str {
    "struct U { values: array<vec4f, 4> }
     @group(0) @binding(0) var<uniform> u: U;
     @compute @workgroup_size(2) fn main() { _ = u.values[1]; }"
}

fn compute_texture_sample() -> &'static str {
    "@group(0) @binding(0) var samp: sampler;
     @group(0) @binding(1) var tex: texture_2d<f32>;
     @compute @workgroup_size(1) fn main() {
         _ = textureSampleLevel(tex, samp, vec2f(0.5), 0.0);
     }"
}

fn compute_texture_load() -> &'static str {
    "@group(0) @binding(0) var tex: texture_2d<f32>;
     @compute @workgroup_size(1) fn main() {
         _ = textureLoad(tex, vec2i(0), 0);
     }"
}

fn vertex_small_uniform() -> &'static str {
    "struct U { value: f32 }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs() -> @builtin(position) vec4f {
         return vec4f(u.value);
     }"
}

fn fragment_big_uniform() -> &'static str {
    "struct U { values: array<vec4f, 4> }
     @group(0) @binding(0) var<uniform> u: U;
     @fragment fn fs() -> @location(0) vec4f {
         return u.values[3];
     }"
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
