use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn entry_point_resolution_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_ok(
            &test,
            "@compute @workgroup_size(1) fn main() {}",
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }",
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            "@compute @workgroup_size(1) fn a() {}
             @compute @workgroup_size(1) fn b() {}",
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            "@compute @workgroup_size(1) fn main() {}",
            Some("missing"),
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @compute @workgroup_size(1) fn cs() {}",
            Some("vs"),
            &[],
            None,
        );
        assert_pipeline_ok(
            &test,
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @compute @workgroup_size(1) fn cs() {}",
            Some("cs"),
            &[],
            None,
        );
    }
}

#[test]
fn workgroup_limits_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        assert_pipeline_ok(
            &test,
            &format!(
                "@compute @workgroup_size({}, {}, {}) fn main() {{}}",
                limits.maxComputeWorkgroupSizeX.min(1),
                limits.maxComputeWorkgroupSizeY.min(1),
                limits.maxComputeWorkgroupSizeZ.min(1)
            ),
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            &format!(
                "@compute @workgroup_size({}, 1, 1) fn main() {{}}",
                limits.maxComputeWorkgroupSizeX + 1
            ),
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            &format!(
                "@compute @workgroup_size(1, {}, 1) fn main() {{}}",
                limits.maxComputeWorkgroupSizeY + 1
            ),
            None,
            &[],
            None,
        );
        assert_pipeline_error(
            &test,
            &format!(
                "@compute @workgroup_size(1, 1, {}) fn main() {{}}",
                limits.maxComputeWorkgroupSizeZ + 1
            ),
            None,
            &[],
            None,
        );
        let product_x = limits
            .maxComputeWorkgroupSizeX
            .min(limits.maxComputeInvocationsPerWorkgroup);
        assert_pipeline_error(
            &test,
            &format!(
                "@compute @workgroup_size({}, 2, 1) fn main() {{}}",
                product_x
            ),
            None,
            &[],
            None,
        );

        let override_source = "override wg: u32; @compute @workgroup_size(wg, 1, 1) fn main() {}";
        assert_pipeline_ok(
            &test,
            override_source,
            None,
            &[constant("wg", f64::from(limits.maxComputeWorkgroupSizeX))],
            None,
        );
        assert_pipeline_error(
            &test,
            override_source,
            None,
            &[constant(
                "wg",
                f64::from(limits.maxComputeWorkgroupSizeX + 1),
            )],
            None,
        );
    }
}

#[test]
fn workgroup_storage_limit_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let count = limits.maxComputeWorkgroupStorageSize / 4 + 1;
        assert_pipeline_error(
            &test,
            &format!(
                "var<workgroup> scratch: array<u32, {count}>;
                 @compute @workgroup_size(1) fn main() {{ scratch[0] = 1u; }}"
            ),
            None,
            &[],
            None,
        );
    }
}

#[test]
fn overridable_constant_keys_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let source = "override a: f32;
             @id(1) override b: i32;
             override c: u32 = 1u;
             @compute @workgroup_size(1) fn main() {}";
        assert_pipeline_ok(
            &test,
            source,
            None,
            &[constant("a", 1.0), constant("1", 2.0)],
            None,
        );
        assert_pipeline_error(
            &test,
            source,
            None,
            &[constant("missing", 1.0), constant("1", 2.0)],
            None,
        );
        assert_pipeline_error(
            &test,
            source,
            None,
            &[constant("a", 1.0), constant("b", 2.0)],
            None,
        );
        assert_pipeline_error(
            &test,
            source,
            None,
            &[constant("a", 1.0), constant("a", 2.0), constant("1", 3.0)],
            None,
        );
        assert_pipeline_error(&test, source, None, &[constant("a", 1.0)], None);
    }
}

#[test]
fn overridable_constant_values_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_pipeline_ok(
            &test,
            "override value: f32; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", f64::from(f32::MAX))],
            None,
        );
        assert_pipeline_error(
            &test,
            "override value: f32; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", f64::NAN)],
            None,
        );
        assert_pipeline_error(
            &test,
            "override value: f32; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", f64::INFINITY)],
            None,
        );
        assert_pipeline_error(
            &test,
            "override value: i32; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", f64::from(i32::MAX) + 1.0)],
            None,
        );
        assert_pipeline_error(
            &test,
            "override value: u32; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", -1.0)],
            None,
        );
        assert_pipeline_error(
            &test,
            "enable f16; override value: f16; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", 65_505.0)],
            None,
        );
        assert_pipeline_ok(
            &test,
            "override value: bool; @compute @workgroup_size(1) fn main() {}",
            None,
            &[constant("value", 1.0)],
            None,
        );
    }
}

#[test]
fn explicit_and_auto_layouts_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let source = "struct U { value: vec4<f32> }
             @group(0) @binding(0) var<uniform> u: U;
             @compute @workgroup_size(1) fn main() { _ = u.value; }";
        assert_pipeline_ok(&test, source, None, &[], None);

        let matching_bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Compute, 16)],
        );
        let matching_layout = create_pipeline_layout(test.device(), &[matching_bgl]);
        assert_pipeline_ok(&test, source, None, &[], Some(matching_layout));

        let empty_bgl = create_bind_group_layout(test.device(), &[]);
        let empty_layout = create_pipeline_layout(test.device(), &[empty_bgl]);
        assert_pipeline_error(&test, source, None, &[], Some(empty_layout));

        let wrong_type_bgl = create_bind_group_layout(test.device(), &[sampler_layout(0)]);
        let wrong_type_layout = create_pipeline_layout(test.device(), &[wrong_type_bgl]);
        assert_pipeline_error(&test, source, None, &[], Some(wrong_type_layout));

        let missing_visibility_bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Fragment, 16)],
        );
        let missing_visibility_layout =
            create_pipeline_layout(test.device(), &[missing_visibility_bgl]);
        assert_pipeline_error(&test, source, None, &[], Some(missing_visibility_layout));

        let too_small_bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Compute, 4)],
        );
        let too_small_layout = create_pipeline_layout(test.device(), &[too_small_bgl]);
        assert_pipeline_error(&test, source, None, &[], Some(too_small_layout));

        yawgpu::wgpuPipelineLayoutRelease(too_small_layout);
        yawgpu::wgpuBindGroupLayoutRelease(too_small_bgl);
        yawgpu::wgpuPipelineLayoutRelease(missing_visibility_layout);
        yawgpu::wgpuBindGroupLayoutRelease(missing_visibility_bgl);
        yawgpu::wgpuPipelineLayoutRelease(wrong_type_layout);
        yawgpu::wgpuBindGroupLayoutRelease(wrong_type_bgl);
        yawgpu::wgpuPipelineLayoutRelease(empty_layout);
        yawgpu::wgpuBindGroupLayoutRelease(empty_bgl);
        yawgpu::wgpuPipelineLayoutRelease(matching_layout);
        yawgpu::wgpuBindGroupLayoutRelease(matching_bgl);

        let texture_source = "@group(0) @binding(0) var tex: texture_2d<u32>;
             @compute @workgroup_size(1) fn main() {
                 _ = textureLoad(tex, vec2i(0), 0);
             }";
        let wrong_texture_bgl = create_bind_group_layout(
            test.device(),
            &[texture_layout(
                0,
                native::WGPUTextureSampleType_Float,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        let wrong_texture_layout = create_pipeline_layout(test.device(), &[wrong_texture_bgl]);
        assert_pipeline_error(&test, texture_source, None, &[], Some(wrong_texture_layout));

        let storage_source =
            "@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
             @compute @workgroup_size(1) fn main() {
                 textureStore(image, vec2i(0), vec4f());
             }";
        let wrong_storage_bgl = create_bind_group_layout(
            test.device(),
            &[storage_texture_layout(
                0,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Sint,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        let wrong_storage_layout = create_pipeline_layout(test.device(), &[wrong_storage_bgl]);
        assert_pipeline_error(&test, storage_source, None, &[], Some(wrong_storage_layout));

        yawgpu::wgpuPipelineLayoutRelease(wrong_storage_layout);
        yawgpu::wgpuBindGroupLayoutRelease(wrong_storage_bgl);
        yawgpu::wgpuPipelineLayoutRelease(wrong_texture_layout);
        yawgpu::wgpuBindGroupLayoutRelease(wrong_texture_bgl);
    }
}

#[test]
fn compute_pipeline_release_is_safe_for_valid_and_error_pipelines() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_pipeline(
            &test,
            "@compute @workgroup_size(1) fn main() {}",
            None,
            &[],
            None,
        );
        yawgpu::wgpuComputePipelineAddRef(pipeline);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuComputePipelineRelease(pipeline);

        let mut error_pipeline = std::ptr::null();
        assert_device_error!({
            error_pipeline = create_pipeline(
                &test,
                "@compute @workgroup_size(1) fn main() {}",
                Some("missing"),
                &[],
                None,
            );
        });
        assert!(!error_pipeline.is_null());
        yawgpu::wgpuComputePipelineRelease(error_pipeline);
    }
}

unsafe fn assert_pipeline_ok(
    test: &ValidationTest,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) {
    test.clear_errors();
    let pipeline = create_pipeline(test, source, entry_point, constants, layout);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuComputePipelineRelease(pipeline);
}

unsafe fn assert_pipeline_error(
    test: &ValidationTest,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) {
    let mut pipeline = std::ptr::null();
    assert_device_error!({
        pipeline = create_pipeline(test, source, entry_point, constants, layout);
    });
    assert!(!pipeline.is_null());
    yawgpu::wgpuComputePipelineRelease(pipeline);
}

unsafe fn create_pipeline(
    test: &ValidationTest,
    source: &str,
    entry_point: Option<&str>,
    constants: &[PipelineConstantInput<'_>],
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(test.device(), source);
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
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
    yawgpu::wgpuShaderModuleRelease(module);
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

fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut limits),
            native::WGPUStatus_Success
        );
        limits
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

fn sampler_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, native::WGPUShaderStage_Compute);
    entry.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    entry
}

fn texture_layout(
    binding: u32,
    sample_type: native::WGPUTextureSampleType,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, native::WGPUShaderStage_Compute);
    entry.texture.sampleType = sample_type;
    entry.texture.viewDimension = view_dimension;
    entry
}

fn storage_texture_layout(
    binding: u32,
    access: native::WGPUStorageTextureAccess,
    format: native::WGPUTextureFormat,
    view_dimension: native::WGPUTextureViewDimension,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_layout(binding, native::WGPUShaderStage_Compute);
    entry.storageTexture.access = access;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = view_dimension;
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
