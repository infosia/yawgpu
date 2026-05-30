use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, constant, create_bind_group_layout,
    create_compute_pipeline_with_module, create_pipeline_layout, create_wgsl_module, device_limits,
    request_device, storage_texture_layout, uniform_layout,
};

#[test]
fn basic() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                None,
            );
        }
    }
}

#[test]
fn shader_module_invalid() {
    let test = ValidationTest::new();
    unsafe {
        let mut module = std::ptr::null();
        assert_device_error!({
            module = create_wgsl_module(test.device(), "not wgsl @@@");
        });
        assert!(!module.is_null());
        for is_async in [false, true] {
            if is_async {
                let pipeline = create_compute_pipeline_with_module(
                    test.instance(),
                    test.device(),
                    true,
                    module,
                    Some("main"),
                    &[],
                    None,
                );
                assert!(pipeline.is_null());
            } else {
                let mut pipeline = std::ptr::null();
                assert_device_error!({
                    pipeline = create_compute_pipeline_with_module(
                        test.instance(),
                        test.device(),
                        false,
                        module,
                        Some("main"),
                        &[],
                        None,
                    );
                });
                assert!(!pipeline.is_null());
                yawgpu::wgpuComputePipelineRelease(pipeline);
            }
        }
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn shader_module_compute() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }",
                Some("main"),
                &[],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@fragment fn main() -> @location(0) vec4f { return vec4f(); }",
                Some("main"),
                &[],
                None,
            );
        }
    }
}

#[test]
fn shader_module_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let own_module =
            create_wgsl_module(test.device(), "@compute @workgroup_size(1) fn main() {}");
        let other_module = create_wgsl_module(other, "@compute @workgroup_size(1) fn main() {}");
        for is_async in [false, true] {
            let pipeline = create_compute_pipeline_with_module(
                test.instance(),
                test.device(),
                is_async,
                own_module,
                Some("main"),
                &[],
                None,
            );
            assert!(!pipeline.is_null());
            yawgpu::wgpuComputePipelineRelease(pipeline);

            if is_async {
                let pipeline = create_compute_pipeline_with_module(
                    test.instance(),
                    test.device(),
                    true,
                    other_module,
                    Some("main"),
                    &[],
                    None,
                );
                assert!(pipeline.is_null());
            } else {
                let mut pipeline = std::ptr::null();
                assert_device_error!({
                    pipeline = create_compute_pipeline_with_module(
                        test.instance(),
                        test.device(),
                        false,
                        other_module,
                        Some("main"),
                        &[],
                        None,
                    );
                });
                assert!(!pipeline.is_null());
                yawgpu::wgpuComputePipelineRelease(pipeline);
            }
        }
        yawgpu::wgpuShaderModuleRelease(other_module);
        yawgpu::wgpuShaderModuleRelease(own_module);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn pipeline_layout_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let own_layout = create_pipeline_layout(test.device(), &[], 0);
        let other_layout = create_pipeline_layout(other, &[], 0);
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                Some(own_layout),
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                Some(other_layout),
            );
        }
        yawgpu::wgpuPipelineLayoutRelease(other_layout);
        yawgpu::wgpuPipelineLayoutRelease(own_layout);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn limits_workgroup_storage_size() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (ty, size) in [("vec4<f32>", 16), ("mat4x4<f32>", 64)] {
            let count_at_limit = limits.maxComputeWorkgroupStorageSize / size;
            for is_async in [false, true] {
                assert_compute_pipeline_ok(
                    &test,
                    is_async,
                    &format!(
                        "var<workgroup> data: array<{ty}, {count_at_limit}>;
                         @compute @workgroup_size(64) fn main() {{ _ = data; }}"
                    ),
                    Some("main"),
                    &[],
                    None,
                );
                assert_compute_pipeline_error(
                    &test,
                    is_async,
                    &format!(
                        "var<workgroup> data: array<{ty}, {}>;
                         @compute @workgroup_size(64) fn main() {{ _ = data; }}",
                        count_at_limit + 1
                    ),
                    Some("main"),
                    &[],
                    None,
                );
            }
        }
    }
}

#[test]
fn limits_invocations_per_workgroup() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let valid_x = limits
            .maxComputeInvocationsPerWorkgroup
            .min(limits.maxComputeWorkgroupSizeX);
        let invalid_y = limits.maxComputeInvocationsPerWorkgroup / valid_x + 1;
        for is_async in [false, true] {
            expect_pipeline(
                &test,
                is_async,
                true,
                &format!("@compute @workgroup_size({valid_x}, 1, 1) fn main() {{}}"),
                &[],
            );
            expect_pipeline(
                &test,
                is_async,
                false,
                &format!("@compute @workgroup_size({valid_x}, {invalid_y}, 1) fn main() {{}}"),
                &[],
            );
        }
    }
}

#[test]
fn limits_invocations_per_workgroup_each_component() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let cases = [
            (limits.maxComputeWorkgroupSizeX, 1, 1, true),
            (limits.maxComputeWorkgroupSizeX + 1, 1, 1, false),
            (1, limits.maxComputeWorkgroupSizeY, 1, true),
            (1, limits.maxComputeWorkgroupSizeY + 1, 1, false),
            (1, 1, limits.maxComputeWorkgroupSizeZ, true),
            (1, 1, limits.maxComputeWorkgroupSizeZ + 1, false),
        ];
        for (x, y, z, success) in cases {
            for is_async in [false, true] {
                expect_pipeline(
                    &test,
                    is_async,
                    success,
                    &format!("@compute @workgroup_size({x}, {y}, {z}) fn main() {{}}"),
                    &[],
                );
            }
        }
    }
}

#[test]
fn overrides_identifier() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
override c0: bool = true;
override c1: u32 = 0u;
override 数: u32 = 0u;
override séquençage: u32 = 0u;
@id(1000) override c2: u32 = 10u;
@id(1) override c3: u32 = 11u;
@compute @workgroup_size(1) fn main() {
  _ = u32(c0);
  _ = c1 + c2 + c3 + 数 + séquençage;
}";
        let cases = [
            (vec![], true),
            (vec![constant("c0", 0.0)], true),
            (vec![constant("c9", 0.0)], false),
            (vec![constant("1", 0.0)], true),
            (vec![constant("c3", 0.0)], false),
            (vec![constant("1000", 0.0)], true),
            (vec![constant("9999", 0.0)], false),
            (vec![constant("数", 0.0)], true),
            (vec![constant("séquençage", 0.0)], false),
        ];
        for (constants, success) in cases {
            for is_async in [false, true] {
                expect_pipeline(&test, is_async, success, source, &constants);
            }
        }
    }
}

#[test]
fn overrides_uninitialized() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
override c0: bool;
override c1: bool = false;
override c2: f32;
override c5: i32;
override c8: u32;
@compute @workgroup_size(1) fn main() {
  _ = u32(c0) + u32(c1) + u32(c2) + u32(c5) + c8;
}";
        let all = [
            constant("c0", 0.0),
            constant("c2", 0.0),
            constant("c5", 0.0),
            constant("c8", 0.0),
        ];
        for is_async in [false, true] {
            assert_compute_pipeline_error(&test, is_async, source, Some("main"), &[], None);
            assert_compute_pipeline_error(&test, is_async, source, Some("main"), &all[..3], None);
            assert_compute_pipeline_ok(&test, is_async, source, Some("main"), &all, None);
        }
    }
}

#[test]
fn overrides_value_type_error() {
    let test = ValidationTest::new();
    unsafe {
        let source = "override cf: f32 = 0.0; @compute @workgroup_size(1) fn main() { _ = cf; }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("cf", 1.0)],
                None,
            );
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_compute_pipeline_error(
                    &test,
                    is_async,
                    source,
                    Some("main"),
                    &[constant("cf", value)],
                    None,
                );
            }
        }
    }
}

#[test]
fn overrides_value_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
override cb: bool = false;
override cu: u32 = 0u;
override ci: i32 = 0;
override cf: f32 = 0.0;
@compute @workgroup_size(1) fn main() { _ = cb; _ = cu; _ = ci; _ = cf; }";
        let cases = [
            (constant("cu", 0.0), true),
            (constant("cu", -1.0), false),
            (constant("cu", f64::from(u32::MAX)), true),
            (constant("cu", f64::from(u32::MAX) + 1.0), false),
            (constant("ci", f64::from(i32::MIN)), true),
            (constant("ci", f64::from(i32::MIN) - 1.0), false),
            (constant("ci", f64::from(i32::MAX)), true),
            (constant("ci", f64::from(i32::MAX) + 1.0), false),
            (constant("cf", f64::from(f32::MAX)), true),
            (constant("cf", f64::MAX), false),
            (constant("cb", f64::MAX), true),
        ];
        for (constant, success) in cases {
            for is_async in [false, true] {
                expect_pipeline(&test, is_async, success, source, &[constant]);
            }
        }
    }
}

#[test]
fn overrides_entry_point_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
override cu: u32 = 0u;
override cx: u32 = 1u / cu;
@compute @workgroup_size(1) fn main_success() { _ = cu; }
@compute @workgroup_size(1) fn main_pipe_error() { _ = cx; }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(&test, is_async, source, Some("main_success"), &[], None);
            assert_compute_pipeline_error(
                &test,
                is_async,
                source,
                Some("main_pipe_error"),
                &[],
                None,
            );
        }
    }
}

#[test]
fn overrides_value_validation_error_f16() {
    unsafe {
        let feature = native::WGPUFeatureName_ShaderF16;
        let probe = ValidationTest::new();
        if yawgpu::wgpuAdapterHasFeature(probe.adapter(), feature) == 0 {
            return;
        }
        drop(probe);

        let test = ValidationTest::with_features(&[feature]);
        let source = "
enable f16;
override cf16: f16 = 0.0h;
@compute @workgroup_size(1) fn main() { _ = cf16; }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("cf16", 65504.0)],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("cf16", 65_505.0)],
                None,
            );
        }
    }
}

#[test]
fn overrides_workgroup_size() {
    let test = ValidationTest::new();
    unsafe {
        for ty in ["u32", "i32"] {
            let suffix = if ty == "u32" { "u" } else { "" };
            let source = format!(
                "override x: {ty} = 1{suffix};
                 override y: {ty} = 1{suffix};
                 override z: {ty} = 1{suffix};
                 @compute @workgroup_size(x, y, z) fn main() {{}}"
            );
            let cases = [
                (vec![], true),
                (
                    vec![constant("x", 0.0), constant("y", 0.0), constant("z", 0.0)],
                    false,
                ),
                (
                    vec![constant("x", 1.0), constant("y", -1.0), constant("z", 1.0)],
                    false,
                ),
                (
                    vec![constant("x", 16.0), constant("y", 1.0), constant("z", 1.0)],
                    true,
                ),
            ];
            for (constants, success) in cases {
                for is_async in [false, true] {
                    expect_pipeline(&test, is_async, success, &source, &constants);
                }
            }
        }
    }
}

#[test]
fn overrides_workgroup_size_limits() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let source = "
override x: u32 = 1u;
override y: u32 = 1u;
override z: u32 = 1u;
@compute @workgroup_size(x, y, z) fn main() {}";
        let cases = [
            (limits.maxComputeWorkgroupSizeX, 1, 1, true),
            (limits.maxComputeWorkgroupSizeX + 1, 1, 1, false),
            (1, limits.maxComputeWorkgroupSizeY, 1, true),
            (1, limits.maxComputeWorkgroupSizeY + 1, 1, false),
            (1, 1, limits.maxComputeWorkgroupSizeZ, true),
            (1, 1, limits.maxComputeWorkgroupSizeZ + 1, false),
        ];
        for (x, y, z, success) in cases {
            let constants = [
                constant("x", f64::from(x)),
                constant("y", f64::from(y)),
                constant("z", f64::from(z)),
            ];
            for is_async in [false, true] {
                expect_pipeline(&test, is_async, success, source, &constants);
            }
        }
    }
}

#[test]
fn overrides_workgroup_size_limits_workgroup_storage_size() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let max_vec4_count = limits.maxComputeWorkgroupStorageSize / 16;
        let source = "
override a: u32;
var<workgroup> vec4_data: array<vec4<f32>, a>;
@compute @workgroup_size(1) fn main() { _ = vec4_data[0]; }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("a", f64::from(max_vec4_count))],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("a", f64::from(max_vec4_count + 1))],
                None,
            );
        }
    }
}

#[test]
fn resource_compatibility() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                vec![uniform_layout(0, native::WGPUShaderStage_Compute, 16)],
                "@group(0) @binding(0) var<uniform> u: vec4<f32>; @compute @workgroup_size(1) fn main() { _ = u; }",
                true,
            ),
            (
                vec![],
                "@group(0) @binding(0) var<uniform> u: vec4<f32>; @compute @workgroup_size(1) fn main() { _ = u; }",
                false,
            ),
            (
                vec![uniform_layout(0, native::WGPUShaderStage_Fragment, 16)],
                "@group(0) @binding(0) var<uniform> u: vec4<f32>; @compute @workgroup_size(1) fn main() { _ = u; }",
                false,
            ),
            (
                vec![sampler_layout(0, native::WGPUShaderStage_Compute)],
                "@group(0) @binding(0) var<uniform> u: vec4<f32>; @compute @workgroup_size(1) fn main() { _ = u; }",
                false,
            ),
        ];
        for (entries, source, success) in cases {
            let bgl = create_bind_group_layout(test.device(), &entries);
            let layout = create_pipeline_layout(test.device(), &[bgl], 0);
            for is_async in [false, true] {
                expect_pipeline_with_layout(&test, is_async, success, source, Some(layout));
            }
            yawgpu::wgpuPipelineLayoutRelease(layout);
            yawgpu::wgpuBindGroupLayoutRelease(bgl);
        }
    }
}

#[test]
fn resource_compatibility_full_matrix_gaps() {
    let test = ValidationTest::new();
    unsafe {
        let bgl = create_bind_group_layout(
            test.device(),
            &[storage_texture_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUStorageTextureAccess_WriteOnly,
                native::WGPUTextureFormat_RGBA8Sint,
                native::WGPUTextureViewDimension_2D,
            )],
        );
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        for is_async in [false, true] {
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
                 @compute @workgroup_size(1) fn main() { textureStore(tex, vec2i(0), vec4f()); }",
                Some("main"),
                &[],
                Some(layout),
            );
        }
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

#[test]
fn storage_texture_format() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn main() { textureStore(tex, vec2i(0), vec4f()); }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(&test, is_async, source, Some("main"), &[], None);
        }

        let explicit_wrong = storage_texture_layout(
            0,
            native::WGPUShaderStage_Compute,
            native::WGPUStorageTextureAccess_WriteOnly,
            native::WGPUTextureFormat_RGBA8Sint,
            native::WGPUTextureViewDimension_2D,
        );
        let bgl = create_bind_group_layout(test.device(), &[explicit_wrong]);
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        for is_async in [false, true] {
            assert_compute_pipeline_error(&test, is_async, source, Some("main"), &[], Some(layout));
        }
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

unsafe fn expect_pipeline(
    test: &ValidationTest,
    is_async: bool,
    success: bool,
    source: &str,
    constants: &[crate::common::PipelineConstantInput<'_>],
) {
    unsafe {
        if success {
            assert_compute_pipeline_ok(test, is_async, source, Some("main"), constants, None);
        } else {
            assert_compute_pipeline_error(test, is_async, source, Some("main"), constants, None);
        }
    }
}

unsafe fn expect_pipeline_with_layout(
    test: &ValidationTest,
    is_async: bool,
    success: bool,
    source: &str,
    layout: Option<native::WGPUPipelineLayout>,
) {
    unsafe {
        if success {
            assert_compute_pipeline_ok(test, is_async, source, Some("main"), &[], layout);
        } else {
            assert_compute_pipeline_error(test, is_async, source, Some("main"), &[], layout);
        }
    }
}

fn sampler_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = native::WGPUBindGroupLayoutEntry {
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
    };
    entry.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    entry
}
