//! CTS: src/webgpu/api/validation/createPipelineLayout.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::bind_group_common::{
    buffer_layout, create_pipeline_layout_raw, dynamic_buffer_layout, expect_bind_group_layout,
    expect_pipeline_layout, release_bind_group_layouts, release_pipeline_layout,
};
use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, create_pipeline_layout,
    device_limits, request_device,
};

#[test]
#[ignore = "core does not yet aggregate dynamic buffer counts across bind group layouts at createPipelineLayout; CTS expects exceeding dynamic buffer limits across layouts to fail"]
fn number_of_dynamic_buffers_exceeds_the_maximum_value() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let maxed_entries = (0..limits.maxDynamicUniformBuffersPerPipelineLayout)
            .map(|binding| {
                dynamic_buffer_layout(
                    binding,
                    native::WGPUShaderStage_Compute,
                    native::WGPUBufferBindingType_Uniform,
                )
            })
            .collect::<Vec<_>>();
        let maxed = expect_bind_group_layout(&test, true, &maxed_entries);
        let extra = expect_bind_group_layout(
            &test,
            true,
            &[dynamic_buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let pipeline_layout = expect_pipeline_layout(&test, false, &[maxed, extra], 0);
        release_pipeline_layout(pipeline_layout);
        release_bind_group_layouts(&[extra, maxed]);
    }
}

#[test]
fn number_of_bind_group_layouts_exceeds_the_maximum_value() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let layouts = (0..limits.maxBindGroups)
            .map(|_| expect_bind_group_layout(&test, true, &[]))
            .collect::<Vec<_>>();
        let pipeline_layout = expect_pipeline_layout(&test, true, &layouts, 0);
        release_pipeline_layout(pipeline_layout);

        let mut too_many = layouts.clone();
        too_many.push(expect_bind_group_layout(&test, true, &[]));
        let pipeline_layout = expect_pipeline_layout(&test, false, &too_many, 0);
        release_pipeline_layout(pipeline_layout);
        release_bind_group_layouts(&too_many);
    }
}

#[test]
fn bind_group_layouts_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let own0 = expect_bind_group_layout(&test, true, &[]);
        let own1 = expect_bind_group_layout(&test, true, &[]);
        let other0 = create_empty_bind_group_layout(other);
        let other1 = create_empty_bind_group_layout(other);

        for (layouts, success) in [
            (vec![own0, own1], true),
            (vec![other0, own1], false),
            (vec![own0, other1], false),
        ] {
            let pipeline_layout = expect_pipeline_layout(&test, success, &layouts, 0);
            release_pipeline_layout(pipeline_layout);
        }

        yawgpu::wgpuBindGroupLayoutRelease(other1);
        yawgpu::wgpuBindGroupLayoutRelease(other0);
        yawgpu::wgpuBindGroupLayoutRelease(own1);
        yawgpu::wgpuBindGroupLayoutRelease(own0);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
#[ignore = "core currently rejects null bind group layout slots; CTS expects sparse/null pipeline layout slots to be valid"]
fn bind_group_layouts_null_bind_group_layouts() {
    let test = ValidationTest::new();
    unsafe {
        let non_empty = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let slots = [non_empty, std::ptr::null()];
        let mut layout = std::ptr::null();
        test.assert_device_error_after(
            || {
                layout = create_pipeline_layout_raw(test.device(), slots.len(), slots.as_ptr(), 0);
            },
            None,
        );
        assert!(!layout.is_null());
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(non_empty);
    }
}

#[test]
#[ignore = "core currently rejects null bind group layout slots; CTS expects pipelines to validate against sparse pipeline layouts"]
fn bind_group_layouts_create_pipeline_with_null_bind_group_layouts() {
    let test = ValidationTest::new();
    unsafe {
        let non_empty = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let slots = [std::ptr::null(), non_empty];
        let layout = create_pipeline_layout_raw(test.device(), slots.len(), slots.as_ptr(), 0);
        assert_compute_pipeline_error(
            &test,
            false,
            "@group(1) @binding(0) var<uniform> u: u32; @compute @workgroup_size(1) fn main() { _ = u; }",
            Some("main"),
            &[],
            Some(layout),
        );
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(non_empty);
    }
}

#[test]
#[ignore = "core currently rejects null bind group layout slots; CTS expects setting pipelines with sparse pipeline layouts to be valid"]
fn bind_group_layouts_set_pipeline_with_null_bind_group_layouts() {
    let test = ValidationTest::new();
    unsafe {
        let non_empty = expect_bind_group_layout(
            &test,
            true,
            &[buffer_layout(
                0,
                native::WGPUShaderStage_Compute,
                native::WGPUBufferBindingType_Uniform,
            )],
        );
        let slots = [std::ptr::null(), non_empty];
        let layout = create_pipeline_layout_raw(test.device(), slots.len(), slots.as_ptr(), 0);
        assert!(!layout.is_null());
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(non_empty);
    }
}

#[test]
#[ignore = "core validates immediateSize range but not the required multiple-of-4 alignment; CTS expects unaligned immediateSize values to fail"]
fn immediate_data_size() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (size, success) in [
            (0, true),
            (4, true),
            (limits.maxImmediateSize, true),
            (1, false),
            (5, false),
            (limits.maxImmediateSize + 4, false),
        ] {
            let pipeline_layout = expect_pipeline_layout(&test, success, &[], size);
            release_pipeline_layout(pipeline_layout);
        }
    }
}

unsafe fn create_empty_bind_group_layout(
    device: native::WGPUDevice,
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: crate::common::empty_string_view(),
        entryCount: 0,
        entries: std::ptr::null(),
    };
    unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) }
}

#[allow(dead_code)]
unsafe fn create_valid_compute_pipeline_with_layout(
    test: &ValidationTest,
    layout: native::WGPUPipelineLayout,
) {
    unsafe {
        assert_compute_pipeline_ok(
            test,
            false,
            "@compute @workgroup_size(1) fn main() {}",
            Some("main"),
            &[],
            Some(layout),
        );
    }
}

#[allow(dead_code)]
unsafe fn create_empty_pipeline_layout(test: &ValidationTest) -> native::WGPUPipelineLayout {
    unsafe { create_pipeline_layout(test.device(), &[], 0) }
}
