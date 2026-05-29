//! CTS port of `webgpu/api/validation/encoding/queries/general.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_render_pass, color_attachment, create_encoder, create_query_set, create_render_target,
    expect_command_buffer, render_pass_descriptor, CommandExpectation,
};

#[test]
fn occlusion_query_query_type() {
    unsafe {
        let default = ValidationTest::new();
        expect_query_type_case(&default, std::ptr::null(), false);

        let occlusion = create_query_set(default.device(), native::WGPUQueryType_Occlusion, 1);
        expect_query_type_case(&default, occlusion, true);
        yawgpu::wgpuQuerySetRelease(occlusion);

        let timestamp_device =
            ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
        let timestamp = create_query_set(
            timestamp_device.device(),
            native::WGPUQueryType_Timestamp,
            1,
        );
        expect_query_type_case(&timestamp_device, timestamp, false);
        yawgpu::wgpuQuerySetRelease(timestamp);
    }
}

#[test]
fn occlusion_query_invalid_query_set() {
    let test = ValidationTest::new();
    unsafe {
        for valid in [true, false] {
            let query_set = if valid {
                create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1)
            } else {
                create_error_query_set(&test)
            };
            expect_query_type_case(&test, query_set, valid);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn occlusion_query_query_index() {
    let test = ValidationTest::new();
    unsafe {
        for query_index in [0, 2] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
            expect_occlusion_index(&test, query_set, query_index, query_index < 2);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

unsafe fn expect_query_type_case(
    test: &ValidationTest,
    query_set: native::WGPUQuerySet,
    success: bool,
) {
    unsafe {
        expect_occlusion_index(test, query_set, 0, success);
    }
}

unsafe fn expect_occlusion_index(
    test: &ValidationTest,
    query_set: native::WGPUQuerySet,
    query_index: u32,
    success: bool,
) {
    unsafe {
        let encoder = create_encoder(test.device());
        let target = create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
        let attachment = color_attachment(target.view);
        let mut descriptor = render_pass_descriptor(&[attachment], None);
        descriptor.occlusionQuerySet = query_set;
        let pass = begin_render_pass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, query_index);
        yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(
            test,
            encoder,
            if success {
                CommandExpectation::Success
            } else {
                CommandExpectation::FinishError
            },
        );
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuTextureViewRelease(target.view);
        yawgpu::wgpuTextureRelease(target.texture);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn create_error_query_set(test: &ValidationTest) -> native::WGPUQuerySet {
    unsafe {
        let descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: crate::common::empty_string_view(),
            type_: native::WGPUQueryType_Occlusion,
            count: 0,
        };
        let mut query_set = std::ptr::null();
        test.assert_device_error_after(
            || {
                query_set = yawgpu::wgpuDeviceCreateQuerySet(test.device(), &descriptor);
            },
            None,
        );
        assert!(!query_set.is_null());
        query_set
    }
}
