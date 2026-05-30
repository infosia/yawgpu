//! CTS: src/webgpu/api/validation/render_pass/render_pass_descriptor.spec.ts
//!
//! The `bindTextureResource` subcases are N/A for the C ABI: render pass
//! attachments take `WGPUTextureView` handles, not texture resources.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn attachments_one_color_attachment() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let attachment = color_attachment(color.view);
        expect_render_pass(&test, true, &render_pass_descriptor(&[attachment], None));
        release_view(color);
    }
}

#[test]
fn attachments_one_depth_stencil_attachment() {
    let test = ValidationTest::new();
    unsafe {
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        let attachment = depth_stencil_attachment(depth.view);
        expect_render_pass(&test, true, &render_pass_descriptor(&[], Some(&attachment)));
        release_view(depth);
    }
}

#[test]
fn color_attachments_empty() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        let sparse = [sparse_color_attachment(), sparse_color_attachment()];
        let color_attachment = color_attachment(color.view);
        let depth_attachment = depth_stencil_attachment(depth.view);

        expect_render_pass(&test, false, &render_pass_descriptor(&[], None));
        expect_render_pass(&test, false, &render_pass_descriptor(&sparse, None));
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(&sparse, Some(&depth_attachment)),
        );
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(&[color_attachment], None),
        );

        release_view(depth);
        release_view(color);
    }
}

#[test]
fn color_attachments_limits_max_color_attachments() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );
        let color = create_view(test.device(), TextureOptions::color(), None);
        let valid = (0..limits.maxColorAttachments)
            .map(|_| color_attachment(color.view))
            .collect::<Vec<_>>();
        let invalid = (0..=limits.maxColorAttachments)
            .map(|_| color_attachment(color.view))
            .collect::<Vec<_>>();
        expect_render_pass(&test, true, &render_pass_descriptor(&valid, None));
        expect_render_pass(&test, false, &render_pass_descriptor(&invalid, None));
        release_view(color);
    }
}

#[test]
fn color_attachments_limits_max_color_attachment_bytes_per_sample_aligned() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_RGBA32Float,
                ..TextureOptions::color()
            },
            None,
        );
        let attachments = [
            color_attachment(color.view),
            color_attachment(color.view),
            color_attachment(color.view),
        ];
        expect_render_pass(&test, false, &render_pass_descriptor(&attachments, None));
        release_view(color);
    }
}

#[test]
fn color_attachments_limits_max_color_attachment_bytes_per_sample_unaligned() {
    let test = ValidationTest::new();
    unsafe {
        let r8 = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_R8Unorm,
                ..TextureOptions::color()
            },
            None,
        );
        let r32 = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_R32Float,
                ..TextureOptions::color()
            },
            None,
        );
        let rgba32 = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_RGBA32Float,
                ..TextureOptions::color()
            },
            None,
        );
        let attachments = [
            color_attachment(r8.view),
            color_attachment(r32.view),
            color_attachment(rgba32.view),
            color_attachment(rgba32.view),
            color_attachment(r8.view),
        ];
        expect_render_pass(&test, false, &render_pass_descriptor(&attachments, None));
        release_view(rgba32);
        release_view(r32);
        release_view(r8);
    }
}

#[test]
fn color_attachments_depth_slice_definedness() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let mut attachment = color_attachment(color.view);
        attachment.depthSlice = 0;
        expect_render_pass(&test, false, &render_pass_descriptor(&[attachment], None));
        release_view(color);
    }
}

#[test]
fn color_attachments_depth_slice_bound_check() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_view(
            test.device(),
            TextureOptions {
                dimension: native::WGPUTextureDimension_3D,
                depth_or_array_layers: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let mut attachment = color_attachment(texture.view);
        attachment.depthSlice = 4;
        expect_render_pass(&test, false, &render_pass_descriptor(&[attachment], None));
        release_view(texture);
    }
}

#[test]
fn color_attachments_depth_slice_overlaps_same_miplevel() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_view(
            test.device(),
            TextureOptions {
                dimension: native::WGPUTextureDimension_3D,
                depth_or_array_layers: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let mut a = color_attachment(texture.view);
        let mut b = color_attachment(texture.view);
        a.depthSlice = 0;
        b.depthSlice = 0;
        expect_render_pass(&test, false, &render_pass_descriptor(&[a, b], None));
        release_view(texture);
    }
}

#[test]
fn color_attachments_depth_slice_overlaps_diff_miplevel() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            TextureOptions {
                dimension: native::WGPUTextureDimension_3D,
                depth_or_array_layers: 4,
                mip_level_count: 2,
                ..TextureOptions::color()
            },
        );
        let view0 = yawgpu::wgpuTextureCreateView(
            texture,
            &view_descriptor(
                native::WGPUTextureViewDimension_3D,
                0,
                1,
                0,
                1,
                native::WGPUTextureFormat_Undefined,
            ),
        );
        let view1 = yawgpu::wgpuTextureCreateView(
            texture,
            &view_descriptor(
                native::WGPUTextureViewDimension_3D,
                0,
                1,
                0,
                1,
                native::WGPUTextureFormat_Undefined,
            ),
        );
        let mut a = color_attachment(view0);
        let mut b = color_attachment(view1);
        a.depthSlice = 0;
        b.depthSlice = 0;
        expect_render_pass(&test, false, &render_pass_descriptor(&[a, b], None));
        yawgpu::wgpuTextureViewRelease(view1);
        yawgpu::wgpuTextureViewRelease(view0);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn attachments_same_size() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let color_same = create_view(test.device(), TextureOptions::color(), None);
        let color_large = create_view(
            test.device(),
            TextureOptions {
                width: 32,
                height: 32,
                ..TextureOptions::color()
            },
            None,
        );
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        let depth_large = create_view(
            test.device(),
            TextureOptions {
                width: 32,
                height: 32,
                ..TextureOptions::depth_stencil()
            },
            None,
        );

        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(
                &[
                    color_attachment(color.view),
                    color_attachment(color_same.view),
                ],
                Some(&depth_stencil_attachment(depth.view)),
            ),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[
                    color_attachment(color.view),
                    color_attachment(color_large.view),
                ],
                None,
            ),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment(color.view)],
                Some(&depth_stencil_attachment(depth_large.view)),
            ),
        );

        release_view(depth_large);
        release_view(depth);
        release_view(color_large);
        release_view(color_same);
        release_view(color);
    }
}

#[test]
fn attachments_color_depth_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(&[color_attachment(depth.view)], None),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(&[], Some(&depth_stencil_attachment(color.view))),
        );
        release_view(depth);
        release_view(color);
    }
}

#[test]
fn attachments_layer_count() {
    let test = ValidationTest::new();
    unsafe {
        for (array_layer_count, base_array_layer, success) in
            [(2, 0, false), (1, 0, true), (1, 9, true)]
        {
            let color = create_view(
                test.device(),
                TextureOptions {
                    depth_or_array_layers: 10,
                    ..TextureOptions::color()
                },
                Some(view_descriptor(
                    native::WGPUTextureViewDimension_2DArray,
                    0,
                    1,
                    base_array_layer,
                    array_layer_count,
                    native::WGPUTextureFormat_Undefined,
                )),
            );
            expect_render_pass(
                &test,
                success,
                &render_pass_descriptor(&[color_attachment(color.view)], None),
            );
            release_view(color);
        }
    }
}

#[test]
fn attachments_mip_level_count() {
    let test = ValidationTest::new();
    unsafe {
        for (mip_level_count, base_mip_level, success) in
            [(2, 0, false), (1, 0, true), (1, 3, true)]
        {
            let color = create_view(
                test.device(),
                TextureOptions {
                    mip_level_count: 4,
                    ..TextureOptions::color()
                },
                Some(view_descriptor(
                    native::WGPUTextureViewDimension_2D,
                    base_mip_level,
                    mip_level_count,
                    0,
                    1,
                    native::WGPUTextureFormat_Undefined,
                )),
            );
            expect_render_pass(
                &test,
                success,
                &render_pass_descriptor(&[color_attachment(color.view)], None),
            );
            release_view(color);
        }
    }
}

#[test]
fn color_attachments_load_op_store_op() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_TransientAttachment,
                ..TextureOptions::color()
            },
            None,
        );
        let mut attachment = color_attachment(color.view);
        attachment.loadOp = native::WGPULoadOp_Load;
        attachment.storeOp = native::WGPUStoreOp_Store;
        expect_render_pass(&test, false, &render_pass_descriptor(&[attachment], None));
        release_view(color);
    }
}

#[test]
fn color_attachments_non_multisampled() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let resolve = create_view(test.device(), TextureOptions::color(), None);
        let attachment = color_attachment_with_resolve(color.view, resolve.view);
        expect_render_pass(&test, false, &render_pass_descriptor(&[attachment], None));
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn color_attachments_sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let msaa = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(&[color_attachment(msaa.view)], None),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment(color.view), color_attachment(msaa.view)],
                None,
            ),
        );
        release_view(msaa);
        release_view(color);
    }
}

#[test]
fn resolve_target_sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn resolve_target_array_layer_count() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(
            test.device(),
            TextureOptions {
                depth_or_array_layers: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2DArray,
                0,
                1,
                0,
                2,
                native::WGPUTextureFormat_Undefined,
            )),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn resolve_target_mipmap_level_count() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(
            test.device(),
            TextureOptions {
                mip_level_count: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                0,
                2,
                0,
                1,
                native::WGPUTextureFormat_Undefined,
            )),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn resolve_target_usage() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        for (usage, success) in [
            (native::WGPUTextureUsage_CopyDst, false),
            (
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_TextureBinding,
                true,
            ),
        ] {
            let resolve = create_view(
                test.device(),
                TextureOptions {
                    usage,
                    ..TextureOptions::color()
                },
                None,
            );
            expect_render_pass(
                &test,
                success,
                &render_pass_descriptor(
                    &[color_attachment_with_resolve(color.view, resolve.view)],
                    None,
                ),
            );
            release_view(resolve);
        }
        release_view(color);
    }
}

#[test]
fn resolve_target_error_state() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve_texture = create_texture(test.device(), TextureOptions::color());
        let mut error_view = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_view = yawgpu::wgpuTextureCreateView(
                    resolve_texture,
                    &view_descriptor(
                        native::WGPUTextureViewDimension_2D,
                        0,
                        1,
                        2,
                        1,
                        native::WGPUTextureFormat_Undefined,
                    ),
                );
            },
            None,
        );
        assert!(!error_view.is_null());
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, error_view)],
                None,
            ),
        );
        yawgpu::wgpuTextureViewRelease(error_view);
        yawgpu::wgpuTextureRelease(resolve_texture);
        release_view(color);
    }
}

#[test]
fn resolve_target_single_sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(test.device(), TextureOptions::color(), None);
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn resolve_target_different_format() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_BGRA8Unorm,
                ..TextureOptions::color()
            },
            None,
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn resolve_target_different_size() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve_large = create_view(
            test.device(),
            TextureOptions {
                width: 32,
                height: 32,
                mip_level_count: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                0,
                1,
                0,
                1,
                native::WGPUTextureFormat_Undefined,
            )),
        );
        let resolve_mip = create_view(
            test.device(),
            TextureOptions {
                width: 32,
                height: 32,
                mip_level_count: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                1,
                1,
                0,
                1,
                native::WGPUTextureFormat_Undefined,
            )),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(
                    color.view,
                    resolve_large.view,
                )],
                None,
            ),
        );
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve_mip.view)],
                None,
            ),
        );
        release_view(resolve_mip);
        release_view(resolve_large);
        release_view(color);
    }
}

#[test]
fn depth_stencil_attachment_sample_counts_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let color_msaa = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let color_single = create_view(test.device(), TextureOptions::color(), None);
        let depth_msaa = create_view(
            test.device(),
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::depth_stencil()
            },
            None,
        );
        let depth_single = create_view(test.device(), TextureOptions::depth_stencil(), None);
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment(color_msaa.view)],
                Some(&depth_stencil_attachment(depth_single.view)),
            ),
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment(color_single.view)],
                Some(&depth_stencil_attachment(depth_msaa.view)),
            ),
        );
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(
                &[color_attachment(color_msaa.view)],
                Some(&depth_stencil_attachment(depth_msaa.view)),
            ),
        );
        expect_render_pass(
            &test,
            true,
            &render_pass_descriptor(&[], Some(&depth_stencil_attachment(depth_msaa.view))),
        );
        release_view(depth_single);
        release_view(depth_msaa);
        release_view(color_single);
        release_view(color_msaa);
    }
}

#[test]
fn depth_stencil_attachment_load_op_store_op_match_depth_read_only_stencil_read_only() {
    let test = ValidationTest::new();
    unsafe {
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        let mut attachment = depth_stencil_attachment(depth.view);
        attachment.depthReadOnly = 1;
        attachment.depthLoadOp = native::WGPULoadOp_Clear;
        attachment.depthStoreOp = native::WGPUStoreOp_Store;
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(&[], Some(&attachment)),
        );
        release_view(depth);
    }
}

#[test]
fn depth_stencil_attachment_depth_clear_value() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let depth = create_view(test.device(), TextureOptions::depth_stencil(), None);
        for (load_op, clear_value, success) in [
            (native::WGPULoadOp_Load, -1.0, true),
            (native::WGPULoadOp_Clear, 0.0, true),
            (native::WGPULoadOp_Clear, 1.0, true),
            (native::WGPULoadOp_Clear, -1.0, false),
            (native::WGPULoadOp_Clear, 1.5, false),
        ] {
            let mut attachment = depth_stencil_attachment(depth.view);
            attachment.depthLoadOp = load_op;
            attachment.depthClearValue = clear_value;
            expect_render_pass(
                &test,
                success,
                &render_pass_descriptor(&[color_attachment(color.view)], Some(&attachment)),
            );
        }
        release_view(depth);
        release_view(color);
    }
}

#[test]
fn resolve_target_format_supports_resolve() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_R8Uint,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        let resolve = create_view(
            test.device(),
            TextureOptions {
                format: native::WGPUTextureFormat_R8Uint,
                ..TextureOptions::color()
            },
            None,
        );
        expect_render_pass(
            &test,
            false,
            &render_pass_descriptor(
                &[color_attachment_with_resolve(color.view, resolve.view)],
                None,
            ),
        );
        release_view(resolve);
        release_view(color);
    }
}

#[test]
fn timestamp_writes_query_set_type() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        for (query_type, success) in [
            (native::WGPUQueryType_Timestamp, true),
            (native::WGPUQueryType_Occlusion, false),
        ] {
            let query_set = create_query_set(test.device(), query_type, 2);
            let writes = timestamp_writes(query_set, 0, 1);
            let mut descriptor = render_pass_descriptor(&[color_attachment(color.view)], None);
            descriptor.timestampWrites = &writes;
            expect_render_pass(&test, success, &descriptor);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        release_view(color);
    }
}

#[test]
fn timestamp_write_query_index() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 2);
        for (beginning, end, success) in [
            (0, 1, true),
            (native::WGPU_QUERY_SET_INDEX_UNDEFINED, 0, true),
            (0, native::WGPU_QUERY_SET_INDEX_UNDEFINED, true),
            (2, 1, false),
            (0, 2, false),
            (0, 0, false),
            (
                native::WGPU_QUERY_SET_INDEX_UNDEFINED,
                native::WGPU_QUERY_SET_INDEX_UNDEFINED,
                false,
            ),
        ] {
            let writes = timestamp_writes(query_set, beginning, end);
            let mut descriptor = render_pass_descriptor(&[color_attachment(color.view)], None);
            descriptor.timestampWrites = &writes;
            expect_render_pass(&test, success, &descriptor);
        }
        yawgpu::wgpuQuerySetRelease(query_set);
        release_view(color);
    }
}

#[test]
fn occlusion_query_set_query_set_type() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        let color = create_view(test.device(), TextureOptions::color(), None);
        for (query_type, success) in [
            (native::WGPUQueryType_Occlusion, true),
            (native::WGPUQueryType_Timestamp, false),
        ] {
            let query_set = create_query_set(test.device(), query_type, 1);
            let mut descriptor = render_pass_descriptor(&[color_attachment(color.view)], None);
            descriptor.occlusionQuerySet = query_set;
            expect_render_pass(&test, success, &descriptor);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        release_view(color);
    }
}
