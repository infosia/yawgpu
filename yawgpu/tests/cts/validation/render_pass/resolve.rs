use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    color_attachment_with_resolve, create_view, expect_render_pass, release_view,
    render_pass_descriptor, view_descriptor, TextureOptions,
};

#[test]
fn resolve_attachment() {
    let test = ValidationTest::new();
    unsafe {
        let multisampled_color = TextureOptions {
            sample_count: 4,
            ..TextureOptions::color()
        };
        let single_sampled_color = TextureOptions::color();
        let resolve_default = TextureOptions::color();

        run_case(&test, true, multisampled_color, None, resolve_default, None);
        run_case(
            &test,
            true,
            TextureOptions {
                width: 8,
                height: 8,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
            TextureOptions {
                mip_level_count: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                1,
                1,
                0,
                1,
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            true,
            multisampled_color,
            None,
            TextureOptions {
                depth_or_array_layers: 2,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                0,
                1,
                1,
                1,
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            true,
            TextureOptions {
                format: native::WGPUTextureFormat_BGRA8Unorm,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
            TextureOptions {
                format: native::WGPUTextureFormat_BGRA8Unorm,
                ..TextureOptions::color()
            },
            None,
        );

        run_case(
            &test,
            false,
            single_sampled_color,
            None,
            resolve_default,
            None,
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                usage: native::WGPUTextureUsage_TextureBinding,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_TransientAttachment,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions::color(),
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                0,
                1,
                1,
                1,
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
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
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            false,
            TextureOptions {
                width: 8,
                height: 8,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
            TextureOptions {
                mip_level_count: 3,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2D,
                1,
                2,
                0,
                1,
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
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
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                depth_or_array_layers: 3,
                ..TextureOptions::color()
            },
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2DArray,
                0,
                1,
                1,
                2,
                native::WGPUTextureFormat_RGBA8Unorm,
            )),
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                format: native::WGPUTextureFormat_BGRA8Unorm,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            multisampled_color,
            None,
            TextureOptions {
                format: native::WGPUTextureFormat_RGBA8UnormSrgb,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            TextureOptions {
                width: 16,
                height: 16,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
            TextureOptions {
                width: 8,
                height: 16,
                ..TextureOptions::color()
            },
            None,
        );
        run_case(
            &test,
            false,
            TextureOptions {
                width: 16,
                height: 16,
                sample_count: 4,
                ..TextureOptions::color()
            },
            None,
            TextureOptions {
                width: 16,
                height: 8,
                ..TextureOptions::color()
            },
            None,
        );
    }
}

unsafe fn run_case(
    test: &ValidationTest,
    success: bool,
    color_options: TextureOptions,
    color_view_descriptor: Option<native::WGPUTextureViewDescriptor>,
    resolve_options: TextureOptions,
    resolve_view_descriptor: Option<native::WGPUTextureViewDescriptor>,
) {
    let color = create_view(test.device(), color_options, color_view_descriptor);
    let resolve = create_view(test.device(), resolve_options, resolve_view_descriptor);
    let attachment = color_attachment_with_resolve(color.view, resolve.view);
    let descriptor = render_pass_descriptor(&[attachment], None);

    expect_render_pass(test, success, &descriptor);

    release_view(resolve);
    release_view(color);
}
