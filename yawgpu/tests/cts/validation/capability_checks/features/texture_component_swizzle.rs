use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::{common, feature_common};

#[test]
#[ignore = "Noop does not advertise texture-component-swizzle"]
fn invalid_swizzle() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    unsafe {
        let texture = common::create_texture(
            test.device(),
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_TextureBinding,
            4,
            4,
        );
        let descriptor =
            swizzle_view_descriptor(native::WGPUTextureFormat_RGBA8Unorm, swizzle_all(999));
        let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn only_identity_swizzle() {
    let test = ValidationTest::new();
    unsafe {
        let texture = common::create_texture(
            test.device(),
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_TextureBinding,
            4,
            4,
        );
        let descriptor = swizzle_view_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureComponentSwizzle {
                r: native::WGPUComponentSwizzle_R,
                g: native::WGPUComponentSwizzle_G,
                b: native::WGPUComponentSwizzle_B,
                a: native::WGPUComponentSwizzle_A,
            },
        );
        let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "Noop does not advertise texture-component-swizzle"]
fn no_render_no_resolve_no_storage() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    unsafe {
        let texture = common::create_texture(
            test.device(),
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_StorageBinding,
            4,
            4,
        );
        let descriptor = swizzle_view_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            swizzle_all(native::WGPUComponentSwizzle_R),
        );
        let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "compatibility mode subcase is deferred"]
fn compatibility_mode() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureComponentSwizzle);
    unsafe {
        let texture = common::create_texture(
            test.device(),
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_TextureBinding,
            4,
            4,
        );
        let descriptor = swizzle_view_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            swizzle_all(native::WGPUComponentSwizzle_R),
        );
        let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
        assert!(!view.is_null());
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

fn swizzle_all(value: native::WGPUComponentSwizzle) -> native::WGPUTextureComponentSwizzle {
    native::WGPUTextureComponentSwizzle {
        r: value,
        g: value,
        b: value,
        a: value,
    }
}

fn swizzle_view_descriptor(
    format: native::WGPUTextureFormat,
    swizzle: native::WGPUTextureComponentSwizzle,
) -> native::WGPUTextureViewDescriptor {
    let swizzle_descriptor = Box::leak(Box::new(native::WGPUTextureComponentSwizzleDescriptor {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_TextureComponentSwizzleDescriptor,
        },
        swizzle,
    }));
    native::WGPUTextureViewDescriptor {
        nextInChain: (&mut swizzle_descriptor.chain) as *mut _,
        format,
        ..feature_common::texture_view_descriptor(format)
    }
}
