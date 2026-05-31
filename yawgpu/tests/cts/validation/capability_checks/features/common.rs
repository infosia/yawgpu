//! Shared helpers for CTS `capability_checks/features` and feature-gated texture specs.

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

use crate::common;

pub const OPTIONAL_FEATURES: &[(&str, native::WGPUFeatureName)] = &[
    ("clip-distances", native::WGPUFeatureName_ClipDistances),
    ("timestamp-query", native::WGPUFeatureName_TimestampQuery),
    ("subgroups", native::WGPUFeatureName_Subgroups),
    (
        "texture-component-swizzle",
        native::WGPUFeatureName_TextureComponentSwizzle,
    ),
    (
        "texture-compression-bc",
        native::WGPUFeatureName_TextureCompressionBC,
    ),
    (
        "texture-compression-bc-sliced-3d",
        native::WGPUFeatureName_TextureCompressionBCSliced3D,
    ),
    (
        "texture-compression-astc",
        native::WGPUFeatureName_TextureCompressionASTC,
    ),
    (
        "texture-compression-astc-sliced-3d",
        native::WGPUFeatureName_TextureCompressionASTCSliced3D,
    ),
    (
        "depth32float-stencil8",
        native::WGPUFeatureName_Depth32FloatStencil8,
    ),
    (
        "rg11b10ufloat-renderable",
        native::WGPUFeatureName_RG11B10UfloatRenderable,
    ),
    (
        "texture-formats-tier1",
        native::WGPUFeatureName_TextureFormatsTier1,
    ),
    (
        "texture-formats-tier2",
        native::WGPUFeatureName_TextureFormatsTier2,
    ),
    (
        "bgra8unorm-storage",
        native::WGPUFeatureName_BGRA8UnormStorage,
    ),
    (
        "float32-filterable",
        native::WGPUFeatureName_Float32Filterable,
    ),
];

pub fn feature_name(feature: native::WGPUFeatureName) -> &'static str {
    OPTIONAL_FEATURES
        .iter()
        .find_map(|(name, candidate)| (*candidate == feature).then_some(*name))
        .unwrap_or("unknown")
}

pub fn adapter_has_feature(feature: native::WGPUFeatureName) -> bool {
    let test = ValidationTest::new();
    unsafe { yawgpu::wgpuAdapterHasFeature(test.adapter(), feature) != 0 }
}

pub fn assert_noop_lacks_feature(feature: native::WGPUFeatureName) {
    assert!(!adapter_has_feature(feature));
}

pub fn assert_noop_advertises_feature(feature: native::WGPUFeatureName) {
    assert!(adapter_has_feature(feature));
}

pub unsafe fn create_query_set_ok(
    test: &ValidationTest,
    query_type: native::WGPUQueryType,
) -> native::WGPUQuerySet {
    let mut query_set = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        query_set = common::create_query_set(test.device(), query_type, 1);
    });
    query_set
}

pub unsafe fn create_query_set_error(test: &ValidationTest, query_type: native::WGPUQueryType) {
    let mut query_set = std::ptr::null();
    assert_device_error!({
        query_set = unsafe { common::create_query_set(test.device(), query_type, 1) };
    });
    assert!(!query_set.is_null());
    unsafe { yawgpu::wgpuQuerySetRelease(query_set) };
}

pub fn test_with_feature(feature: native::WGPUFeatureName) -> ValidationTest {
    ValidationTest::with_features(&[feature])
}

pub fn assert_device_has_feature(test: &ValidationTest, feature: native::WGPUFeatureName) {
    assert_ne!(
        unsafe { yawgpu::wgpuDeviceHasFeature(test.device(), feature) },
        0
    );
}

pub unsafe fn assert_texture_ok(
    test: &ValidationTest,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let mut texture = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
    });
    assert!(!texture.is_null());
    texture
}

pub unsafe fn assert_texture_error(
    test: &ValidationTest,
    descriptor: native::WGPUTextureDescriptor,
) {
    let mut texture = std::ptr::null();
    assert_device_error!({
        texture = unsafe { yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor) };
    });
    assert!(!texture.is_null());
    unsafe { yawgpu::wgpuTextureRelease(texture) };
}

pub fn texture_descriptor(
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: common::empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: block_width(format),
            height: block_height(format),
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

pub fn texture_descriptor_3d(
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        dimension: native::WGPUTextureDimension_3D,
        size: native::WGPUExtent3D {
            width: block_width(format),
            height: block_height(format),
            depthOrArrayLayers: 4,
        },
        ..texture_descriptor(format, usage)
    }
}

pub unsafe fn assert_texture_view_format_error(
    test: &ValidationTest,
    texture_format: native::WGPUTextureFormat,
    view_format: native::WGPUTextureFormat,
) {
    let texture = unsafe {
        assert_texture_ok(
            test,
            texture_descriptor(texture_format, native::WGPUTextureUsage_TextureBinding),
        )
    };
    let descriptor = texture_view_descriptor(view_format);
    let mut view = std::ptr::null();
    assert_device_error!({
        view = unsafe { yawgpu::wgpuTextureCreateView(texture, &descriptor) };
    });
    assert!(!view.is_null());
    unsafe {
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

pub fn texture_view_descriptor(
    format: native::WGPUTextureFormat,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: common::empty_string_view(),
        format,
        dimension: native::WGPUTextureViewDimension_2D,
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_None,
    }
}

pub unsafe fn assert_texture_view_ok(
    test: &ValidationTest,
    texture_format: native::WGPUTextureFormat,
    view_format: native::WGPUTextureFormat,
) -> native::WGPUTextureView {
    let texture = unsafe {
        assert_texture_ok(
            test,
            texture_descriptor(texture_format, native::WGPUTextureUsage_TextureBinding),
        )
    };
    let descriptor = texture_view_descriptor(view_format);
    let mut view = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    });
    assert!(!view.is_null());
    unsafe { yawgpu::wgpuTextureRelease(texture) };
    view
}

pub unsafe fn assert_storage_texture_bgl_ok(
    test: &ValidationTest,
    format: native::WGPUTextureFormat,
    access: native::WGPUStorageTextureAccess,
) -> native::WGPUBindGroupLayout {
    let entry = storage_texture_entry(format, access);
    let mut layout = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        layout = common::create_bind_group_layout(test.device(), &[entry]);
    });
    layout
}

pub unsafe fn assert_storage_texture_bgl_error(
    test: &ValidationTest,
    format: native::WGPUTextureFormat,
    access: native::WGPUStorageTextureAccess,
) {
    let entry = storage_texture_entry(format, access);
    let mut layout = std::ptr::null();
    assert_device_error!({
        layout = unsafe { common::create_bind_group_layout(test.device(), &[entry]) };
    });
    assert!(!layout.is_null());
    unsafe { yawgpu::wgpuBindGroupLayoutRelease(layout) };
}

pub fn storage_texture_entry(
    format: native::WGPUTextureFormat,
    access: native::WGPUStorageTextureAccess,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Fragment,
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
            access,
            format,
            viewDimension: native::WGPUTextureViewDimension_2D,
        },
    }
}

pub fn optional_formats() -> &'static [native::WGPUTextureFormat] {
    &[
        native::WGPUTextureFormat_RG11B10Ufloat,
        native::WGPUTextureFormat_Depth32FloatStencil8,
        native::WGPUTextureFormat_BC1RGBAUnorm,
        native::WGPUTextureFormat_BC1RGBAUnormSrgb,
        native::WGPUTextureFormat_BC7RGBAUnorm,
        native::WGPUTextureFormat_BC7RGBAUnormSrgb,
    ]
}

pub fn tier1_storage_formats() -> &'static [native::WGPUTextureFormat] {
    &[
        native::WGPUTextureFormat_R8Unorm,
        native::WGPUTextureFormat_R8Snorm,
        native::WGPUTextureFormat_R8Uint,
        native::WGPUTextureFormat_R8Sint,
        native::WGPUTextureFormat_RG8Unorm,
        native::WGPUTextureFormat_RG8Snorm,
        native::WGPUTextureFormat_RG8Uint,
        native::WGPUTextureFormat_RG8Sint,
        native::WGPUTextureFormat_R16Uint,
        native::WGPUTextureFormat_R16Sint,
        native::WGPUTextureFormat_R16Float,
        native::WGPUTextureFormat_RG16Uint,
        native::WGPUTextureFormat_RG16Sint,
        native::WGPUTextureFormat_RG16Float,
        native::WGPUTextureFormat_RGB10A2Uint,
        native::WGPUTextureFormat_RGB10A2Unorm,
        native::WGPUTextureFormat_RG11B10Ufloat,
        native::WGPUTextureFormat_RGBA8Snorm,
    ]
}

pub fn tier2_read_write_formats() -> &'static [native::WGPUTextureFormat] {
    &[
        native::WGPUTextureFormat_R32Float,
        native::WGPUTextureFormat_RGBA16Float,
        native::WGPUTextureFormat_RGBA32Float,
    ]
}

pub unsafe fn assert_color_target_pipeline_ok(
    test: &ValidationTest,
    format: native::WGPUTextureFormat,
) -> native::WGPURenderPipeline {
    let vertex = unsafe { common::create_wgsl_module(test.device(), VERTEX_SHADER) };
    let fragment = unsafe { common::create_wgsl_module(test.device(), FRAGMENT_SHADER) };
    let target = native::WGPUColorTargetState {
        format,
        ..common::color_target()
    };
    let fragment_state = common::fragment_state(fragment, "fs", &target);
    let descriptor = common::render_pipeline_descriptor(vertex, &fragment_state);
    let mut pipeline = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    });
    assert!(!pipeline.is_null());
    pipeline
}

pub unsafe fn assert_color_target_pipeline_error(
    test: &ValidationTest,
    format: native::WGPUTextureFormat,
) {
    let vertex = unsafe { common::create_wgsl_module(test.device(), VERTEX_SHADER) };
    let fragment = unsafe { common::create_wgsl_module(test.device(), FRAGMENT_SHADER) };
    let target = native::WGPUColorTargetState {
        format,
        ..common::color_target()
    };
    let fragment_state = common::fragment_state(fragment, "fs", &target);
    let descriptor = common::render_pipeline_descriptor(vertex, &fragment_state);
    let mut pipeline = std::ptr::null();
    assert_device_error!({
        pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) };
    });
    assert!(!pipeline.is_null());
    unsafe { yawgpu::wgpuRenderPipelineRelease(pipeline) };
}

pub unsafe fn assert_render_bundle_encoder_ok(
    test: &ValidationTest,
    color_format: native::WGPUTextureFormat,
    depth_stencil_format: native::WGPUTextureFormat,
) -> native::WGPURenderBundleEncoder {
    let colors = [color_format];
    let descriptor = native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: common::empty_string_view(),
        colorFormatCount: usize::from(color_format != native::WGPUTextureFormat_Undefined),
        colorFormats: colors.as_ptr(),
        depthStencilFormat: depth_stencil_format,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    };
    let mut encoder = std::ptr::null();
    test.expect_no_validation_error(|| unsafe {
        encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
    });
    assert!(!encoder.is_null());
    encoder
}

pub unsafe fn assert_render_bundle_encoder_error(
    test: &ValidationTest,
    color_format: native::WGPUTextureFormat,
    depth_stencil_format: native::WGPUTextureFormat,
) {
    let colors = [color_format];
    let descriptor = native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: common::empty_string_view(),
        colorFormatCount: usize::from(color_format != native::WGPUTextureFormat_Undefined),
        colorFormats: colors.as_ptr(),
        depthStencilFormat: depth_stencil_format,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    };
    let mut encoder = std::ptr::null();
    assert_device_error!({
        encoder =
            unsafe { yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor) };
    });
    assert!(!encoder.is_null());
    unsafe { yawgpu::wgpuRenderBundleEncoderRelease(encoder) };
}

pub fn block_width(format: native::WGPUTextureFormat) -> u32 {
    match format {
        native::WGPUTextureFormat_BC1RGBAUnorm
        | native::WGPUTextureFormat_BC1RGBAUnormSrgb
        | native::WGPUTextureFormat_BC2RGBAUnorm
        | native::WGPUTextureFormat_BC2RGBAUnormSrgb
        | native::WGPUTextureFormat_BC3RGBAUnorm
        | native::WGPUTextureFormat_BC3RGBAUnormSrgb
        | native::WGPUTextureFormat_BC4RUnorm
        | native::WGPUTextureFormat_BC4RSnorm
        | native::WGPUTextureFormat_BC5RGUnorm
        | native::WGPUTextureFormat_BC5RGSnorm
        | native::WGPUTextureFormat_BC6HRGBUfloat
        | native::WGPUTextureFormat_BC6HRGBFloat
        | native::WGPUTextureFormat_BC7RGBAUnorm
        | native::WGPUTextureFormat_BC7RGBAUnormSrgb
        | native::WGPUTextureFormat_ETC2RGB8Unorm
        | native::WGPUTextureFormat_ETC2RGB8UnormSrgb
        | native::WGPUTextureFormat_ETC2RGB8A1Unorm
        | native::WGPUTextureFormat_ETC2RGB8A1UnormSrgb
        | native::WGPUTextureFormat_ETC2RGBA8Unorm
        | native::WGPUTextureFormat_ETC2RGBA8UnormSrgb
        | native::WGPUTextureFormat_ASTC4x4Unorm
        | native::WGPUTextureFormat_ASTC4x4UnormSrgb => 4,
        native::WGPUTextureFormat_ASTC5x4Unorm
        | native::WGPUTextureFormat_ASTC5x4UnormSrgb
        | native::WGPUTextureFormat_ASTC5x5Unorm
        | native::WGPUTextureFormat_ASTC5x5UnormSrgb => 5,
        native::WGPUTextureFormat_ASTC6x5Unorm
        | native::WGPUTextureFormat_ASTC6x5UnormSrgb
        | native::WGPUTextureFormat_ASTC6x6Unorm
        | native::WGPUTextureFormat_ASTC6x6UnormSrgb => 6,
        native::WGPUTextureFormat_ASTC8x5Unorm
        | native::WGPUTextureFormat_ASTC8x5UnormSrgb
        | native::WGPUTextureFormat_ASTC8x6Unorm
        | native::WGPUTextureFormat_ASTC8x6UnormSrgb
        | native::WGPUTextureFormat_ASTC8x8Unorm
        | native::WGPUTextureFormat_ASTC8x8UnormSrgb => 8,
        native::WGPUTextureFormat_ASTC10x5Unorm
        | native::WGPUTextureFormat_ASTC10x5UnormSrgb
        | native::WGPUTextureFormat_ASTC10x6Unorm
        | native::WGPUTextureFormat_ASTC10x6UnormSrgb
        | native::WGPUTextureFormat_ASTC10x8Unorm
        | native::WGPUTextureFormat_ASTC10x8UnormSrgb
        | native::WGPUTextureFormat_ASTC10x10Unorm
        | native::WGPUTextureFormat_ASTC10x10UnormSrgb => 10,
        native::WGPUTextureFormat_ASTC12x10Unorm
        | native::WGPUTextureFormat_ASTC12x10UnormSrgb
        | native::WGPUTextureFormat_ASTC12x12Unorm
        | native::WGPUTextureFormat_ASTC12x12UnormSrgb => 12,
        _ => 4,
    }
}

pub fn block_height(format: native::WGPUTextureFormat) -> u32 {
    match format {
        native::WGPUTextureFormat_BC1RGBAUnorm
        | native::WGPUTextureFormat_BC1RGBAUnormSrgb
        | native::WGPUTextureFormat_BC2RGBAUnorm
        | native::WGPUTextureFormat_BC2RGBAUnormSrgb
        | native::WGPUTextureFormat_BC3RGBAUnorm
        | native::WGPUTextureFormat_BC3RGBAUnormSrgb
        | native::WGPUTextureFormat_BC4RUnorm
        | native::WGPUTextureFormat_BC4RSnorm
        | native::WGPUTextureFormat_BC5RGUnorm
        | native::WGPUTextureFormat_BC5RGSnorm
        | native::WGPUTextureFormat_BC6HRGBUfloat
        | native::WGPUTextureFormat_BC6HRGBFloat
        | native::WGPUTextureFormat_BC7RGBAUnorm
        | native::WGPUTextureFormat_BC7RGBAUnormSrgb
        | native::WGPUTextureFormat_ETC2RGB8Unorm
        | native::WGPUTextureFormat_ETC2RGB8UnormSrgb
        | native::WGPUTextureFormat_ETC2RGB8A1Unorm
        | native::WGPUTextureFormat_ETC2RGB8A1UnormSrgb
        | native::WGPUTextureFormat_ETC2RGBA8Unorm
        | native::WGPUTextureFormat_ETC2RGBA8UnormSrgb
        | native::WGPUTextureFormat_ASTC4x4Unorm
        | native::WGPUTextureFormat_ASTC4x4UnormSrgb
        | native::WGPUTextureFormat_ASTC5x4Unorm
        | native::WGPUTextureFormat_ASTC5x4UnormSrgb => 4,
        native::WGPUTextureFormat_ASTC5x5Unorm
        | native::WGPUTextureFormat_ASTC5x5UnormSrgb
        | native::WGPUTextureFormat_ASTC6x5Unorm
        | native::WGPUTextureFormat_ASTC6x5UnormSrgb
        | native::WGPUTextureFormat_ASTC8x5Unorm
        | native::WGPUTextureFormat_ASTC8x5UnormSrgb
        | native::WGPUTextureFormat_ASTC10x5Unorm
        | native::WGPUTextureFormat_ASTC10x5UnormSrgb => 5,
        native::WGPUTextureFormat_ASTC6x6Unorm
        | native::WGPUTextureFormat_ASTC6x6UnormSrgb
        | native::WGPUTextureFormat_ASTC8x6Unorm
        | native::WGPUTextureFormat_ASTC8x6UnormSrgb
        | native::WGPUTextureFormat_ASTC10x6Unorm
        | native::WGPUTextureFormat_ASTC10x6UnormSrgb => 6,
        native::WGPUTextureFormat_ASTC8x8Unorm
        | native::WGPUTextureFormat_ASTC8x8UnormSrgb
        | native::WGPUTextureFormat_ASTC10x8Unorm
        | native::WGPUTextureFormat_ASTC10x8UnormSrgb => 8,
        native::WGPUTextureFormat_ASTC10x10Unorm
        | native::WGPUTextureFormat_ASTC10x10UnormSrgb
        | native::WGPUTextureFormat_ASTC12x10Unorm
        | native::WGPUTextureFormat_ASTC12x10UnormSrgb => 10,
        native::WGPUTextureFormat_ASTC12x12Unorm | native::WGPUTextureFormat_ASTC12x12UnormSrgb => {
            12
        }
        _ => 4,
    }
}

const VERTEX_SHADER: &str = r#"
@vertex
fn main() -> @builtin(position) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER: &str = r#"
@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(0.0);
}
"#;
