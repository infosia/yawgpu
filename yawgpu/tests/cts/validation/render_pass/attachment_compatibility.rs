use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    color_attachment, create_view, depth_stencil_attachment, empty_string_view,
    expect_render_pass_with_commands, release_view, render_pass_descriptor,
    sparse_color_attachment, TextureOptions, ViewResource,
};

#[test]
fn render_pass_and_bundle_color_format() {
    let test = ValidationTest::new();
    unsafe {
        let rgba = [native::WGPUTextureFormat_RGBA8Unorm];
        let bgra = [native::WGPUTextureFormat_BGRA8Unorm];
        let matching = create_bundle(&test, &rgba, native::WGPUTextureFormat_Undefined, 1);
        let mismatched = create_bundle(&test, &bgra, native::WGPUTextureFormat_Undefined, 1);

        expect_execute_bundle(
            &test,
            true,
            &rgba,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_execute_bundle(
            &test,
            false,
            &rgba,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(matching);
    }
}

#[test]
fn render_pass_and_bundle_color_count() {
    let test = ValidationTest::new();
    unsafe {
        let one = [native::WGPUTextureFormat_RGBA8Unorm];
        let two = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let matching = create_bundle(&test, &two, native::WGPUTextureFormat_Undefined, 1);
        let mismatched = create_bundle(&test, &one, native::WGPUTextureFormat_Undefined, 1);

        expect_execute_bundle(
            &test,
            true,
            &two,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_execute_bundle(
            &test,
            false,
            &two,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(matching);
    }
}

#[test]
fn render_pass_and_bundle_color_sparse() {
    let test = ValidationTest::new();
    unsafe {
        let sparse = [
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let dense = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let matching = create_bundle(&test, &sparse, native::WGPUTextureFormat_Undefined, 1);
        let mismatched = create_bundle(&test, &dense, native::WGPUTextureFormat_Undefined, 1);

        expect_execute_bundle(
            &test,
            true,
            &sparse,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_execute_bundle(
            &test,
            false,
            &sparse,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(matching);
    }
}

#[test]
fn render_pass_and_bundle_depth_format() {
    let test = ValidationTest::new();
    unsafe {
        let matching = create_bundle(&test, &[], native::WGPUTextureFormat_Depth24PlusStencil8, 1);
        let mismatched = create_bundle(&test, &[], native::WGPUTextureFormat_Depth32Float, 1);

        expect_execute_bundle(
            &test,
            true,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            matching,
        );
        expect_execute_bundle(
            &test,
            false,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(matching);
    }
}

#[test]
fn render_pass_and_bundle_sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let rgba = [native::WGPUTextureFormat_RGBA8Unorm];
        let matching = create_bundle(&test, &rgba, native::WGPUTextureFormat_Undefined, 4);
        let mismatched = create_bundle(&test, &rgba, native::WGPUTextureFormat_Undefined, 1);

        expect_execute_bundle(
            &test,
            true,
            &rgba,
            native::WGPUTextureFormat_Undefined,
            4,
            matching,
        );
        expect_execute_bundle(
            &test,
            false,
            &rgba,
            native::WGPUTextureFormat_Undefined,
            4,
            mismatched,
        );

        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(matching);
    }
}

#[test]
fn render_pass_and_bundle_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        let rgba = [native::WGPUTextureFormat_RGBA8Unorm];
        let bundle = create_bundle(&foreign, &rgba, native::WGPUTextureFormat_Undefined, 1);

        expect_execute_bundle(
            &test,
            false,
            &rgba,
            native::WGPUTextureFormat_Undefined,
            1,
            bundle,
        );

        yawgpu::wgpuRenderBundleRelease(bundle);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_color_format() {
    let test = ValidationTest::new();
    unsafe {
        let pass_formats = [native::WGPUTextureFormat_RGBA8Unorm];
        let matching = create_pipeline(
            &test,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );
        let mismatched = create_pipeline(
            &test,
            &[native::WGPUTextureFormat_BGRA8Unorm],
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );

        expect_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );
        expect_bundle_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_bundle_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderPipelineRelease(mismatched);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_color_count() {
    let test = ValidationTest::new();
    unsafe {
        let pass_formats = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let matching = create_pipeline(
            &test,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );
        let mismatched = create_pipeline(
            &test,
            &[native::WGPUTextureFormat_RGBA8Unorm],
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );

        expect_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );
        expect_bundle_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_bundle_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderPipelineRelease(mismatched);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_color_sparse() {
    let test = ValidationTest::new();
    unsafe {
        let pass_formats = [
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let dense_formats = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
        ];
        let matching = create_pipeline(
            &test,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );
        let mismatched = create_pipeline(
            &test,
            &dense_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );

        expect_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );
        expect_bundle_set_pipeline(
            &test,
            true,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            matching,
        );
        expect_bundle_set_pipeline(
            &test,
            false,
            &pass_formats,
            native::WGPUTextureFormat_Undefined,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderPipelineRelease(mismatched);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_depth_format() {
    let test = ValidationTest::new();
    unsafe {
        let matching = create_pipeline(
            &test,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            native::WGPUOptionalBool_True,
        );
        let mismatched = create_pipeline(
            &test,
            &[],
            native::WGPUTextureFormat_Depth32Float,
            1,
            native::WGPUOptionalBool_True,
        );

        expect_set_pipeline(
            &test,
            true,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            matching,
        );
        expect_set_pipeline(
            &test,
            false,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            mismatched,
        );
        expect_bundle_set_pipeline(
            &test,
            true,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            matching,
        );
        expect_bundle_set_pipeline(
            &test,
            false,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            mismatched,
        );

        yawgpu::wgpuRenderPipelineRelease(mismatched);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_depth_stencil_read_only_write_state() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_pipeline(
            &test,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            native::WGPUOptionalBool_True,
        );

        expect_read_only_depth_pipeline(&test, false, pipeline);
        expect_read_only_depth_bundle_pipeline(&test, false, pipeline);

        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn render_pass_or_bundle_and_pipeline_sample_count() {
    let test = ValidationTest::new();
    unsafe {
        let formats = [native::WGPUTextureFormat_RGBA8Unorm];
        let matching = create_pipeline(
            &test,
            &formats,
            native::WGPUTextureFormat_Undefined,
            4,
            native::WGPUOptionalBool_False,
        );
        let mismatched = create_pipeline(
            &test,
            &formats,
            native::WGPUTextureFormat_Undefined,
            1,
            native::WGPUOptionalBool_False,
        );

        expect_set_pipeline(
            &test,
            true,
            &formats,
            native::WGPUTextureFormat_Undefined,
            4,
            matching,
        );
        expect_set_pipeline(
            &test,
            false,
            &formats,
            native::WGPUTextureFormat_Undefined,
            4,
            mismatched,
        );
        expect_bundle_set_pipeline(
            &test,
            true,
            &formats,
            native::WGPUTextureFormat_Undefined,
            4,
            matching,
        );
        expect_bundle_set_pipeline(
            &test,
            false,
            &formats,
            native::WGPUTextureFormat_Undefined,
            4,
            mismatched,
        );

        yawgpu::wgpuRenderPipelineRelease(mismatched);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

unsafe fn expect_execute_bundle(
    test: &ValidationTest,
    success: bool,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    bundle: native::WGPURenderBundle,
) {
    with_pass_descriptor(
        test.device(),
        color_formats,
        depth_format,
        sample_count,
        |descriptor| unsafe {
            expect_render_pass_with_commands(test, success, &descriptor, |pass| {
                yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
            });
        },
    );
}

unsafe fn expect_set_pipeline(
    test: &ValidationTest,
    success: bool,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    pipeline: native::WGPURenderPipeline,
) {
    with_pass_descriptor(
        test.device(),
        color_formats,
        depth_format,
        sample_count,
        |descriptor| unsafe {
            expect_render_pass_with_commands(test, success, &descriptor, |pass| {
                yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            });
        },
    );
}

unsafe fn expect_bundle_set_pipeline(
    test: &ValidationTest,
    success: bool,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    pipeline: native::WGPURenderPipeline,
) {
    let descriptor = bundle_descriptor(color_formats, depth_format, sample_count, false, false);
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
    assert!(!encoder.is_null());
    test.clear_errors();
    yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);
    if success {
        assert!(test.errors().is_empty());
    } else {
        test.clear_errors();
    }
    let bundle = finish_bundle(test, encoder, success);
    yawgpu::wgpuRenderBundleRelease(bundle);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
}

unsafe fn expect_read_only_depth_pipeline(
    test: &ValidationTest,
    success: bool,
    pipeline: native::WGPURenderPipeline,
) {
    let depth = create_view(
        test.device(),
        TextureOptions {
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            ..TextureOptions::depth_stencil()
        },
        None,
    );
    let mut depth_attachment = depth_stencil_attachment(depth.view);
    depth_attachment.depthReadOnly = 1;
    depth_attachment.depthLoadOp = native::WGPULoadOp_Undefined;
    depth_attachment.depthStoreOp = native::WGPUStoreOp_Undefined;
    let descriptor = render_pass_descriptor(&[], Some(&depth_attachment));
    expect_render_pass_with_commands(test, success, &descriptor, |pass| unsafe {
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    });
    release_view(depth);
}

unsafe fn expect_read_only_depth_bundle_pipeline(
    test: &ValidationTest,
    success: bool,
    pipeline: native::WGPURenderPipeline,
) {
    let descriptor = bundle_descriptor(
        &[],
        native::WGPUTextureFormat_Depth24PlusStencil8,
        1,
        true,
        false,
    );
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
    assert!(!encoder.is_null());
    test.clear_errors();
    yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);
    if success {
        assert!(test.errors().is_empty());
    } else {
        test.clear_errors();
    }
    let bundle = finish_bundle(test, encoder, success);
    yawgpu::wgpuRenderBundleRelease(bundle);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
}

unsafe fn create_bundle(
    test: &ValidationTest,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
) -> native::WGPURenderBundle {
    let descriptor = bundle_descriptor(color_formats, depth_format, sample_count, false, false);
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
    assert!(!encoder.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected bundle encoder errors: {:?}",
        test.errors()
    );
    let bundle = finish_bundle(test, encoder, true);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
}

fn bundle_descriptor(
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    depth_read_only: bool,
    stencil_read_only: bool,
) -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: color_formats.len(),
        colorFormats: color_formats.as_ptr(),
        depthStencilFormat: depth_format,
        sampleCount: sample_count,
        depthReadOnly: depth_read_only.into(),
        stencilReadOnly: stencil_read_only.into(),
    }
}

unsafe fn finish_bundle(
    test: &ValidationTest,
    encoder: native::WGPURenderBundleEncoder,
    success: bool,
) -> native::WGPURenderBundle {
    if success {
        test.clear_errors();
        let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
        assert!(!bundle.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected bundle finish errors: {:?}",
            test.errors()
        );
        bundle
    } else {
        let mut bundle = std::ptr::null();
        test.assert_device_error_after(
            || {
                bundle =
                    unsafe { yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null()) };
            },
            None,
        );
        assert!(!bundle.is_null());
        bundle
    }
}

unsafe fn with_pass_descriptor<F>(
    device: native::WGPUDevice,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    action: F,
) where
    F: FnOnce(native::WGPURenderPassDescriptor),
{
    let mut views: Vec<ViewResource> = Vec::new();
    let mut attachments = Vec::new();
    for &format in color_formats {
        if format == native::WGPUTextureFormat_Undefined {
            attachments.push(sparse_color_attachment());
        } else {
            let view = create_view(
                device,
                TextureOptions {
                    format,
                    sample_count,
                    ..TextureOptions::color()
                },
                None,
            );
            attachments.push(color_attachment(view.view));
            views.push(view);
        }
    }

    let depth = if depth_format == native::WGPUTextureFormat_Undefined {
        None
    } else {
        Some(create_view(
            device,
            TextureOptions {
                format: depth_format,
                sample_count,
                ..TextureOptions::depth_stencil()
            },
            None,
        ))
    };
    let depth_attachment = depth.map(|resource| depth_stencil_attachment(resource.view));
    let descriptor = render_pass_descriptor(&attachments, depth_attachment.as_ref());
    action(descriptor);

    if let Some(resource) = depth {
        release_view(resource);
    }
    for view in views {
        release_view(view);
    }
}

unsafe fn create_pipeline(
    test: &ValidationTest,
    color_formats: &[native::WGPUTextureFormat],
    depth_format: native::WGPUTextureFormat,
    sample_count: u32,
    depth_write_enabled: native::WGPUOptionalBool,
) -> native::WGPURenderPipeline {
    let vertex_source = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }";
    let fragment_source = fragment_source(color_formats);
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), &fragment_source);
    let targets: Vec<_> = color_formats
        .iter()
        .copied()
        .map(|format| native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        })
        .collect();
    let fragment = (!targets.is_empty()).then_some(native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: targets.len(),
        targets: targets.as_ptr(),
    });
    let depth_stencil = (depth_format != native::WGPUTextureFormat_Undefined)
        .then(|| depth_stencil_state(depth_format, depth_write_enabled));
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
        primitive: primitive_state(),
        depthStencil: depth_stencil
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: sample_count,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: fragment
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
    };
    test.clear_errors();
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected pipeline errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuShaderModuleRelease(fragment_module);
    yawgpu::wgpuShaderModuleRelease(vertex_module);
    pipeline
}

fn fragment_source(color_formats: &[native::WGPUTextureFormat]) -> String {
    let locations: Vec<_> = color_formats
        .iter()
        .enumerate()
        .filter_map(|(index, format)| {
            (*format != native::WGPUTextureFormat_Undefined).then_some(index)
        })
        .collect();
    if locations.is_empty() {
        return "@fragment fn fs() {}".to_owned();
    }
    if locations.len() == 1 {
        return format!(
            "@fragment fn fs() -> @location({}) vec4f {{ return vec4f(0.0); }}",
            locations[0]
        );
    }
    let fields = locations
        .iter()
        .map(|location| format!("@location({location}) c{location}: vec4f,"))
        .collect::<String>();
    let values = locations
        .iter()
        .map(|_| "vec4f(0.0)")
        .collect::<Vec<_>>()
        .join(", ");
    format!("struct FsOut {{ {fields} }} @fragment fn fs() -> FsOut {{ return FsOut({values}); }}")
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
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

fn depth_stencil_state(
    format: native::WGPUTextureFormat,
    depth_write_enabled: native::WGPUOptionalBool,
) -> native::WGPUDepthStencilState {
    native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format,
        depthWriteEnabled: depth_write_enabled,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    }
}

fn stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Undefined,
        failOp: native::WGPUStencilOperation_Undefined,
        depthFailOp: native::WGPUStencilOperation_Undefined,
        passOp: native::WGPUStencilOperation_Undefined,
    }
}

fn primitive_state() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
