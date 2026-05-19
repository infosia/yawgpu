use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn render_pass_debug_markers_success() {
    let test = ValidationTest::new();
    unsafe {
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
            yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn render_pass_debug_group_unbalanced_push_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn render_pass_debug_group_unbalanced_pop_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
            yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn render_bundle_debug_markers_success() {
    let test = ValidationTest::new();
    unsafe {
        let bundle = assert_render_bundle_ok(&test, |encoder| {
            yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, string_view("Marker"));
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view("Event Start"));
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view("Event Start"));
            yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, string_view("Marker"));
            yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
            yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
        });
        yawgpu::wgpuRenderBundleRelease(bundle);
    }
}

#[test]
fn render_bundle_debug_group_unbalanced_push_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let bundle = assert_render_bundle_error(&test, |encoder| {
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view("Event Start"));
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view("Event Start"));
            yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, string_view("Marker"));
            yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
        });
        yawgpu::wgpuRenderBundleRelease(bundle);
    }
}

#[test]
fn render_bundle_debug_group_unbalanced_pop_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let bundle = assert_render_bundle_error(&test, |encoder| {
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view("Event Start"));
            yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, string_view("Marker"));
            yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
            yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
        });
        yawgpu::wgpuRenderBundleRelease(bundle);
    }
}

#[test]
fn compute_pass_debug_markers_success() {
    let test = ValidationTest::new();
    unsafe {
        assert_compute_pass_ok(&test, |pass| {
            yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
            yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn compute_pass_debug_group_unbalanced_push_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn compute_pass_debug_group_unbalanced_pop_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
            yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
            yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
            yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
        });
    }
}

#[test]
fn command_encoder_debug_markers_success() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        test.clear_errors();
        yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view("Marker"));
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view("Marker"));
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        assert!(test.errors().is_empty());
        let command_buffer = finish_ok(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn command_encoder_debug_group_unbalanced_push_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view("Marker"));
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        let command_buffer = finish_error(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn command_encoder_debug_group_unbalanced_pop_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view("Marker"));
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        let command_buffer = finish_error(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn compute_pass_and_command_encoder_debug_groups_are_independent() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        assert!(!pass.is_null());
        yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("Event Start"));
        yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
        yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        let command_buffer = finish_ok(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let independent_encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(independent_encoder, string_view("Event Start"));
        let independent_pass =
            yawgpu::wgpuCommandEncoderBeginComputePass(independent_encoder, std::ptr::null());
        assert!(!independent_pass.is_null());
        yawgpu::wgpuComputePassEncoderInsertDebugMarker(independent_pass, string_view("Marker"));
        yawgpu::wgpuComputePassEncoderPopDebugGroup(independent_pass);
        test.clear_errors();
        yawgpu::wgpuComputePassEncoderEnd(independent_pass);
        let independent_buffer = finish_error(&test, independent_encoder);
        yawgpu::wgpuCommandBufferRelease(independent_buffer);
        yawgpu::wgpuComputePassEncoderRelease(independent_pass);
        yawgpu::wgpuCommandEncoderRelease(independent_encoder);
    }
}

#[test]
fn render_pass_and_command_encoder_debug_groups_are_independent() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        let target = create_render_target(test.device());
        let attachment = color_attachment(target.view);
        let descriptor = render_pass_descriptor(&[attachment]);
        yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("Event Start"));
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("Event Start"));
        yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
        yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
        let command_buffer = finish_ok(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let independent_encoder = create_encoder(&test);
        let independent_target = create_render_target(test.device());
        let independent_attachment = color_attachment(independent_target.view);
        let independent_descriptor = render_pass_descriptor(&[independent_attachment]);
        yawgpu::wgpuCommandEncoderPushDebugGroup(independent_encoder, string_view("Event Start"));
        let independent_pass =
            yawgpu::wgpuCommandEncoderBeginRenderPass(independent_encoder, &independent_descriptor);
        assert!(!independent_pass.is_null());
        yawgpu::wgpuRenderPassEncoderInsertDebugMarker(independent_pass, string_view("Marker"));
        yawgpu::wgpuRenderPassEncoderPopDebugGroup(independent_pass);
        test.clear_errors();
        yawgpu::wgpuRenderPassEncoderEnd(independent_pass);
        let independent_buffer = finish_error(&test, independent_encoder);
        yawgpu::wgpuCommandBufferRelease(independent_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(independent_pass);
        release_render_target(independent_target);
        yawgpu::wgpuCommandEncoderRelease(independent_encoder);
    }
}

unsafe fn assert_render_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(test.device());
    let attachment = color_attachment(target.view);
    let descriptor = render_pass_descriptor(&[attachment]);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_render_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(test.device());
    let attachment = color_attachment(target.view);
    let descriptor = render_pass_descriptor(&[attachment]);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_compute_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    commands(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuComputePassEncoderEnd(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_compute_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    commands(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuComputePassEncoderEnd(pass);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_render_bundle_ok<F>(test: &ValidationTest, commands: F) -> native::WGPURenderBundle
where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let format = native::WGPUTextureFormat_RGBA8Unorm;
    let descriptor = bundle_descriptor(&[format]);
    let encoder = create_bundle_encoder_ok(test, &descriptor);
    commands(encoder);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let bundle = finish_bundle_ok(test, encoder);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
}

unsafe fn assert_render_bundle_error<F>(
    test: &ValidationTest,
    commands: F,
) -> native::WGPURenderBundle
where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let format = native::WGPUTextureFormat_RGBA8Unorm;
    let descriptor = bundle_descriptor(&[format]);
    let encoder = create_bundle_encoder_ok(test, &descriptor);
    commands(encoder);
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    let bundle = finish_bundle_error(test, encoder);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
}

unsafe fn create_bundle_encoder_ok(
    test: &ValidationTest,
    descriptor: &native::WGPURenderBundleEncoderDescriptor,
) -> native::WGPURenderBundleEncoder {
    test.clear_errors();
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), descriptor);
    assert!(!encoder.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    encoder
}

unsafe fn finish_bundle_ok(
    test: &ValidationTest,
    encoder: native::WGPURenderBundleEncoder,
) -> native::WGPURenderBundle {
    test.clear_errors();
    let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
    assert!(!bundle.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    bundle
}

unsafe fn finish_bundle_error(
    test: &ValidationTest,
    encoder: native::WGPURenderBundleEncoder,
) -> native::WGPURenderBundle {
    let mut bundle = std::ptr::null();
    test.assert_device_error_after(
        || {
            bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!bundle.is_null());
    bundle
}

unsafe fn create_encoder(test: &ValidationTest) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    command_buffer
}

unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

struct RenderTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn create_render_target(device: native::WGPUDevice) -> RenderTarget {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    RenderTarget { texture, view }
}

unsafe fn release_render_target(target: RenderTarget) {
    yawgpu::wgpuTextureViewRelease(target.view);
    yawgpu::wgpuTextureRelease(target.texture);
}

fn render_pass_descriptor(
    attachments: &[native::WGPURenderPassColorAttachment],
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Load,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    }
}

fn bundle_descriptor(
    formats: &[native::WGPUTextureFormat],
) -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: formats.len(),
        colorFormats: formats.as_ptr(),
        depthStencilFormat: native::WGPUTextureFormat_Undefined,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
