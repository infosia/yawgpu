use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

const MAX_QUERY_COUNT: u32 = 4096;

#[test]
fn occlusion_query_set_creation_and_reflection() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 18);
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 18);
        yawgpu::wgpuQuerySetRelease(query_set);

        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_count_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(
            test.device(),
            native::WGPUQueryType_Occlusion,
            MAX_QUERY_COUNT,
        );
        assert!(!query_set.is_null());
        yawgpu::wgpuQuerySetRelease(query_set);

        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 0);
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 0);
        yawgpu::wgpuQuerySetRelease(query_set);

        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(
                test.device(),
                native::WGPUQueryType_Occlusion,
                MAX_QUERY_COUNT + 1,
            );
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), MAX_QUERY_COUNT + 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_type_is_validated_and_reflected_for_error_query_sets() {
    let test = ValidationTest::new();
    unsafe {
        let invalid_type = 0xFFFF;
        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), invalid_type, 76);
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetType(query_set), invalid_type);
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 76);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_requires_feature() {
    let test = ValidationTest::new();
    unsafe {
        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 1);
        });
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Timestamp
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_succeeds_when_feature_is_requested() {
    unsafe {
        let fixture = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        let query_set = create_query_set(fixture.device, native::WGPUQueryType_Timestamp, 1);

        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Timestamp
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        assert!(fixture.errors());
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn destroy_is_idempotent_and_reflection_stays_available() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
        yawgpu::wgpuQuerySetDestroy(query_set);
        yawgpu::wgpuQuerySetDestroy(query_set);
        yawgpu::wgpuQuerySetSetLabel(query_set, string_view("destroyed query set"));

        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 2);
        assert!(test.errors().is_empty());
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_add_ref_release_balances() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
        yawgpu::wgpuQuerySetAddRef(query_set);
        yawgpu::wgpuQuerySetRelease(query_set);
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_feature_plumbing() {
    unsafe {
        let default = FeatureDeviceFixture::new(&[]);
        assert_eq!(
            yawgpu::wgpuDeviceHasFeature(default.device, native::WGPUFeatureName_TimestampQuery),
            0
        );

        let timestamp = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        assert_eq!(
            yawgpu::wgpuDeviceHasFeature(timestamp.device, native::WGPUFeatureName_TimestampQuery),
            1
        );

        let mut features = supported_features_init();
        yawgpu::wgpuDeviceGetFeatures(timestamp.device, &mut features);
        let features_slice = std::slice::from_raw_parts(features.features, features.featureCount);
        assert!(features_slice.contains(&native::WGPUFeatureName_TimestampQuery));
        yawgpu::wgpuSupportedFeaturesFreeMembers(features);
    }
}

#[test]
fn render_pass_query_sets_are_validated() {
    unsafe {
        let fixture = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        let color = create_view(fixture.device);
        let color_attachment = make_color_attachment(color.view);
        let occlusion = create_query_set(fixture.device, native::WGPUQueryType_Occlusion, 2);
        let timestamp = create_query_set(fixture.device, native::WGPUQueryType_Timestamp, 2);

        let mut descriptor = render_pass_descriptor(&[color_attachment]);
        descriptor.occlusionQuerySet = timestamp;
        finish_render_pass_error(&fixture, &descriptor);

        descriptor.occlusionQuerySet = occlusion;
        descriptor.timestampWrites = std::ptr::null();
        finish_render_pass_ok(&fixture, &descriptor);

        let mut timestamp_writes =
            pass_timestamp_writes(occlusion, 0, native::WGPU_QUERY_SET_INDEX_UNDEFINED);
        descriptor.occlusionQuerySet = std::ptr::null();
        descriptor.timestampWrites = &timestamp_writes;
        finish_render_pass_error(&fixture, &descriptor);

        timestamp_writes =
            pass_timestamp_writes(timestamp, 2, native::WGPU_QUERY_SET_INDEX_UNDEFINED);
        descriptor.timestampWrites = &timestamp_writes;
        finish_render_pass_error(&fixture, &descriptor);

        timestamp_writes = pass_timestamp_writes(timestamp, 0, 0);
        descriptor.timestampWrites = &timestamp_writes;
        finish_render_pass_error(&fixture, &descriptor);

        timestamp_writes = pass_timestamp_writes(timestamp, 0, 1);
        descriptor.timestampWrites = &timestamp_writes;
        finish_render_pass_ok(&fixture, &descriptor);

        yawgpu::wgpuQuerySetRelease(occlusion);
        yawgpu::wgpuQuerySetRelease(timestamp);
        release_view(color);
    }
}

#[test]
fn command_encoder_write_timestamp_is_validated() {
    unsafe {
        let fixture = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        let timestamp = create_query_set(fixture.device, native::WGPUQueryType_Timestamp, 2);
        let occlusion = create_query_set(fixture.device, native::WGPUQueryType_Occlusion, 2);

        let encoder = create_encoder(fixture.device);
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, timestamp, 0);
        finish_encoder_ok(&fixture, encoder);

        let encoder = create_encoder(fixture.device);
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, occlusion, 0);
        finish_encoder_error(&fixture, encoder);

        let encoder = create_encoder(fixture.device);
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, timestamp, 2);
        finish_encoder_error(&fixture, encoder);

        yawgpu::wgpuQuerySetRelease(timestamp);
        yawgpu::wgpuQuerySetRelease(occlusion);
    }
}

#[test]
fn command_encoder_resolve_query_set_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 4);
        let destination = create_buffer(test.device(), 4 * 8, native::WGPUBufferUsage_QueryResolve);

        finish_resolve_ok(&test, query_set, 0, 4, destination, 0);
        finish_resolve_error(&test, query_set, 1, 4, destination, 0);
        finish_resolve_error(&test, query_set, 0, 0, destination, 0);

        let without_query_resolve =
            create_buffer(test.device(), 4 * 8, native::WGPUBufferUsage_CopyDst);
        finish_resolve_error(&test, query_set, 0, 4, without_query_resolve, 0);

        let aligned_destination = create_buffer(
            test.device(),
            256 + 3 * 8,
            native::WGPUBufferUsage_QueryResolve,
        );
        finish_resolve_ok(&test, query_set, 1, 3, aligned_destination, 256);
        finish_resolve_error(&test, query_set, 0, 4, aligned_destination, 128);
        finish_resolve_error(&test, query_set, 0, 4, aligned_destination, 256);

        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(without_query_resolve);
        yawgpu::wgpuBufferRelease(aligned_destination);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn render_pass_occlusion_query_pairing_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(test.device());
        let color_attachment = make_color_attachment(color.view);
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);

        finish_occlusion_pass_ok(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        finish_occlusion_pass_error(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
        });

        finish_occlusion_pass_error(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        finish_occlusion_pass_error(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 1);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        finish_occlusion_pass_error(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 2);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        finish_occlusion_pass_error(&test, &color_attachment, None, |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        finish_occlusion_pass_error(&test, &color_attachment, Some(query_set), |pass| {
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        });

        yawgpu::wgpuQuerySetRelease(query_set);
        release_view(color);
    }
}

unsafe fn create_query_set(
    device: native::WGPUDevice,
    query_type: native::WGPUQueryType,
    count: u32,
) -> native::WGPUQuerySet {
    let descriptor = native::WGPUQuerySetDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        type_: query_type,
        count,
    };
    yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor)
}

unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn finish_resolve_ok(
    test: &ValidationTest,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
) {
    let encoder = create_encoder(test.device());
    yawgpu::wgpuCommandEncoderResolveQuerySet(
        encoder,
        query_set,
        first_query,
        query_count,
        destination,
        destination_offset,
    );
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_resolve_error(
    test: &ValidationTest,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
) {
    let encoder = create_encoder(test.device());
    yawgpu::wgpuCommandEncoderResolveQuerySet(
        encoder,
        query_set,
        first_query,
        query_count,
        destination,
        destination_offset,
    );
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

#[derive(Clone, Copy)]
struct ViewResource {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn create_view(device: native::WGPUDevice) -> ViewResource {
    let texture_descriptor = native::WGPUTextureDescriptor {
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
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor);
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    ViewResource { texture, view }
}

unsafe fn release_view(resource: ViewResource) {
    yawgpu::wgpuTextureViewRelease(resource.view);
    yawgpu::wgpuTextureRelease(resource.texture);
}

fn make_color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
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

fn render_pass_descriptor(
    color_attachments: &[native::WGPURenderPassColorAttachment],
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: color_attachments.len(),
        colorAttachments: color_attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

fn pass_timestamp_writes(
    query_set: native::WGPUQuerySet,
    beginning_index: u32,
    end_index: u32,
) -> native::WGPUPassTimestampWrites {
    native::WGPUPassTimestampWrites {
        nextInChain: std::ptr::null_mut(),
        querySet: query_set,
        beginningOfPassWriteIndex: beginning_index,
        endOfPassWriteIndex: end_index,
    }
}

unsafe fn finish_render_pass_ok(
    fixture: &FeatureDeviceFixture,
    descriptor: &native::WGPURenderPassDescriptor,
) {
    let encoder = create_encoder(fixture.device);
    fixture.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor);
    assert!(!pass.is_null());
    assert!(fixture.errors());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    finish_encoder_ok(fixture, encoder);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

unsafe fn finish_render_pass_error(
    fixture: &FeatureDeviceFixture,
    descriptor: &native::WGPURenderPassDescriptor,
) {
    let encoder = create_encoder(fixture.device);
    fixture.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor);
    assert!(!pass.is_null());
    assert!(fixture.errors());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    finish_encoder_error(fixture, encoder);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

unsafe fn finish_encoder_ok(fixture: &FeatureDeviceFixture, encoder: native::WGPUCommandEncoder) {
    fixture.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(fixture.errors());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_encoder_error(
    fixture: &FeatureDeviceFixture,
    encoder: native::WGPUCommandEncoder,
) {
    fixture.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(!fixture.errors());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_occlusion_pass_ok<F>(
    test: &ValidationTest,
    color_attachment: &native::WGPURenderPassColorAttachment,
    query_set: Option<native::WGPUQuerySet>,
    commands: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test.device());
    let mut descriptor = render_pass_descriptor(std::slice::from_ref(color_attachment));
    descriptor.occlusionQuerySet = query_set.unwrap_or(std::ptr::null());
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_occlusion_pass_error<F>(
    test: &ValidationTest,
    color_attachment: &native::WGPURenderPassColorAttachment,
    query_set: Option<native::WGPUQuerySet>,
    commands: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test.device());
    let mut descriptor = render_pass_descriptor(std::slice::from_ref(color_attachment));
    descriptor.occlusionQuerySet = query_set.unwrap_or(std::ptr::null());
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

struct FeatureDeviceFixture {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl FeatureDeviceFixture {
    unsafe fn new(required_features: &[native::WGPUFeatureName]) -> Self {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());

        let mut adapter: native::WGPUAdapter = std::ptr::null();
        let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future =
            yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
        wait(instance, future);
        assert!(!adapter.is_null());

        let descriptor = native::WGPUDeviceDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            requiredFeatureCount: required_features.len(),
            requiredFeatures: required_features.as_ptr(),
            requiredLimits: std::ptr::null(),
            defaultQueue: queue_descriptor_init(),
            deviceLostCallbackInfo: unsafe { std::mem::zeroed() },
            uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
        };
        let mut device: native::WGPUDevice = std::ptr::null();
        let device_callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_device_callback),
            userdata1: (&mut device as *mut native::WGPUDevice).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, device_callback_info);
        wait(instance, future);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        Self {
            instance,
            adapter,
            device,
            errors,
        }
    }

    fn errors(&self) -> bool {
        self.errors.lock().expect("error lock").is_empty()
    }

    fn clear_errors(&self) {
        self.errors.lock().expect("error lock").clear();
    }
}

impl Drop for FeatureDeviceFixture {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuDeviceRelease(self.device);
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    *(userdata1 as *mut native::WGPUAdapter) = adapter;
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

fn queue_descriptor_init() -> native::WGPUQueueDescriptor {
    native::WGPUQueueDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
    }
}

fn supported_features_init() -> native::WGPUSupportedFeatures {
    native::WGPUSupportedFeatures {
        featureCount: 0,
        features: std::ptr::null(),
    }
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
