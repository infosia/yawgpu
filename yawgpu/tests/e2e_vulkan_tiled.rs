#![cfg(all(feature = "vulkan", feature = "tiled"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YaWGPUTiledCapabilities, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_tiled_features_and_capabilities_are_advertised() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        assert!(!adapter.is_null());

        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_MultiSubpass),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_TransientAttachments,),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                adapter,
                yawgpu::YaWGPUFeatureName_ShaderFramebufferFetch,
            ),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                adapter,
                yawgpu::YaWGPUFeatureName_ProgrammableTileDispatch,
            ),
            1
        );

        let mut capabilities = zeroed_tiled_capabilities();
        assert_eq!(
            yawgpu::yawgpuAdapterGetTiledCapabilities(adapter, &mut capabilities),
            native::WGPUStatus_Success
        );
        assert!(capabilities.maxSubpasses > 0);
        assert!(capabilities.maxSubpassColorAttachments > 0);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_explicit_transient_attachment_allocates_without_device_error() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );
        let descriptor = yawgpu::YaWGPUTransientAttachmentDescriptor {
            nextInChain: std::ptr::null(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sizeMode: yawgpu::YaWGPUTransientSizeMode_Explicit,
            width: 16,
            height: 16,
            sampleCount: 1,
        };

        let attachment = yawgpu::yawgpuDeviceCreateTransientAttachment(device, &descriptor);
        assert!(!attachment.is_null());
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::yawgpuTransientAttachmentRelease(attachment);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_clear_only_subpass_pass_submits_without_device_error() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let texture = create_color_texture(device);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let layout = create_single_color_subpass_layout(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_clear_subpass_pass(encoder, layout, view);
        submit_encoder(queue, encoder);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_vulkan_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_VULKAN,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    yawgpu::wgpuCreateInstance(&descriptor)
}

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    adapter
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
    device
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

fn zeroed_tiled_capabilities() -> YaWGPUTiledCapabilities {
    YaWGPUTiledCapabilities {
        nextInChain: std::ptr::null(),
        maxSubpasses: 0,
        maxSubpassColorAttachments: 0,
        maxInputAttachments: 0,
        estimatedTileMemoryBytes: 0,
    }
}

unsafe fn create_single_color_subpass_layout(
    device: native::WGPUDevice,
) -> yawgpu::YaWGPUSubpassPassLayout {
    let color = yawgpu::YaWGPUAttachmentLayout {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sampleCount: 1,
    };
    let color_index = 0_u32;
    let subpass = yawgpu::YaWGPUSubpassLayoutDesc {
        colorAttachmentIndices: &color_index,
        colorAttachmentIndexCount: 1,
        usesDepthStencil: 0,
        inputAttachments: std::ptr::null(),
        inputAttachmentCount: 0,
    };
    let descriptor = yawgpu::YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: &color,
        colorAttachmentCount: 1,
        depthStencilAttachment: yawgpu::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_Undefined,
            sampleCount: 1,
        },
        subpasses: &subpass,
        subpassCount: 1,
        dependencies: std::ptr::null(),
        dependencyCount: 0,
    };
    let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn record_clear_subpass_pass(
    encoder: native::WGPUCommandEncoder,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    view: native::WGPUTextureView,
) {
    let color = yawgpu::YaWGPUColorAttachmentBinding {
        kind: yawgpu::YaWGPUSubpassAttachmentKind_Persistent,
        view,
        resolveTarget: std::ptr::null(),
        transient: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.25,
            g: 0.5,
            b: 0.75,
            a: 1.0,
        },
    };
    let descriptor = yawgpu::YaWGPUSubpassRenderPassDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        passLayout: layout,
        extent: texture_extent(),
        colorAttachments: &color,
        colorAttachmentCount: 1,
        depthStencilAttachment: std::ptr::null(),
    };
    let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
}

unsafe fn create_color_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 16,
        height: 16,
        depthOrArrayLayers: 1,
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
