//! Real-Metal surface e2e: configuring with the `webgpu.h` INIT zero
//! sentinels (`WGPUPresentMode_Undefined`, `WGPUCompositeAlphaMode_Auto`)
//! must succeed and acquire a texture from a standalone `CAMetalLayer`
//! (externally reported 2026-08-09; contract in
//! `specs/blocks/70-finalize.md` → Surface → Sentinel resolution), and —
//! Block 103 (`specs/blocks/103-surface-capabilities.md`) — the surface
//! capabilities come from the Metal HAL (Dawn's list), so sRGB formats,
//! `Immediate` / `Mailbox`, `Premultiplied` and `CopySrc` configure, while
//! modes outside the list keep being rejected.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use objc2_quartz_core::CAMetalLayer;
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_with_init_sentinels_acquires_texture() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );

        // The repro's configuration: capabilities-reported format, render
        // attachment usage, nonzero size, and both modes left at the INIT
        // zero sentinels.
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Auto,
            presentMode: native::WGPUPresentMode_Undefined,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        );
        assert!(!surface_texture.texture.is_null());
        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        yawgpu::wgpuTextureRelease(surface_texture.texture.cast_mut());

        yawgpu::wgpuSurfaceUnconfigure(surface);
        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Block 103: the C ABI reports Dawn's Metal surface capabilities.
#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_capabilities_match_dawn_metal_list() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );
        let mut caps: native::WGPUSurfaceCapabilities = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(surface, adapter, &mut caps),
            native::WGPUStatus_Success
        );
        let formats = std::slice::from_raw_parts(caps.formats, caps.formatCount);
        let present_modes = std::slice::from_raw_parts(caps.presentModes, caps.presentModeCount);
        let alpha_modes = std::slice::from_raw_parts(caps.alphaModes, caps.alphaModeCount);
        assert_eq!(
            formats,
            &[
                native::WGPUTextureFormat_BGRA8Unorm,
                native::WGPUTextureFormat_BGRA8UnormSrgb,
                native::WGPUTextureFormat_RGBA16Float,
                native::WGPUTextureFormat_RGB10A2Unorm,
            ]
        );
        assert_eq!(
            present_modes,
            &[
                native::WGPUPresentMode_Fifo,
                native::WGPUPresentMode_Immediate,
                native::WGPUPresentMode_Mailbox,
            ]
        );
        assert_eq!(
            alpha_modes,
            &[
                native::WGPUCompositeAlphaMode_Opaque,
                native::WGPUCompositeAlphaMode_Premultiplied,
            ]
        );
        assert_eq!(
            caps.usages,
            native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst
        );
        yawgpu::wgpuSurfaceCapabilitiesFreeMembers(caps);

        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Block 103: an sRGB format, `Immediate`, `Premultiplied` and `CopySrc`
/// all configure; the acquired texture can be cleared and read back.
#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_srgb_immediate_premultiplied_and_copy_back() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8UnormSrgb,
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Premultiplied,
            presentMode: native::WGPUPresentMode_Immediate,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "configure errors: {:?}",
            errors.lock().expect("error lock")
        );

        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        );
        let texture = surface_texture.texture.cast_mut();
        assert!(!texture.is_null());
        assert_eq!(
            yawgpu::wgpuTextureGetFormat(texture),
            native::WGPUTextureFormat_BGRA8UnormSrgb
        );

        // Clear to opaque red and read the top-left texel back through CopySrc.
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let readback = create_buffer(
            device,
            256 * 64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_clear_render_pass(encoder, view, [1.0, 0.0, 0.0, 1.0]);
        record_texture_to_buffer(encoder, texture, readback, 64, 64);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let bytes = read_buffer(instance, readback, 4);
        // BGRA: blue, green, red, alpha.
        assert_eq!(bytes, vec![0, 0, 255, 255], "sRGB clear-to-red round trip");
        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuSurfaceUnconfigure(surface);
        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Block 103: `Rgba16Float` configures and presents.
/// Block 104 (Phase Review M2 mirror): the acquired drawable is lazy-init
/// eligible, so a `CopySrc` readback before any render pass reads zeros.
#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_texture_reads_zero_before_first_render_pass() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Opaque,
            presentMode: native::WGPUPresentMode_Fifo,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        );
        let texture = surface_texture.texture.cast_mut();
        assert!(!texture.is_null());

        // No render pass: the copy is the first use of the acquired drawable.
        let readback = create_buffer(
            device,
            256 * 64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_texture_to_buffer(encoder, texture, readback, 64, 64);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let bytes = read_buffer(instance, readback, 256 * 64);
        let nonzero = bytes.iter().filter(|&&b| b != 0).count();
        assert_eq!(
            nonzero, 0,
            "uninitialized surface texture must read zero ({nonzero} non-zero bytes)"
        );
        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuSurfaceUnconfigure(surface);
        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_rgba16float_presents() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_RGBA16Float,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 32,
            height: 32,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Auto,
            presentMode: native::WGPUPresentMode_Mailbox,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        );
        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );
        yawgpu::wgpuTextureRelease(surface_texture.texture.cast_mut());
        yawgpu::wgpuSurfaceUnconfigure(surface);
        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Block 103: modes outside the reported capabilities are still rejected
/// with the Block 70 messages, and a rejected configure leaves the surface
/// unconfigured.
#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_rejects_modes_outside_capabilities() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );
        let mut config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Opaque,
            presentMode: native::WGPUPresentMode_FifoRelaxed,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        config.presentMode = native::WGPUPresentMode_Fifo;
        config.alphaMode = native::WGPUCompositeAlphaMode_Unpremultiplied;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        config.alphaMode = native::WGPUCompositeAlphaMode_Opaque;
        config.format = native::WGPUTextureFormat_RGBA8Unorm;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        config.format = native::WGPUTextureFormat_BGRA8Unorm;
        config.usage = native::WGPUTextureUsage_StorageBinding;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        let captured = errors.lock().expect("error lock");
        let messages: Vec<&str> = captured
            .iter()
            .map(|error| error.message.as_str())
            .collect();
        assert_eq!(
            messages,
            vec![
                "surface configuration present mode is not supported",
                "surface configuration alpha mode is not supported",
                "surface configuration format is not supported",
                "surface configuration usage is not supported",
            ]
        );
        drop(captured);

        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_Error
        );
        assert!(surface_texture.texture.is_null());

        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured_errors = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured_errors.lock().expect("error lock").push(error)),
    );
    errors
}

unsafe fn record_clear_render_pass(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
    clear: [f64; 4],
) {
    let attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: clear[0],
            g: clear[1],
            b: clear[2],
            a: clear[3],
        },
    };
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 1,
        colorAttachments: &attachment,
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

unsafe fn record_texture_to_buffer(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
    width: u32,
    height: u32,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: 256,
            rowsPerImage: height,
        },
        buffer,
    };
    let extent = native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
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

unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

unsafe fn create_surface_from_layer(
    instance: native::WGPUInstance,
    layer: *mut c_void,
) -> native::WGPUSurface {
    let mut source = native::WGPUSurfaceSourceMetalLayer {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_SurfaceSourceMetalLayer,
        },
        layer,
    };
    let descriptor = native::WGPUSurfaceDescriptor {
        nextInChain: (&mut source.chain) as *mut _,
        label: empty_string_view(),
    };
    let surface = yawgpu::wgpuInstanceCreateSurface(instance, &descriptor);
    assert!(!surface.is_null());
    surface
}

unsafe fn create_metal_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_METAL,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    instance
}

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!adapter.is_null());
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
