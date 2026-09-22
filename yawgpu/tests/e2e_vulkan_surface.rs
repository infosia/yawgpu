//! Real-Vulkan surface e2e on macOS (MoltenVK via CAMetalLayer) and Windows
//! (the native driver via a hidden Win32 window) for
//! Block 103 (`specs/blocks/103-surface-capabilities.md`): the surface
//! capabilities are the driver's (`vkGetPhysicalDeviceSurface*`), a
//! reported non-`Fifo` present mode configures, `CopySrc` reads back, and
//! modes outside the reported set are rejected.

#![cfg(all(feature = "vulkan", any(target_os = "macos", windows)))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
#[cfg(target_os = "macos")]
use objc2_quartz_core::CAMetalLayer;
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_surface_configure_with_init_sentinels_acquires_texture() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let (surface, _window) = create_test_surface(instance);

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

/// Block 103: the C ABI reports the driver's Vulkan surface capabilities.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_surface_capabilities_are_driver_reported() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let (surface, _window) = create_test_surface(instance);
        let mut caps: native::WGPUSurfaceCapabilities = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(surface, adapter, &mut caps),
            native::WGPUStatus_Success
        );
        let formats = std::slice::from_raw_parts(caps.formats, caps.formatCount);
        let present_modes = std::slice::from_raw_parts(caps.presentModes, caps.presentModeCount);
        let alpha_modes = std::slice::from_raw_parts(caps.alphaModes, caps.alphaModeCount);
        assert!(
            !formats.is_empty(),
            "driver must report at least one format"
        );
        assert!(
            formats.contains(&native::WGPUTextureFormat_BGRA8Unorm),
            "the driver reports BGRA8Unorm: {formats:?}"
        );
        assert!(
            present_modes.contains(&native::WGPUPresentMode_Fifo),
            "Fifo is mandatory: {present_modes:?}"
        );
        assert!(
            alpha_modes.contains(&native::WGPUCompositeAlphaMode_Opaque),
            "the driver reports Opaque: {alpha_modes:?}"
        );
        assert!(caps.usages & native::WGPUTextureUsage_RenderAttachment != 0);
        assert!(caps.usages & native::WGPUTextureUsage_CopySrc != 0);
        eprintln!("vulkan surface caps: formats={formats:?} present={present_modes:?} alpha={alpha_modes:?} usages={:#x}", caps.usages);
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
fn vulkan_surface_configure_reported_srgb_and_non_fifo_modes_and_copy_back() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let (surface, _window) = create_test_surface(instance);
        let (format, present_mode, alpha_mode) = pick_reported_modes(surface, adapter);
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format,
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: alpha_mode,
            presentMode: present_mode,
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
        assert_eq!(yawgpu::wgpuTextureGetFormat(texture), format);

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
        let expected = if format == native::WGPUTextureFormat_RGBA8Unorm
            || format == native::WGPUTextureFormat_RGBA8UnormSrgb
        {
            vec![255, 0, 0, 255]
        } else {
            vec![0, 0, 255, 255]
        };
        assert_eq!(bytes, expected, "clear-to-red round trip for {format}");
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

/// Block 104 (Phase Review M2): a swapchain texture is lazy-init eligible,
/// so reading it back through `CopySrc` before any render pass touches it
/// must yield zeros — the clear needs `TRANSFER_DST` on the swapchain image.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_surface_texture_reads_zero_before_first_render_pass() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let (surface, _window) = create_test_surface(instance);
        let (format, present_mode, alpha_mode) = pick_reported_modes(surface, adapter);
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format,
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: alpha_mode,
            presentMode: present_mode,
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

        // No render pass: the copy is the first use of the acquired image.
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

/// Block 103: modes outside the reported capabilities are still rejected
/// with the Block 70 messages, and a rejected configure leaves the surface
/// unconfigured.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_surface_configure_rejects_modes_outside_capabilities() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let (surface, _window) = create_test_surface(instance);
        // Reject a depth format, an unknown usage bit, and any present / alpha
        // mode that the driver leaves unreported.
        let (present_mode, alpha_mode) = pick_unreported_modes(surface, adapter);
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
            presentMode: native::WGPUPresentMode_Fifo,
        };
        let mut expected_messages = Vec::new();
        if let Some(present_mode) = present_mode {
            config.presentMode = present_mode;
            yawgpu::wgpuSurfaceConfigure(surface, &config);
            expected_messages.push("surface configuration present mode is not supported");
            config.presentMode = native::WGPUPresentMode_Fifo;
        }
        if let Some(alpha_mode) = alpha_mode {
            config.alphaMode = alpha_mode;
            yawgpu::wgpuSurfaceConfigure(surface, &config);
            expected_messages.push("surface configuration alpha mode is not supported");
            config.alphaMode = native::WGPUCompositeAlphaMode_Opaque;
        }
        config.format = native::WGPUTextureFormat_Depth32Float;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        expected_messages.push("surface configuration format is not supported");
        config.format = native::WGPUTextureFormat_BGRA8Unorm;
        config.usage = native::WGPUTextureUsage_RenderAttachment | 0x8000_0000;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        expected_messages.push("surface configuration usage is not supported");
        let captured = errors.lock().expect("error lock");
        let messages: Vec<&str> = captured
            .iter()
            .map(|error| error.message.as_str())
            .collect();
        assert_eq!(messages, expected_messages);
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

/// Picks a driver-reported (format, present mode, alpha mode) triple,
/// preferring an sRGB format, a non-`Fifo` present mode and a non-`Opaque`
/// alpha mode when the driver offers them.
unsafe fn pick_reported_modes(
    surface: native::WGPUSurface,
    adapter: native::WGPUAdapter,
) -> (
    native::WGPUTextureFormat,
    native::WGPUPresentMode,
    native::WGPUCompositeAlphaMode,
) {
    let mut caps: native::WGPUSurfaceCapabilities = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuSurfaceGetCapabilities(surface, adapter, &mut caps),
        native::WGPUStatus_Success
    );
    let formats = std::slice::from_raw_parts(caps.formats, caps.formatCount);
    let present_modes = std::slice::from_raw_parts(caps.presentModes, caps.presentModeCount);
    let alpha_modes = std::slice::from_raw_parts(caps.alphaModes, caps.alphaModeCount);
    let format = formats
        .iter()
        .copied()
        .find(|&f| {
            f == native::WGPUTextureFormat_BGRA8UnormSrgb
                || f == native::WGPUTextureFormat_RGBA8UnormSrgb
        })
        .or_else(|| {
            formats.iter().copied().find(|&f| {
                f == native::WGPUTextureFormat_BGRA8Unorm
                    || f == native::WGPUTextureFormat_RGBA8Unorm
            })
        })
        .expect("an 8-bit surface format");
    let present_mode = present_modes
        .iter()
        .copied()
        .find(|&m| m != native::WGPUPresentMode_Fifo)
        .unwrap_or(native::WGPUPresentMode_Fifo);
    let alpha_mode = alpha_modes
        .iter()
        .copied()
        .find(|&m| m != native::WGPUCompositeAlphaMode_Opaque)
        .unwrap_or(native::WGPUCompositeAlphaMode_Opaque);
    yawgpu::wgpuSurfaceCapabilitiesFreeMembers(caps);
    (format, present_mode, alpha_mode)
}

/// Picks an unreported mode of each kind, if the driver leaves one absent.
unsafe fn pick_unreported_modes(
    surface: native::WGPUSurface,
    adapter: native::WGPUAdapter,
) -> (
    Option<native::WGPUPresentMode>,
    Option<native::WGPUCompositeAlphaMode>,
) {
    let mut caps: native::WGPUSurfaceCapabilities = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuSurfaceGetCapabilities(surface, adapter, &mut caps),
        native::WGPUStatus_Success
    );
    let present_modes = std::slice::from_raw_parts(caps.presentModes, caps.presentModeCount);
    let alpha_modes = std::slice::from_raw_parts(caps.alphaModes, caps.alphaModeCount);
    let present_mode = [
        native::WGPUPresentMode_Mailbox,
        native::WGPUPresentMode_Immediate,
        native::WGPUPresentMode_FifoRelaxed,
    ]
    .into_iter()
    .find(|m| !present_modes.contains(m));
    let alpha_mode = [
        native::WGPUCompositeAlphaMode_Unpremultiplied,
        native::WGPUCompositeAlphaMode_Premultiplied,
        native::WGPUCompositeAlphaMode_Inherit,
    ]
    .into_iter()
    .find(|m| !alpha_modes.contains(m));
    yawgpu::wgpuSurfaceCapabilitiesFreeMembers(caps);
    (present_mode, alpha_mode)
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

/// Owns the native surface source until after `wgpuSurfaceRelease`.
struct TestSurfaceWindow {
    #[cfg(target_os = "macos")]
    _layer: Retained<CAMetalLayer>,
    #[cfg(windows)]
    hwnd: windows_sys::Win32::Foundation::HWND,
    #[cfg(windows)]
    hinstance: windows_sys::Win32::Foundation::HINSTANCE,
    #[cfg(windows)]
    class_name: Vec<u16>,
}

#[cfg(windows)]
impl Drop for TestSurfaceWindow {
    fn drop(&mut self) {
        use windows_sys::Win32::UI::WindowsAndMessaging::{DestroyWindow, UnregisterClassW};

        unsafe {
            if !self.hwnd.is_null() {
                let _ = DestroyWindow(self.hwnd);
            }
            let _ = UnregisterClassW(self.class_name.as_ptr(), self.hinstance);
        }
    }
}

unsafe fn create_test_surface(
    instance: native::WGPUInstance,
) -> (native::WGPUSurface, TestSurfaceWindow) {
    #[cfg(target_os = "macos")]
    let (mut source, window) = {
        // A bare CAMetalLayer has a zero drawable size; MoltenVK derives the
        // swapchain extent from it, so size it like a 64x64 window first.
        let layer = CAMetalLayer::layer();
        layer.setBounds(CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 64.0,
                height: 64.0,
            },
        });
        layer.setContentsScale(1.0);
        layer.setDrawableSize(CGSize {
            width: 64.0,
            height: 64.0,
        });
        let source = native::WGPUSurfaceSourceMetalLayer {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_SurfaceSourceMetalLayer,
            },
            layer: (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        };
        (source, TestSurfaceWindow { _layer: layer })
    };
    #[cfg(windows)]
    let (mut source, window) = {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, RegisterClassExW, WNDCLASSEXW, WS_POPUP,
        };

        static NEXT_CLASS_ID: AtomicUsize = AtomicUsize::new(0);
        let hinstance = GetModuleHandleW(std::ptr::null());
        assert!(!hinstance.is_null(), "GetModuleHandleW failed");
        let class_id = NEXT_CLASS_ID.fetch_add(1, Ordering::Relaxed);
        let class_name: Vec<u16> = format!("yawgpu_vulkan_surface_{class_id}\0")
            .encode_utf16()
            .collect();
        let class = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(DefWindowProcW),
            hInstance: hinstance,
            lpszClassName: class_name.as_ptr(),
            ..std::mem::zeroed()
        };
        assert_ne!(RegisterClassExW(&class), 0, "RegisterClassExW failed");
        let mut window = TestSurfaceWindow {
            hwnd: std::ptr::null_mut(),
            hinstance,
            class_name,
        };
        // WS_POPUP has no frame: the hidden window's client area is 64x64.
        window.hwnd = CreateWindowExW(
            0,
            window.class_name.as_ptr(),
            window.class_name.as_ptr(),
            WS_POPUP,
            0,
            0,
            64,
            64,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            hinstance,
            std::ptr::null(),
        );
        assert!(!window.hwnd.is_null(), "CreateWindowExW failed");
        let source = native::WGPUSurfaceSourceWindowsHWND {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_SurfaceSourceWindowsHWND,
            },
            hinstance,
            hwnd: window.hwnd,
        };
        (source, window)
    };
    let descriptor = native::WGPUSurfaceDescriptor {
        nextInChain: (&mut source.chain) as *mut _,
        label: empty_string_view(),
    };
    let surface = yawgpu::wgpuInstanceCreateSurface(instance, &descriptor);
    assert!(!surface.is_null());
    (surface, window)
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
