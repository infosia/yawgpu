use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn create_surface_decodes_descriptor_chain_and_returns_error_surface_for_invalid_descriptors() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface(test.instance());
        assert!(!surface.is_null());
        yawgpu::wgpuSurfaceAddRef(surface);
        yawgpu::wgpuSurfaceSetLabel(surface, string_view("surface"));
        yawgpu::wgpuSurfaceRelease(surface);
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );

        let missing_source = native::WGPUSurfaceDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        };
        let error_surface = yawgpu::wgpuInstanceCreateSurface(test.instance(), &missing_source);
        assert!(!error_surface.is_null());
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(
                error_surface,
                test.adapter(),
                &mut empty_capabilities()
            ),
            native::WGPUStatus_Error
        );

        let null_descriptor = yawgpu::wgpuInstanceCreateSurface(test.instance(), std::ptr::null());
        assert!(!null_descriptor.is_null());
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(
                null_descriptor,
                test.adapter(),
                &mut empty_capabilities()
            ),
            native::WGPUStatus_Error
        );

        yawgpu::wgpuSurfaceRelease(null_descriptor);
        yawgpu::wgpuSurfaceRelease(error_surface);
        yawgpu::wgpuSurfaceRelease(surface);
    }
}

#[test]
fn create_surface_accepts_windows_hwnd_source_on_noop() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface_from_windows_hwnd(test.instance(), std::ptr::dangling_mut());
        assert!(!surface.is_null());
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(surface, test.adapter(), &mut empty_capabilities()),
            native::WGPUStatus_Success
        );

        yawgpu::wgpuSurfaceRelease(surface);
    }
}

#[test]
fn create_surface_with_null_windows_hwnd_source_does_not_panic_on_noop() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface_from_windows_hwnd(test.instance(), std::ptr::null_mut());
        assert!(!surface.is_null());
        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(surface, test.adapter(), &mut empty_capabilities()),
            native::WGPUStatus_Success
        );

        yawgpu::wgpuSurfaceRelease(surface);
    }
}

#[test]
fn get_capabilities_returns_synthetic_noop_values_and_free_members_is_safe() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface(test.instance());
        let mut capabilities = empty_capabilities();

        assert_eq!(
            yawgpu::wgpuSurfaceGetCapabilities(surface, test.adapter(), &mut capabilities),
            native::WGPUStatus_Success
        );
        assert_eq!(
            capabilities.usages,
            native::WGPUTextureUsage_RenderAttachment
        );
        let formats = std::slice::from_raw_parts(capabilities.formats, capabilities.formatCount);
        assert!(formats.contains(&native::WGPUTextureFormat_BGRA8Unorm));
        assert!(formats.contains(&native::WGPUTextureFormat_RGBA8Unorm));
        let present_modes =
            std::slice::from_raw_parts(capabilities.presentModes, capabilities.presentModeCount);
        assert_eq!(present_modes, &[native::WGPUPresentMode_Fifo]);
        let alpha_modes =
            std::slice::from_raw_parts(capabilities.alphaModes, capabilities.alphaModeCount);
        assert_eq!(alpha_modes, &[native::WGPUCompositeAlphaMode_Opaque]);

        yawgpu::wgpuSurfaceCapabilitiesFreeMembers(capabilities);
        yawgpu::wgpuSurfaceRelease(surface);
    }
}

#[test]
fn configure_validates_format_usage_size_and_keeps_surface_unconfigured_on_error() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface(test.instance());

        let mut bad_format = valid_config(test.device());
        bad_format.format = native::WGPUTextureFormat_R8Unorm;
        assert_configure_error(&test, surface, &bad_format);
        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Error);

        let mut bad_usage = valid_config(test.device());
        bad_usage.usage =
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_StorageBinding;
        assert_configure_error(&test, surface, &bad_usage);
        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Error);

        let mut zero_size = valid_config(test.device());
        zero_size.width = 0;
        assert_configure_error(&test, surface, &zero_size);
        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Error);

        test.clear_errors();
        let config = valid_config(test.device());
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        yawgpu::wgpuSurfaceRelease(surface);
    }
}

#[test]
fn current_texture_reports_unconfigured_and_noop_presentation_boundary() {
    let test = ValidationTest::new();
    unsafe {
        let surface = create_surface(test.instance());

        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Error);

        let config = valid_config(test.device());
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Lost);

        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );

        yawgpu::wgpuSurfaceUnconfigure(surface);
        assert_current_texture_status(surface, native::WGPUSurfaceGetCurrentTextureStatus_Error);
        yawgpu::wgpuSurfaceRelease(surface);
    }
}

unsafe fn assert_configure_error(
    test: &ValidationTest,
    surface: native::WGPUSurface,
    config: &native::WGPUSurfaceConfiguration,
) {
    test.assert_device_error_after(
        || {
            yawgpu::wgpuSurfaceConfigure(surface, config);
        },
        None,
    );
}

unsafe fn assert_current_texture_status(
    surface: native::WGPUSurface,
    expected: native::WGPUSurfaceGetCurrentTextureStatus,
) {
    let mut surface_texture = native::WGPUSurfaceTexture {
        nextInChain: std::ptr::null_mut(),
        texture: std::ptr::null(),
        status: 0,
    };
    yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
    assert_eq!(surface_texture.status, expected);
    assert!(surface_texture.texture.is_null());
}

unsafe fn create_surface(instance: native::WGPUInstance) -> native::WGPUSurface {
    let mut source = native::WGPUSurfaceSourceMetalLayer {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_SurfaceSourceMetalLayer,
        },
        layer: std::ptr::dangling_mut(),
    };
    let descriptor = native::WGPUSurfaceDescriptor {
        nextInChain: (&mut source.chain) as *mut _,
        label: empty_string_view(),
    };
    let surface = yawgpu::wgpuInstanceCreateSurface(instance, &descriptor);
    assert!(!surface.is_null());
    surface
}

unsafe fn create_surface_from_windows_hwnd(
    instance: native::WGPUInstance,
    hwnd: *mut std::ffi::c_void,
) -> native::WGPUSurface {
    let mut source = native::WGPUSurfaceSourceWindowsHWND {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_SurfaceSourceWindowsHWND,
        },
        hinstance: std::ptr::null_mut(),
        hwnd,
    };
    let descriptor = native::WGPUSurfaceDescriptor {
        nextInChain: (&mut source.chain) as *mut _,
        label: empty_string_view(),
    };
    let surface = yawgpu::wgpuInstanceCreateSurface(instance, &descriptor);
    assert!(!surface.is_null());
    surface
}

fn valid_config(device: native::WGPUDevice) -> native::WGPUSurfaceConfiguration {
    native::WGPUSurfaceConfiguration {
        nextInChain: std::ptr::null_mut(),
        device,
        format: native::WGPUTextureFormat_BGRA8Unorm,
        usage: native::WGPUTextureUsage_RenderAttachment,
        width: 640,
        height: 480,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
        alphaMode: native::WGPUCompositeAlphaMode_Opaque,
        presentMode: native::WGPUPresentMode_Fifo,
    }
}

fn empty_capabilities() -> native::WGPUSurfaceCapabilities {
    native::WGPUSurfaceCapabilities {
        nextInChain: std::ptr::null_mut(),
        usages: native::WGPUTextureUsage_None,
        formatCount: 0,
        formats: std::ptr::null(),
        presentModeCount: 0,
        presentModes: std::ptr::null(),
        alphaModeCount: 0,
        alphaModes: std::ptr::null(),
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
