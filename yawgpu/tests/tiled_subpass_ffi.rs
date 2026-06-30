#![cfg(feature = "tiled")]

use yawgpu::{
    native, YaWGPUAttachmentLayout, YaWGPUInputAttachmentBindingLayout, YaWGPUSubpassLayout,
    YaWGPUSubpassPassLayoutDescriptor, YaWGPUTiledCapabilities,
    YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT,
};
use yawgpu_test::ValidationTest;

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn valid_subpass_layout_descriptor(
    color: &YaWGPUAttachmentLayout,
    color_index: &u32,
    subpass: &YaWGPUSubpassLayout,
) -> YaWGPUSubpassPassLayoutDescriptor {
    let _ = color_index;
    YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: color,
        colorAttachmentCount: 1,
        depthStencilAttachment: std::ptr::null(),
        subpasses: subpass,
        subpassCount: 1,
        dependencies: std::ptr::null(),
        dependencyCount: 0,
    }
}

#[test]
fn noop_tiled_capabilities_returns_success_and_zero_caps() {
    let test = ValidationTest::new();

    unsafe {
        let mut caps = YaWGPUTiledCapabilities {
            nextInChain: std::ptr::null(),
            maxSubpasses: u32::MAX,
            maxSubpassColorAttachments: u32::MAX,
            maxInputAttachments: u32::MAX,
            estimatedTileMemoryBytes: u32::MAX,
        };
        let status = yawgpu::yawgpuAdapterGetTiledCapabilities(test.adapter(), &mut caps);

        assert_eq!(status, native::WGPUStatus_Success);
        assert_eq!(caps.maxSubpasses, 0);
        assert_eq!(caps.maxSubpassColorAttachments, 0);
        assert_eq!(caps.maxInputAttachments, 0);
        assert_eq!(caps.estimatedTileMemoryBytes, 0);
    }
}

#[test]
fn noop_create_subpass_pass_layout_returns_handle_and_refcounts() {
    let test = ValidationTest::new();
    let color = YaWGPUAttachmentLayout {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sampleCount: 1,
    };
    let color_index = 0;
    let subpass = YaWGPUSubpassLayout {
        colorAttachmentIndices: &color_index,
        colorAttachmentIndexCount: 1,
        usesDepthStencil: 0,
        inputAttachments: std::ptr::null(),
        inputAttachmentCount: 0,
    };
    let descriptor = valid_subpass_layout_descriptor(&color, &color_index, &subpass);

    test.expect_no_validation_error(|| unsafe {
        let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(test.device(), &descriptor);
        assert!(!layout.is_null());
        yawgpu::yawgpuSubpassPassLayoutAddRef(layout);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
    });
}

#[test]
fn malformed_subpass_pass_layout_routes_device_error() {
    let test = ValidationTest::new();
    let descriptor = YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: std::ptr::null(),
        colorAttachmentCount: 0,
        depthStencilAttachment: std::ptr::null(),
        subpasses: std::ptr::null(),
        subpassCount: 0,
        dependencies: std::ptr::null(),
        dependencyCount: 0,
    };

    test.assert_device_error_after(
        || unsafe {
            let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(test.device(), &descriptor);
            assert!(!layout.is_null());
            yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        },
        Some("requires at least one subpass"),
    );
}

#[test]
fn input_attachment_bind_group_layout_entry_is_accepted_by_c_api() {
    let test = ValidationTest::new();
    let mut input = YaWGPUInputAttachmentBindingLayout {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT,
        },
        sampleType: native::WGPUTextureSampleType_Float,
        multisampled: 0,
    };
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: (&mut input.chain) as *mut native::WGPUChainedStruct,
        binding: 3,
        visibility: native::WGPUShaderStage_Fragment,
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
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
        bindingArraySize: 0,
    };
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &entry,
    };

    test.expect_no_validation_error(|| unsafe {
        let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(test.device(), &descriptor);
        assert!(!layout.is_null());
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    });
}
