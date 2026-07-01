#![cfg(feature = "tiled")]

use yawgpu::{
    native, YaWGPUAttachmentLayout, YaWGPUInputAttachmentBindingLayout,
    YaWGPUSubpassColorAttachment, YaWGPUSubpassDependency,
    YaWGPUSubpassDependencyType_ColorToInput, YaWGPUSubpassInputAttachment, YaWGPUSubpassLayout,
    YaWGPUSubpassPassLayoutDescriptor, YaWGPUSubpassRenderPassDescriptor,
    YaWGPUSubpassRenderPipelineDescriptor, YaWGPUTiledCapabilities,
    YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT,
};
use yawgpu_test::ValidationTest;

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

/// End-to-end proof (Noop, no GPU) that a subpass render pipeline reading a
/// **multisampled** (sampleCount 4) input attachment can be created with an
/// explicit pipeline layout whose input-attachment entry is `multisampled: 1`.
/// This exercises the relaxed shader↔layout compatibility (the Tint reflection
/// always reports `multisampled: false` for `input_attachment<T>`) together with
/// the MSAA flag derivation that lets the 2-arg `inputAttachmentLoad(g, s)`
/// compile. Creation only — no pass execution / readback.
#[test]
fn noop_msaa_subpass_input_pipeline_creates_with_explicit_multisampled_layout() {
    let test = ValidationTest::new();

    test.expect_no_validation_error(|| unsafe {
        let layout = create_two_subpass_msaa_input_layout(test.device());

        // Subpass 0: trivial MSAA write to attachment [0].
        let pipeline0 = create_msaa_write_pipeline(test.device(), layout);
        assert!(!pipeline0.is_null());

        // Subpass 1: explicit pipeline layout with an input-attachment entry at
        // @group(0) @binding(0) declared `multisampled: 1`, and a fragment that
        // reads the g-buffer per-sample via the 2-arg load.
        let bgl = create_msaa_input_bind_group_layout(test.device());
        let pipeline_layout = create_pipeline_layout(test.device(), bgl);
        let pipeline1 = create_msaa_input_read_pipeline(test.device(), layout, pipeline_layout);
        assert!(!pipeline1.is_null());

        yawgpu::wgpuRenderPipelineRelease(pipeline1);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuRenderPipelineRelease(pipeline0);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
    });
}

unsafe fn create_two_subpass_msaa_input_layout(
    device: native::WGPUDevice,
) -> yawgpu::YaWGPUSubpassPassLayout {
    let colors = [
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 4,
        },
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 4,
        },
    ];
    let subpass0_colors = [0u32];
    let subpass1_colors = [1u32];
    let input = YaWGPUSubpassInputAttachment {
        group: 0,
        binding: 0,
        sourceSubpass: 0,
        sourceAttachment: 0,
    };
    let subpasses = [
        YaWGPUSubpassLayout {
            colorAttachmentIndices: subpass0_colors.as_ptr(),
            colorAttachmentIndexCount: subpass0_colors.len(),
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        },
        YaWGPUSubpassLayout {
            colorAttachmentIndices: subpass1_colors.as_ptr(),
            colorAttachmentIndexCount: subpass1_colors.len(),
            usesDepthStencil: 0,
            inputAttachments: &input,
            inputAttachmentCount: 1,
        },
    ];
    let dependency = YaWGPUSubpassDependency {
        srcSubpass: 0,
        dstSubpass: 1,
        dependencyType: YaWGPUSubpassDependencyType_ColorToInput,
        byRegion: 1,
    };
    let descriptor = YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: colors.as_ptr(),
        colorAttachmentCount: colors.len(),
        depthStencilAttachment: std::ptr::null(),
        subpasses: subpasses.as_ptr(),
        subpassCount: subpasses.len(),
        dependencies: &dependency,
        dependencyCount: 1,
    };
    let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_msaa_input_bind_group_layout(
    device: native::WGPUDevice,
) -> native::WGPUBindGroupLayout {
    let mut input = YaWGPUInputAttachmentBindingLayout {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT,
        },
        sampleType: native::WGPUTextureSampleType_Float,
        multisampled: 1,
    };
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: (&mut input.chain) as *mut native::WGPUChainedStruct,
        binding: 0,
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
    let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!bgl.is_null());
    bgl
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    bgl: native::WGPUBindGroupLayout,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: 1,
        bindGroupLayouts: &bgl,
        immediateSize: 0,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_msaa_write_pipeline(
    device: native::WGPUDevice,
    layout: yawgpu::YaWGPUSubpassPassLayout,
) -> native::WGPURenderPipeline {
    let shader = create_wgsl_module(
        device,
        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
         @fragment fn fs() -> @location(0) vec4f { return vec4f(0.5, 0.0, 0.0, 1.0); }",
    );
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: shader,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let base = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: multisample_state_4(),
        fragment: &fragment,
    };
    let descriptor = YaWGPUSubpassRenderPipelineDescriptor {
        nextInChain: std::ptr::null(),
        base,
        passLayout: layout,
        subpassIndex: 0,
    };
    let pipeline = yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &descriptor);
    yawgpu::wgpuShaderModuleRelease(shader);
    pipeline
}

unsafe fn create_msaa_input_read_pipeline(
    device: native::WGPUDevice,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    pipeline_layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let shader = create_wgsl_module(
        device,
        "enable chromium_internal_input_attachments;
         @group(0) @binding(0) @input_attachment_index(0) var g: input_attachment<f32>;
         @vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
         @fragment fn fs(@builtin(sample_index) s: u32) -> @location(1) vec4f {
             return inputAttachmentLoad(g, s);
         }",
    );
    // Subpass 1 writes @location(1); slot 0 is unused by this pipeline.
    let targets = [
        native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Undefined,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_None,
        },
        native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        },
    ];
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: shader,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: targets.len(),
        targets: targets.as_ptr(),
    };
    let base = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: pipeline_layout,
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: multisample_state_4(),
        fragment: &fragment,
    };
    let descriptor = YaWGPUSubpassRenderPipelineDescriptor {
        nextInChain: std::ptr::null(),
        base,
        passLayout: layout,
        subpassIndex: 1,
    };
    let pipeline = yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &descriptor);
    yawgpu::wgpuShaderModuleRelease(shader);
    pipeline
}

fn multisample_state_4() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 4,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

#[test]
fn noop_subpass_render_pass_lifecycle_records_two_subpasses() {
    let test = ValidationTest::new();

    test.expect_no_validation_error(|| unsafe {
        let layout = create_two_subpass_layout(test.device());
        let pipeline0 = create_subpass_pipeline(test.device(), layout, 0);
        let pipeline1 = create_subpass_pipeline(test.device(), layout, 1);
        let target = create_render_target(test.device());
        let attachment = subpass_color_attachment(target.view);
        let pass_descriptor = YaWGPUSubpassRenderPassDescriptor {
            nextInChain: std::ptr::null(),
            label: empty_string_view(),
            passLayout: layout,
            extent: target_extent(),
            colorAttachments: &attachment,
            colorAttachmentCount: 1,
            depthStencilAttachment: std::ptr::null(),
        };

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
        assert!(!pass.is_null());
        yawgpu::yawgpuSubpassRenderPassEncoderAddRef(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline0);
        yawgpu::yawgpuSubpassRenderPassEncoderSetViewport(pass, 0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        yawgpu::yawgpuSubpassRenderPassEncoderSetScissorRect(pass, 0, 0, 4, 4);
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline1);
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);

        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        release_render_target(target);
        yawgpu::wgpuRenderPipelineRelease(pipeline1);
        yawgpu::wgpuRenderPipelineRelease(pipeline0);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
    });
}

#[test]
fn noop_subpass_next_past_last_routes_device_error() {
    let test = ValidationTest::new();

    test.assert_device_error_after(
        || unsafe {
            let layout = create_two_subpass_layout(test.device());
            let target = create_render_target(test.device());
            let attachment = subpass_color_attachment(target.view);
            let pass_descriptor = YaWGPUSubpassRenderPassDescriptor {
                nextInChain: std::ptr::null(),
                label: empty_string_view(),
                passLayout: layout,
                extent: target_extent(),
                colorAttachments: &attachment,
                colorAttachmentCount: 1,
                depthStencilAttachment: std::ptr::null(),
            };
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
            let pass =
                yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
            yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
            yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
            yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
            yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
            let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandBufferRelease(command_buffer);
            yawgpu::wgpuCommandEncoderRelease(encoder);
            release_render_target(target);
            yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        },
        Some("cannot advance past the last subpass"),
    );
}

#[test]
fn noop_subpass_begin_after_encoder_command_routes_device_error() {
    let test = ValidationTest::new();

    test.assert_device_error_after(
        || unsafe {
            let layout = create_two_subpass_layout(test.device());
            let target = create_render_target(test.device());
            let attachment = subpass_color_attachment(target.view);
            let pass_descriptor = YaWGPUSubpassRenderPassDescriptor {
                nextInChain: std::ptr::null(),
                label: empty_string_view(),
                passLayout: layout,
                extent: target_extent(),
                colorAttachments: &attachment,
                colorAttachmentCount: 1,
                depthStencilAttachment: std::ptr::null(),
            };
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
            yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, empty_string_view());
            let pass =
                yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
            yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
            let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandBufferRelease(command_buffer);
            yawgpu::wgpuCommandEncoderRelease(encoder);
            release_render_target(target);
            yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        },
        Some("subpass render pass must be the first command encoder operation"),
    );
}

unsafe fn create_two_subpass_layout(device: native::WGPUDevice) -> yawgpu::YaWGPUSubpassPassLayout {
    let color = YaWGPUAttachmentLayout {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sampleCount: 1,
    };
    let color_index = 0;
    let subpasses = [
        YaWGPUSubpassLayout {
            colorAttachmentIndices: &color_index,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        },
        YaWGPUSubpassLayout {
            colorAttachmentIndices: &color_index,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        },
    ];
    let descriptor = YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: &color,
        colorAttachmentCount: 1,
        depthStencilAttachment: std::ptr::null(),
        subpasses: subpasses.as_ptr(),
        subpassCount: subpasses.len(),
        dependencies: std::ptr::null(),
        dependencyCount: 0,
    };
    let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_subpass_pipeline(
    device: native::WGPUDevice,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    subpass_index: u32,
) -> native::WGPURenderPipeline {
    let shader = create_wgsl_module(
        device,
        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
         @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); }",
    );
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: shader,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let base = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
        fragment: &fragment,
    };
    let descriptor = YaWGPUSubpassRenderPipelineDescriptor {
        nextInChain: std::ptr::null(),
        base,
        passLayout: layout,
        subpassIndex: subpass_index,
    };
    let pipeline = yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &descriptor);
    yawgpu::wgpuShaderModuleRelease(shader);
    assert!(!pipeline.is_null());
    pipeline
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
    let shader = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!shader.is_null());
    shader
}

fn default_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
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
        size: target_extent(),
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

fn subpass_color_attachment(view: native::WGPUTextureView) -> YaWGPUSubpassColorAttachment {
    YaWGPUSubpassColorAttachment {
        view,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    }
}

fn target_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 4,
        height: 4,
        depthOrArrayLayers: 1,
    }
}
