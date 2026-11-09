//! Shared body of the tiled external-texture e2e (`specs/blocks/149-subpass-external-textures-and-color-target-format.md`
//! R1), included by `e2e_metal_tiled_external_texture.rs` and
//! `e2e_vulkan_tiled_external_texture.rs` through `#[path]`.
//!
//! A subpass pipeline samples an RGBA external texture. The readback must be the
//! external texture's texel, which proves the subpass draw binds the plane and the
//! params buffer.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUAttachmentLayout, YaWGPUExtent2D, YaWGPUExternalTextureDescriptor,
    YaWGPUInstanceBackendSelect, YaWGPUOrigin2D, YaWGPUSubpassColorAttachment,
    YaWGPUSubpassDependency, YaWGPUSubpassDependencyType_ColorToInput,
    YaWGPUSubpassInputAttachment, YaWGPUSubpassLayout, YaWGPUSubpassPassLayout,
    YaWGPUSubpassPassLayoutDescriptor, YaWGPUSubpassRenderPassDescriptor,
    YaWGPUSubpassRenderPipelineDescriptor, YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA,
    YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

/// The external texture's texel.
const EXTERNAL_RGBA: [u8; 4] = [0, 128, 192, 255];
/// Subpass 0 of the two-subpass scenario writes red 0.25 into the g-buffer.
const GBUFFER_RED: u8 = 64;

const SINGLE_SUBPASS_SHADER: &str = r#"
@group(0) @binding(0) var ext: texture_external;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return textureLoad(ext, vec2u(0u, 0u));
}
"#;

const WRITE_GBUFFER_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.25, 0.0, 0.0, 0.0);
}
"#;

const INPUT_AND_EXTERNAL_SHADER: &str = r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var gbuffer: input_attachment<f32>;
@group(1) @binding(0) var ext: texture_external;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(1) vec4<f32> {
  return inputAttachmentLoad(gbuffer) + textureLoad(ext, vec2u(0u, 0u));
}
"#;

/// One subpass; its pipeline samples an external texture at group 0.
pub fn subpass_samples_external_texture(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        eprintln!("skipping: no real {backend:?} device");
        return;
    }
    unsafe {
        let ctx = Context::new(select);
        let (external, source, plane0) = create_external_texture(&ctx);

        let layout = create_single_subpass_layout(ctx.device);
        let pipeline = create_subpass_pipeline(
            ctx.device,
            layout,
            0,
            SINGLE_SUBPASS_SHADER,
            0,
            None,
            std::ptr::null_mut(),
        );
        let bind_group = create_external_bind_group(&ctx, pipeline, 0, external);

        let target = create_texture(
            ctx.device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let target_view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        let attachments = [color_attachment(target_view)];
        let pass_descriptor = pass_descriptor(layout, &attachments);

        let readback = create_readback(ctx.device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
        assert!(!pass.is_null());
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::yawgpuSubpassRenderPassEncoderSetBindGroup(
            pass,
            0,
            bind_group,
            0,
            std::ptr::null(),
        );
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
        submit_with_readback(&ctx, encoder, target, readback);

        let pixels = read_pixels(ctx.instance, readback);
        ctx.assert_no_errors("subpass external-texture draw");
        assert_every_pixel(
            &pixels,
            EXTERNAL_RGBA,
            "single subpass sampling an external texture",
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(target_view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::wgpuExternalTextureRelease(external);
        yawgpu::wgpuTextureViewRelease(plane0);
        yawgpu::wgpuTextureRelease(source);
    }
}

/// Two subpasses; subpass 1 reads an input attachment at group 0 and samples an
/// external texture at group 1, and writes their sum.
pub fn subpass_reads_input_attachment_and_external_texture(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        eprintln!("skipping: no real {backend:?} device");
        return;
    }
    unsafe {
        let ctx = Context::new(select);
        let (external, source, plane0) = create_external_texture(&ctx);

        let layout = create_two_subpass_input_layout(ctx.device);
        let pipeline0 = create_subpass_pipeline(
            ctx.device,
            layout,
            0,
            WRITE_GBUFFER_SHADER,
            0,
            None,
            std::ptr::null_mut(),
        );
        let pipeline1 = create_subpass_pipeline(
            ctx.device,
            layout,
            1,
            INPUT_AND_EXTERNAL_SHADER,
            1,
            None,
            std::ptr::null_mut(),
        );
        let bind_group = create_external_bind_group(&ctx, pipeline1, 1, external);

        let gbuffer = create_texture(ctx.device, native::WGPUTextureUsage_RenderAttachment);
        let target = create_texture(
            ctx.device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let gbuffer_view = yawgpu::wgpuTextureCreateView(gbuffer, std::ptr::null());
        let target_view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        let attachments = [
            color_attachment(gbuffer_view),
            color_attachment(target_view),
        ];
        let pass_descriptor = pass_descriptor(layout, &attachments);

        let readback = create_readback(ctx.device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
        assert!(!pass.is_null());
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline0);
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline1);
        yawgpu::yawgpuSubpassRenderPassEncoderSetBindGroup(
            pass,
            1,
            bind_group,
            0,
            std::ptr::null(),
        );
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
        submit_with_readback(&ctx, encoder, target, readback);

        let pixels = read_pixels(ctx.instance, readback);
        ctx.assert_no_errors("subpass input-attachment + external-texture draw");
        let expected = [
            GBUFFER_RED,
            EXTERNAL_RGBA[1],
            EXTERNAL_RGBA[2],
            EXTERNAL_RGBA[3],
        ];
        assert_every_pixel(&pixels, expected, "input attachment + external texture");

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(target_view);
        yawgpu::wgpuTextureViewRelease(gbuffer_view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuTextureRelease(gbuffer);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuRenderPipelineRelease(pipeline1);
        yawgpu::wgpuRenderPipelineRelease(pipeline0);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::wgpuExternalTextureRelease(external);
        yawgpu::wgpuTextureViewRelease(plane0);
        yawgpu::wgpuTextureRelease(source);
    }
}

/// Number of `u32` elements in the storage buffer of
/// [`subpass_unused_external_texture_keeps_buffer_sizes`]; the shader writes it as red.
const STORAGE_ELEMENTS: u32 = 64;

const ARRAY_LENGTH_SHADER: &str = r#"
@group(0) @binding(1) var<storage, read> data: array<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(f32(arrayLength(&data)) / 255.0, 0.0, 0.0, 1.0);
}
"#;

/// One subpass with an explicit layout: binding 0 is an external texture the
/// shader does not use, binding 1 a storage buffer whose `arrayLength` the
/// fragment writes as red. The buffer sizes reach the shader in the immediates
/// block, and the block's Metal slot is chosen among the slots the entry point
/// does not use, which includes the unused external texture's params slot. The
/// params buffer must not overwrite the block.
pub fn subpass_unused_external_texture_keeps_buffer_sizes(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        eprintln!("skipping: no real {backend:?} device");
        return;
    }
    unsafe {
        let ctx = Context::new(select);
        let (external, source, plane0) = create_external_texture(&ctx);

        let mut external_layout = native::WGPUExternalTextureBindingLayout {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ExternalTextureBindingLayout,
            },
        };
        let mut external_entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
        external_entry.nextInChain = (&mut external_layout.chain) as *mut native::WGPUChainedStruct;
        external_entry.binding = 0;
        external_entry.visibility = native::WGPUShaderStage_Fragment;
        let mut storage_entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
        storage_entry.binding = 1;
        storage_entry.visibility = native::WGPUShaderStage_Fragment;
        storage_entry.buffer.type_ = native::WGPUBufferBindingType_ReadOnlyStorage;
        let layout_entries = [external_entry, storage_entry];
        let bind_group_layout_descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: layout_entries.len(),
            entries: layout_entries.as_ptr(),
        };
        let bind_group_layout =
            yawgpu::wgpuDeviceCreateBindGroupLayout(ctx.device, &bind_group_layout_descriptor);
        assert!(!bind_group_layout.is_null());
        let pipeline_layout_descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bind_group_layout,
            immediateSize: 0,
        };
        let pipeline_layout =
            yawgpu::wgpuDeviceCreatePipelineLayout(ctx.device, &pipeline_layout_descriptor);
        assert!(!pipeline_layout.is_null());

        let layout = create_single_subpass_layout(ctx.device);
        let pipeline = create_subpass_pipeline(
            ctx.device,
            layout,
            0,
            ARRAY_LENGTH_SHADER,
            0,
            None,
            pipeline_layout,
        );

        let storage_size = u64::from(STORAGE_ELEMENTS) * 4;
        let storage_descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUBufferUsage_Storage,
            size: storage_size,
            mappedAtCreation: 0,
        };
        let storage = yawgpu::wgpuDeviceCreateBuffer(ctx.device, &storage_descriptor);
        assert!(!storage.is_null());

        let mut ext_entry = native::WGPUExternalTextureBindingEntry {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ExternalTextureBindingEntry,
            },
            externalTexture: external,
        };
        let mut external_bind: native::WGPUBindGroupEntry = std::mem::zeroed();
        external_bind.nextInChain = (&mut ext_entry.chain) as *mut native::WGPUChainedStruct;
        external_bind.binding = 0;
        let mut storage_bind: native::WGPUBindGroupEntry = std::mem::zeroed();
        storage_bind.binding = 1;
        storage_bind.buffer = storage;
        storage_bind.size = storage_size;
        let bind_entries = [external_bind, storage_bind];
        let bind_group_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: bind_group_layout,
            entryCount: bind_entries.len(),
            entries: bind_entries.as_ptr(),
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(ctx.device, &bind_group_descriptor);
        assert!(
            !bind_group.is_null(),
            "bind group creation failed: {:?}",
            ctx.errors.lock().expect("lock")
        );

        let target = create_texture(
            ctx.device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let target_view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        let attachments = [color_attachment(target_view)];
        let pass_descriptor = pass_descriptor(layout, &attachments);

        let readback = create_readback(ctx.device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
        assert!(!pass.is_null());
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::yawgpuSubpassRenderPassEncoderSetBindGroup(
            pass,
            0,
            bind_group,
            0,
            std::ptr::null(),
        );
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
        submit_with_readback(&ctx, encoder, target, readback);

        let pixels = read_pixels(ctx.instance, readback);
        ctx.assert_no_errors("subpass draw with an unused external texture");
        let expected = [
            u8::try_from(STORAGE_ELEMENTS).expect("fits in u8"),
            0,
            0,
            255,
        ];
        assert_every_pixel(
            &pixels,
            expected,
            "arrayLength next to an unused external texture",
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(target_view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bind_group_layout);
        yawgpu::wgpuExternalTextureRelease(external);
        yawgpu::wgpuTextureViewRelease(plane0);
        yawgpu::wgpuTextureRelease(source);
    }
}

struct Context {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<String>>>,
}

impl Context {
    unsafe fn new(select: u32) -> Self {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured.lock().expect("lock").push(format!("{error:?}"))),
        );
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        Self {
            instance,
            adapter,
            device,
            queue,
            errors,
        }
    }

    fn assert_no_errors(&self, what: &str) {
        let errors = self.errors.lock().expect("lock");
        assert!(errors.is_empty(), "{what} raised device errors: {errors:?}");
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuQueueRelease(self.queue);
            yawgpu::wgpuDeviceRelease(self.device);
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

/// An RGBA external texture over a 1x1 plane holding [`EXTERNAL_RGBA`], with the
/// YUV-only identity params of `common/external_texture.rs`.
unsafe fn create_external_texture(
    ctx: &Context,
) -> (
    native::WGPUExternalTexture,
    native::WGPUTexture,
    native::WGPUTextureView,
) {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 1,
            height: 1,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let source = yawgpu::wgpuDeviceCreateTexture(ctx.device, &descriptor);
    assert!(!source.is_null());
    let destination = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: 4,
        rowsPerImage: 1,
    };
    let extent = native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuQueueWriteTexture(
        ctx.queue,
        &destination,
        EXTERNAL_RGBA.as_ptr().cast(),
        EXTERNAL_RGBA.len(),
        &layout,
        &extent,
    );
    let plane0 = yawgpu::wgpuTextureCreateView(source, std::ptr::null());
    assert!(!plane0.is_null());

    let descriptor = YaWGPUExternalTextureDescriptor {
        plane0,
        plane1: std::ptr::null_mut(),
        format: YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA,
        cropOrigin: YaWGPUOrigin2D { x: 0, y: 0 },
        cropSize: YaWGPUExtent2D {
            width: 1,
            height: 1,
        },
        apparentSize: YaWGPUExtent2D {
            width: 1,
            height: 1,
        },
        doYuvToRgbConversionOnly: 1,
        yuvToRgbConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        srcTransferFunctionParameters: [0.0; 7],
        dstTransferFunctionParameters: [0.0; 7],
        gamutConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        mirrored: 0,
        rotation: YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES,
    };
    let external = yawgpu::yawgpuDeviceCreateExternalTexture(ctx.device, &descriptor);
    assert!(!external.is_null(), "external texture creation failed");
    (external, source, plane0)
}

unsafe fn create_external_bind_group(
    ctx: &Context,
    pipeline: native::WGPURenderPipeline,
    group: u32,
    external: native::WGPUExternalTexture,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, group);
    assert!(!layout.is_null());
    let mut ext_entry = native::WGPUExternalTextureBindingEntry {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ExternalTextureBindingEntry,
        },
        externalTexture: external,
    };
    let entry = native::WGPUBindGroupEntry {
        nextInChain: (&mut ext_entry.chain) as *mut native::WGPUChainedStruct,
        binding: 0,
        buffer: std::ptr::null_mut(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(ctx.device, &descriptor);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    assert!(
        !bind_group.is_null(),
        "bind group creation failed: {:?}",
        ctx.errors.lock().expect("lock")
    );
    bind_group
}

unsafe fn create_single_subpass_layout(device: native::WGPUDevice) -> YaWGPUSubpassPassLayout {
    let colors = [YaWGPUAttachmentLayout {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sampleCount: 1,
    }];
    let subpass_colors = [0u32];
    let subpasses = [YaWGPUSubpassLayout {
        colorAttachmentIndices: subpass_colors.as_ptr(),
        colorAttachmentIndexCount: subpass_colors.len(),
        usesDepthStencil: 0,
        inputAttachments: std::ptr::null(),
        inputAttachmentCount: 0,
    }];
    let descriptor = YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: colors.as_ptr(),
        colorAttachmentCount: colors.len(),
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

unsafe fn create_two_subpass_input_layout(device: native::WGPUDevice) -> YaWGPUSubpassPassLayout {
    let colors = [
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        },
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
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

/// Targets are indexed by global color slot; only `write_slot` is written. A null
/// `pipeline_layout` means `auto`.
unsafe fn create_subpass_pipeline(
    device: native::WGPUDevice,
    layout: YaWGPUSubpassPassLayout,
    subpass_index: u32,
    shader_src: &str,
    write_slot: u32,
    read_slot: Option<u32>,
    pipeline_layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let shader = create_wgsl_module(device, shader_src);
    let max_slot = read_slot.map_or(write_slot, |r| r.max(write_slot));
    let targets: Vec<native::WGPUColorTargetState> = (0..=max_slot)
        .map(|slot| native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: if slot == write_slot || read_slot == Some(slot) {
                native::WGPUTextureFormat_RGBA8Unorm
            } else {
                native::WGPUTextureFormat_Undefined
            },
            blend: std::ptr::null(),
            writeMask: if slot == write_slot {
                native::WGPUColorWriteMask_All
            } else {
                native::WGPUColorWriteMask_None
            },
        })
        .collect();
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
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
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
    assert!(
        !pipeline.is_null(),
        "subpass pipeline {subpass_index} creation failed"
    );
    pipeline
}

fn color_attachment(view: native::WGPUTextureView) -> YaWGPUSubpassColorAttachment {
    YaWGPUSubpassColorAttachment {
        view,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        // Red: a draw that never reaches the target leaves it visible.
        clearValue: native::WGPUColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    }
}

fn pass_descriptor(
    layout: YaWGPUSubpassPassLayout,
    attachments: &[YaWGPUSubpassColorAttachment],
) -> YaWGPUSubpassRenderPassDescriptor {
    YaWGPUSubpassRenderPassDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        passLayout: layout,
        extent: texture_extent(),
        colorAttachments: attachments.as_ptr(),
        colorAttachmentCount: attachments.len(),
        depthStencilAttachment: std::ptr::null(),
    }
}

unsafe fn submit_with_readback(
    ctx: &Context,
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    readback: native::WGPUBuffer,
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
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: readback,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &source,
        &destination,
        &texture_extent(),
    );
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(ctx.queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

fn assert_every_pixel(pixels: &[u8], expected: [u8; 4], what: &str) {
    for (index, pixel) in pixels.as_chunks::<BYTES_PER_PIXEL>().0.iter().enumerate() {
        let close = pixel
            .iter()
            .zip(expected.iter())
            .all(|(&actual, &want)| actual.abs_diff(want) <= 2);
        assert!(
            close,
            "{what}: pixel {index} is {pixel:?}, expected ~{expected:?}"
        );
    }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
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

unsafe fn create_readback(device: native::WGPUDevice) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        size: READBACK_SIZE as u64,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
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
        nextInChain: (&mut wgsl.chain) as *mut native::WGPUChainedStruct,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null(), "shader module creation failed");
    module
}

/// Reads the readback buffer and drops the row padding.
unsafe fn read_pixels(instance: native::WGPUInstance, buffer: native::WGPUBuffer) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuBufferMapAsync(
        buffer,
        native::WGPUMapMode_Read,
        0,
        READBACK_SIZE,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, READBACK_SIZE);
    assert!(!ptr.is_null());
    let mapped = std::slice::from_raw_parts(ptr.cast::<u8>(), READBACK_SIZE);
    let mut pixels = Vec::with_capacity(ROW_BYTES * HEIGHT as usize);
    for row in 0..HEIGHT as usize {
        let start = row * BYTES_PER_ROW as usize;
        pixels.extend_from_slice(&mapped[start..start + ROW_BYTES]);
    }
    yawgpu::wgpuBufferUnmap(buffer);
    pixels
}

unsafe fn create_instance(select: u32) -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: select,
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
    let mut adapter: native::WGPUAdapter = std::ptr::null_mut();
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
    let mut device: native::WGPUDevice = std::ptr::null_mut();
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
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

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
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
