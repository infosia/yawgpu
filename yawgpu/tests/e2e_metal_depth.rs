//! Real-GPU Metal diagnosis for F-031: the depth aspect of
//! `copyTextureToTexture` (and the depth-attachment render path that the
//! CTS `copy_depth_stencil` test exercises) on `Depth32Float`.
//!
//! The CTS test never reads the depth aspect back to a buffer — it renders
//! depth, copies texture-to-texture, then re-renders with `depthCompare:equal`
//! and checks the colour output. That couples three stages (depth render,
//! t2t copy, equal re-render). These tests split them apart by reading the
//! depth aspect back via `copyTextureToBuffer` on `Depth32Float` (a depth-only
//! format whose depth aspect is buffer-copyable as raw f32), so a failure
//! localises to a single stage.
//!
//! These tests verify real-GPU depth behaviour and have no meaningful Noop
//! equivalent, so the whole file is gated on the `metal` feature.
#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "metal")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
#[cfg(feature = "metal")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
#[cfg(feature = "metal")]
const DEPTH_VALUE: f32 = 0.5;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

// Full-viewport quad rendered at a constant depth, no fragment output.
const DEPTH_INIT_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.5, 1.0);
}
"#;

/// Stage A — depth-attachment render path + depth-aspect readback.
/// Render a constant depth of 0.5 into a `Depth32Float` texture, then copy the
/// depth aspect to a buffer and confirm every texel reads back 0.5.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth_render_writes_depth_then_reads_back() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        encode_depth_init_pass(device, encoder, depth, DEPTH_INIT_SHADER);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "depth render/readback mismatch: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Stage B — `copyTextureToTexture` of the depth aspect (aspect = All, as the
/// CTS test issues it). Render depth 0.5 into a source `Depth32Float`, copy it
/// texture-to-texture into a destination, then read the destination depth
/// aspect back and confirm it still reads 0.5.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth_copy_texture_to_texture_preserves_depth() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let src = create_depth_texture(device);
        let dst = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        encode_depth_init_pass(device, encoder, src, DEPTH_INIT_SHADER);
        record_t2t(encoder, src, dst);
        record_depth_t2b(encoder, dst, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "copyTextureToTexture depth aspect not preserved: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(dst);
        yawgpu::wgpuTextureRelease(src);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Color shader pair for the color+depth isolation test: writes red so the
// colour attachment proves the pass executed, while the depth-stencil state
// writes the constant depth.
const COLOR_DEPTH_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.5, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }
"#;

/// Isolation — a color+depth render pass (so the pass is not dropped for being
/// colour-less) that writes a constant depth. Reads the colour attachment back
/// to confirm the pass executed, then reads the depth aspect back. This pins
/// the depth-attachment-binding gap independent of the depth-only (no-colour)
/// render-pass support that `initializeDepthAspect` also needs.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_color_depth_render_writes_depth_then_reads_back() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let color = create_color_texture(device);
        let color_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let depth_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        encode_color_depth_pass(device, encoder, color, depth);
        record_color_t2b(encoder, color, color_readback);
        record_depth_t2b(encoder, depth, depth_readback);
        submit_encoder(queue, encoder);

        let colors = read_buffer(instance, color_readback, 0, READBACK_SIZE);
        let red = colors[0] == 255 && colors[1] == 0 && colors[2] == 0 && colors[3] == 255;
        assert!(
            red,
            "colour attachment did not render red: {:?}",
            &colors[0..4]
        );

        let depths = read_depth_buffer(instance, depth_readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "depth attachment not written in a color+depth pass: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(depth_readback);
        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Green fragment + constant-depth vertex, for the depthCompare=Equal re-render
// that mirrors the CTS `verifyDepthAspect` stage.
const DEPTH_EQUAL_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.5, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
"#;

/// Isolation — the `verifyDepthAspect` path: render a constant depth 0.5, then a
/// second pass that loads that depth (`depthLoadOp=Load`) and re-renders with
/// `depthCompare=Equal` + `depthWriteEnabled=false`, writing green where equal
/// over a red clear. The CTS exercises this with an interpolated depth gradient;
/// this uses a constant to isolate the Equal + Load wiring from gradient
/// precision.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth_equal_load_renders_green() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let color = create_color_texture(device);
        let color_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        encode_depth_init_pass(device, encoder, depth, DEPTH_INIT_SHADER);
        encode_depth_equal_verify_pass(device, encoder, color, depth, DEPTH_EQUAL_SHADER);
        record_color_t2b(encoder, color, color_readback);
        submit_encoder(queue, encoder);

        let colors = read_buffer(instance, color_readback, 0, READBACK_SIZE);
        let green = colors[0] == 0 && colors[1] == 255 && colors[2] == 0 && colors[3] == 255;
        assert!(
            green,
            "depthCompare=Equal re-render did not produce green: {:?}",
            &colors[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// CTS `kDepthVertexShader` geometry (layer 0, depthValue = 0.5): an interpolated
// depth gradient (z from 0.0 to 1.0 across the quad), no fragment — for the init.
const GRADIENT_INIT_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0,  1.0, 0.5),
        vec3<f32>(-1.0, -1.0, 0.0),
        vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>(-1.0, -1.0, 0.0),
        vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>( 1.0, -1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}
"#;

// Same gradient geometry plus a green fragment, for the depthCompare=Equal pass.
const GRADIENT_EQUAL_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0,  1.0, 0.5),
        vec3<f32>(-1.0, -1.0, 0.0),
        vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>(-1.0, -1.0, 0.0),
        vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>( 1.0, -1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
"#;

/// Isolation — the CTS `verifyDepthAspect` shape with an interpolated depth
/// gradient (no copy, no uniform): render the gradient, then re-render the same
/// gradient with `depthCompare=Equal` + `depthLoadOp=Load`. Equality must hold at
/// every texel (Dawn/wgpu-native produce all green). Pins gradient-depth
/// reproducibility between a vertex-only init pipeline and a vertex+fragment
/// verify pipeline.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth_gradient_equal_renders_green() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let color = create_color_texture(device);
        let color_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        encode_depth_init_pass(device, encoder, depth, GRADIENT_INIT_SHADER);
        encode_depth_equal_verify_pass(device, encoder, color, depth, GRADIENT_EQUAL_SHADER);
        record_color_t2b(encoder, color, color_readback);
        submit_encoder(queue, encoder);

        let colors = read_buffer(instance, color_readback, 0, READBACK_SIZE);
        // The readback buffer is row-padded to BYTES_PER_ROW; only the first
        // WIDTH*4 bytes of each row are real pixels.
        let mut non_green = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                let p = &colors[off..off + 4];
                if !(p[0] == 0 && p[1] == 255 && p[2] == 0 && p[3] == 255) {
                    non_green.push(((col, row), [p[0], p[1], p[2], p[3]]));
                }
            }
        }
        assert!(
            non_green.is_empty(),
            "gradient depthCompare=Equal re-render not all green: {non_green:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Separate vertex and fragment modules (the CTS verify pipeline uses
// kDepthVertexShader + kGreenFragmentShader as two distinct modules).
const SEPARATE_VS: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.5, 1.0);
}
"#;
const SEPARATE_FS: &str = r#"
@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
"#;

/// Isolation — a render pipeline whose vertex and fragment entries live in
/// SEPARATE shader modules (as the CTS `verifyDepthAspect` pipeline does). Pins
/// whether yawgpu can build a Metal pipeline from two distinct WGSL modules.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_render_pipeline_accepts_separate_vs_fs_modules() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let vs_module = create_wgsl_module(device, SEPARATE_VS);
        let fs_module = create_wgsl_module(device, SEPARATE_FS);
        let color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module: fs_module,
            entryPoint: string_view("fs"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: &color_target,
        };
        let descriptor = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module: vs_module,
                entryPoint: string_view("vs"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: primitive_state(),
            depthStencil: std::ptr::null(),
            multisample: multisample_state(),
            fragment: &fragment,
        };
        let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
        assert!(!pipeline.is_null());
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "separate vs/fs modules produced a device error: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_layered_depth_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    layers: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: layers,
        },
        format: native::WGPUTextureFormat_Depth32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let tex = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!tex.is_null());
    tex
}

/// Isolation — `copyTextureToTexture` of a multi-layer (`Type2DArray`) depth
/// texture with aspect = All and all layers in one copy, exactly as the CTS
/// multi-layer `copy_depth_stencil` cases issue it. Pins whether yawgpu accepts
/// a multi-layer depth texture-to-texture copy (WebGPU permits it; Dawn passes).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth_multilayer_copy_texture_to_texture() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let src = create_layered_depth_texture(
            device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            2,
        );
        let dst = create_layered_depth_texture(
            device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopyDst,
            2,
        );

        let source = native::WGPUTexelCopyTextureInfo {
            texture: src,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let destination = native::WGPUTexelCopyTextureInfo {
            texture: dst,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let copy_size = native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: 2,
        };

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &copy_size);
        submit_encoder(queue, encoder);

        assert!(
            errors.lock().expect("error lock").is_empty(),
            "multi-layer depth copyTextureToTexture errored: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuTextureRelease(dst);
        yawgpu::wgpuTextureRelease(src);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn encode_depth_equal_verify_pass(
    device: native::WGPUDevice,
    encoder: native::WGPUCommandEncoder,
    color: native::WGPUTexture,
    depth: native::WGPUTexture,
    shader: &str,
) {
    let module = create_wgsl_module(device, shader);
    let pipeline = create_depth_equal_pipeline(device, module);
    let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
    let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

    let color_attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view: color_view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    };
    let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view: depth_view,
        depthLoadOp: native::WGPULoadOp_Load,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 0.0,
        depthReadOnly: false.into(),
        stencilLoadOp: native::WGPULoadOp_Undefined,
        stencilStoreOp: native::WGPUStoreOp_Undefined,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    };
    let attachments = [color_attachment];
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: &depth_attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(depth_view);
    yawgpu::wgpuTextureViewRelease(color_view);
}

#[cfg(feature = "metal")]
unsafe fn create_depth_equal_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_False,
        depthCompare: native::WGPUCompareFunction_Equal,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn encode_color_depth_pass(
    device: native::WGPUDevice,
    encoder: native::WGPUCommandEncoder,
    color: native::WGPUTexture,
    depth: native::WGPUTexture,
) {
    let module = create_wgsl_module(device, COLOR_DEPTH_SHADER);
    let pipeline = create_color_depth_pipeline(device, module);
    let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
    let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

    let color_attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view: color_view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        },
    };
    let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view: depth_view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 0.0,
        depthReadOnly: false.into(),
        stencilLoadOp: native::WGPULoadOp_Undefined,
        stencilStoreOp: native::WGPUStoreOp_Undefined,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    };
    let attachments = [color_attachment];
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: &depth_attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(depth_view);
    yawgpu::wgpuTextureViewRelease(color_view);
}

#[cfg(feature = "metal")]
unsafe fn create_color_depth_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe fn record_color_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
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
        buffer: destination,
    };
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
unsafe fn encode_depth_init_pass(
    device: native::WGPUDevice,
    encoder: native::WGPUCommandEncoder,
    depth: native::WGPUTexture,
    shader: &str,
) {
    let module = create_wgsl_module(device, shader);
    let pipeline = create_depth_init_pipeline(device, module);
    let view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

    let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 0.0,
        depthReadOnly: false.into(),
        stencilLoadOp: native::WGPULoadOp_Undefined,
        stencilStoreOp: native::WGPUStoreOp_Undefined,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    };
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 0,
        colorAttachments: std::ptr::null(),
        depthStencilAttachment: &depth_attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(view);
}

#[cfg(feature = "metal")]
unsafe fn create_depth_init_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: std::ptr::null(),
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
fn stencil_face() -> native::WGPUStencilFaceState {
    // Default (no-op) stencil face for a depth-only format: matches
    // WGPU_STENCIL_FACE_STATE_INIT so core does not treat the pipeline as
    // using stencil (which would require a stencil format).
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
    }
}

#[cfg(feature = "metal")]
unsafe fn record_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let source = depth_texture_copy_info(source);
    let destination = depth_texture_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
unsafe fn record_depth_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let mut source = depth_texture_copy_info(source);
    source.aspect = native::WGPUTextureAspect_DepthOnly;
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: destination,
    };
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
fn depth_texture_copy_info(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

#[cfg(feature = "metal")]
unsafe fn create_depth_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment
            | native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_Depth32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "metal")]
unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

#[cfg(feature = "metal")]
unsafe fn read_depth_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
) -> Vec<f32> {
    let mapped = read_buffer(instance, buffer, 0, READBACK_SIZE);
    let mut depths = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for row in 0..HEIGHT as usize {
        let row_offset = row * BYTES_PER_ROW as usize;
        for col in 0..WIDTH as usize {
            let off = row_offset + col * 4;
            let bytes = [
                mapped[off],
                mapped[off + 1],
                mapped[off + 2],
                mapped[off + 3],
            ];
            depths.push(f32::from_le_bytes(bytes));
        }
    }
    depths
}

#[cfg(feature = "metal")]
unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: u64,
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
    let future = yawgpu::wgpuBufferMapAsync(
        buffer,
        native::WGPUMapMode_Read,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);

    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
fn multisample_state() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

#[cfg(feature = "metal")]
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
