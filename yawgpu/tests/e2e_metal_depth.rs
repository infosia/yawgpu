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

/// Isolation — render a constant `frag_depth` into **mip level 1** of a 2-mip
/// `Depth32Float` texture (via a mip-1 view), then read mip 1's depth aspect
/// back. Pins whether the depth render attachment + depth t2b target a non-zero
/// mip level (the CTS depth cases vary `mipLevel`).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_frag_depth_into_mip_level_1() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // 4x4 Depth32Float with 2 mips; mip 1 is 2x2.
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: texture_extent(),
            format: native::WGPUTextureFormat_Depth32Float,
            mipLevelCount: 2,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!depth.is_null());

        let view_descriptor = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_Depth32Float,
            dimension: native::WGPUTextureViewDimension_2D,
            baseMipLevel: 1,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_RenderAttachment,
        };
        let mip_view = yawgpu::wgpuTextureCreateView(depth, &view_descriptor);
        assert!(!mip_view.is_null());

        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, FRAG_DEPTH_FS);
        let pipeline = create_frag_depth_pipeline(device, vs_module, fs_module);

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: mip_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        // Copy mip 1's depth aspect (2x2) back.
        let source = native::WGPUTexelCopyTextureInfo {
            texture: depth,
            mipLevel: 1,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: HEIGHT / 2,
            },
            buffer: readback,
        };
        let mip_size = native::WGPUExtent3D {
            width: WIDTH / 2,
            height: HEIGHT / 2,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &mip_size);
        submit_encoder(queue, encoder);

        let mapped = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..(HEIGHT / 2) as usize {
            for col in 0..(WIDTH / 2) as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                let got = f32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]);
                if (got - 0.7).abs() > 1e-5 {
                    mismatches.push((col, row, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "frag_depth into mip 1 not read back from mip 1 (col,row,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(mip_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Samples a bound `r32float` texture per-fragment and writes the value as
// `@builtin(frag_depth)` (no colour output) — the exact CTS T27 depth staging.
const SAMPLE_FRAG_DEPTH_FS: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@fragment
fn fs(@builtin(position) coord: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(input_tex, vec2<i32>(coord.xy), 0).x;
}
"#;

/// Isolation — the exact CTS depth staging: a frag-depth-only fragment that
/// SAMPLES a bound `r32float` per fragment and writes it as `frag_depth`, into a
/// `Depth32Float` texture; then reads the depth aspect back per texel. Combines
/// texture sampling + varying frag_depth (each verified separately elsewhere).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampled_frag_depth_writes_per_texel_depth() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // r32float input: depth value = (x + y*4 + 1)/32 per texel (in [0,1)).
        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        let expected_depth =
            |col: usize, row: usize| (col + row * WIDTH as usize + 1) as f32 / 32.0;
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_FS);
        let pipeline = create_frag_depth_pipeline(device, vs_module, fs_module);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let bind_group = create_texture_bind_group(device, pipeline, input_view);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

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
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let got = depths[row * WIDTH as usize + col];
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "sampled frag_depth not written per texel (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Samples a bound `r32float` texture per-fragment and writes it as the red
// channel — exercises sampled-texture bind-group execution (the CTS depth
// staging samples an r32float in a frag-depth shader).
const SAMPLE_TEXTURE_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let v = textureLoad(input_tex, vec2<i32>(coord.xy), 0).x;
    return vec4<f32>(v, 0.0, 0.0, 1.0);
}
"#;

/// Isolation — a fragment that `textureLoad`s a bound `r32float` texture
/// per-fragment and writes it as the red channel. Confirms sampled-texture
/// bind-group execution binds the texture AND reads the correct texel per
/// fragment (not just texel (0,0)). The CTS depth staging relies on this.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_fragment_samples_bound_texture_per_texel() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // 4x4 r32float input: value = (x + y*4 + 1) / 16 per texel.
        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let v = (col + row * WIDTH as usize + 1) as f32 / 16.0;
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        let color = create_color_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, SAMPLE_TEXTURE_SHADER);
        let pipeline = create_sample_texture_pipeline(device, module);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let bind_group = create_texture_bind_group(device, pipeline, input_view);
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

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
                b: 0.0,
                a: 1.0,
            },
        };
        let attachments = [color_attachment];
        let descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_color_t2b(encoder, color, readback);
        submit_encoder(queue, encoder);

        let got = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let expected =
                    ((col + row * WIDTH as usize + 1) as f32 / 16.0 * 255.0).round() as i32;
                let r = got[row * BYTES_PER_ROW as usize + col * 4] as i32;
                if (r - expected).abs() > 1 {
                    mismatches.push((col, row, expected, r));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "per-texel texture sampling wrong (col,row,expected,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — same per-texel texture sampling as
/// `metal_fragment_samples_bound_texture_per_texel`, but with an EXPLICIT
/// pipeline layout + bind group layout (as the CTS depth staging uses) rather
/// than an auto layout. Pins whether texture/sampler binding indices are
/// resolved correctly for explicit pipeline layouts.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_fragment_samples_bound_texture_explicit_layout() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let v = (col + row * WIDTH as usize + 1) as f32 / 16.0;
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        // Explicit BGL (one sampled-texture entry, fragment-visible) + layout.
        let bgl_entry = native::WGPUBindGroupLayoutEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            visibility: native::WGPUShaderStage_Fragment,
            bindingArraySize: 0,
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
                sampleType: native::WGPUTextureSampleType_UnfilterableFloat,
                viewDimension: native::WGPUTextureViewDimension_2D,
                multisampled: 0,
            },
            storageTexture: native::WGPUStorageTextureBindingLayout {
                nextInChain: std::ptr::null_mut(),
                access: native::WGPUStorageTextureAccess_BindingNotUsed,
                format: native::WGPUTextureFormat_Undefined,
                viewDimension: native::WGPUTextureViewDimension_Undefined,
            },
        };
        let bgl_descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 1,
            entries: &bgl_entry,
        };
        let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &bgl_descriptor);
        assert!(!bgl.is_null());
        let pl_descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &pl_descriptor);
        assert!(!pipeline_layout.is_null());

        let color = create_color_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, SAMPLE_TEXTURE_SHADER);
        let pipeline = create_sample_texture_pipeline_with_layout(device, module, pipeline_layout);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());

        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: input_view,
        };
        let bg_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: bgl,
            entryCount: 1,
            entries: &entry,
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_descriptor);
        assert!(!bind_group.is_null());
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

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
                b: 0.0,
                a: 1.0,
            },
        };
        let attachments = [color_attachment];
        let descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_color_t2b(encoder, color, readback);
        submit_encoder(queue, encoder);

        let got = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let expected =
                    ((col + row * WIDTH as usize + 1) as f32 / 16.0 * 255.0).round() as i32;
                let r = got[row * BYTES_PER_ROW as usize + col * 4] as i32;
                if (r - expected).abs() > 1 {
                    mismatches.push((col, row, expected, r));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "explicit-layout per-texel texture sampling wrong (col,row,expected,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the EXACT CTS depth staging: an explicit pipeline layout +
/// bind group layout, a frag-depth-only fragment that samples a bound
/// `r32float`, rendered into `Depth32Float`, read back per texel. This is the
/// one combination (explicit layout + sample→frag_depth) the other probes don't
/// cover together.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampled_frag_depth_explicit_layout_per_texel() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        let expected_depth =
            |col: usize, row: usize| (col + row * WIDTH as usize + 1) as f32 / 32.0;
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        let bgl = create_fragment_texture_bgl(device);
        let pl_descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &pl_descriptor);
        assert!(!pipeline_layout.is_null());

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_FS);
        let pipeline =
            create_frag_depth_pipeline_with_layout(device, vs_module, fs_module, pipeline_layout);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: input_view,
        };
        let bg_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: bgl,
            entryCount: 1,
            entries: &entry,
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_descriptor);
        assert!(!bind_group.is_null());
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

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
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let got = depths[row * WIDTH as usize + col];
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "explicit-layout sampled frag_depth wrong (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the untested combination: render `sample → frag_depth` into
/// **mip level 1** (2x2) of a 2-mip `Depth32Float`, sampling a 2x2 `r32float`
/// per fragment, then read mip 1's depth back. The CTS varies `mipLevel`, so
/// this combines mip>0 + per-texel sampling + frag_depth.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampled_frag_depth_into_mip_level_1() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let mip_w = WIDTH / 2;
        let mip_h = HEIGHT / 2;
        // 2x2 r32float input (matches mip 1 size).
        let input_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: mip_w,
                height: mip_h,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_R32Float,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let input = yawgpu::wgpuDeviceCreateTexture(device, &input_desc);
        assert!(!input.is_null());
        let expected_depth = |col: usize, row: usize| (col + row * mip_w as usize + 1) as f32 / 8.0;
        let mut floats = vec![0u8; READBACK_SIZE];
        for row in 0..mip_h as usize {
            for col in 0..mip_w as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: mip_h,
        };
        let input_size = native::WGPUExtent3D {
            width: mip_w,
            height: mip_h,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &input_size,
        );

        // 4x4 Depth32Float, 2 mips; render into mip 1 (2x2).
        let depth_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: texture_extent(),
            format: native::WGPUTextureFormat_Depth32Float,
            mipLevelCount: 2,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &depth_desc);
        assert!(!depth.is_null());
        let mip_view_desc = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_Depth32Float,
            dimension: native::WGPUTextureViewDimension_2D,
            baseMipLevel: 1,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_RenderAttachment,
        };
        let depth_view = yawgpu::wgpuTextureCreateView(depth, &mip_view_desc);
        assert!(!depth_view.is_null());

        let bgl = create_fragment_texture_bgl(device);
        let pl_descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &pl_descriptor);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_FS);
        let pipeline =
            create_frag_depth_pipeline_with_layout(device, vs_module, fs_module, pipeline_layout);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: input_view,
        };
        let bg_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: bgl,
            entryCount: 1,
            entries: &entry,
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_descriptor);
        assert!(!bind_group.is_null());

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
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        let source = native::WGPUTexelCopyTextureInfo {
            texture: depth,
            mipLevel: 1,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: mip_h,
            },
            buffer: readback,
        };
        let mip_size = native::WGPUExtent3D {
            width: mip_w,
            height: mip_h,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &mip_size);
        submit_encoder(queue, encoder);

        let mapped = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..mip_h as usize {
            for col in 0..mip_w as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                let got = f32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]);
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "mip>0 sampled frag_depth wrong (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_fragment_texture_bgl(device: native::WGPUDevice) -> native::WGPUBindGroupLayout {
    let bgl_entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Fragment,
        bindingArraySize: 0,
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
            sampleType: native::WGPUTextureSampleType_UnfilterableFloat,
            viewDimension: native::WGPUTextureViewDimension_2D,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    };
    let bgl_descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &bgl_entry,
    };
    let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &bgl_descriptor);
    assert!(!bgl.is_null());
    bgl
}

#[cfg(feature = "metal")]
unsafe fn create_frag_depth_pipeline_with_layout(
    device: native::WGPUDevice,
    vs_module: native::WGPUShaderModule,
    fs_module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
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
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fs_module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
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
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_sample_texture_pipeline_with_layout(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
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
        layout,
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
        depthStencil: std::ptr::null(),
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_r32float_input(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_R32Float,
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
unsafe fn create_sample_texture_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
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
        depthStencil: std::ptr::null(),
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_texture_bind_group(
    device: native::WGPUDevice,
    pipeline: native::WGPURenderPipeline,
    view: native::WGPUTextureView,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    bind_group
}

/// Isolation — round-trip the **stencil** aspect of a packed `Depth24PlusStencil8`
/// texture through a buffer: `writeTexture(StencilOnly)` then
/// `copyTextureToBuffer(StencilOnly)`. Pins the packed-aspect buffer copy path
/// (Metal needs `MTLBlitOption::stencilFromDepthStencil`). Captures device errors.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_packed_stencil_buffer_roundtrips() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst
                | native::WGPUTextureUsage_RenderAttachment,
            dimension: native::WGPUTextureDimension_2D,
            size: texture_extent(),
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!texture.is_null());

        // Stencil is 1 byte/texel; upload 4x4 with bytesPerRow=256.
        let mut stencil = vec![0u8; READBACK_SIZE];
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                stencil[row * BYTES_PER_ROW as usize + col] =
                    (row * WIDTH as usize + col + 1) as u8;
            }
        }
        let dst = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_StencilOnly,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &dst,
            stencil.as_ptr().cast(),
            stencil.len(),
            &layout,
            &texture_extent(),
        );

        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let src = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_StencilOnly,
        };
        let buffer_info = native::WGPUTexelCopyBufferInfo {
            layout,
            buffer: readback,
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &src,
            &buffer_info,
            &texture_extent(),
        );
        submit_encoder(queue, encoder);

        let got = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col;
                if got[off] != stencil[off] {
                    mismatches.push((col, row, stencil[off], got[off]));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "packed stencil round-trip mismatch (col,row,expected,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("lock").is_empty());

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

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

/// Isolation — render a constant `frag_depth` into the **depth plane of a packed
/// `Depth32FloatStencil8`** texture, then read the depth aspect back via
/// `copyTextureToBuffer(DepthOnly)` (needs `MTLBlitOption::DepthFromDepthStencil`).
/// Constant frag_depth isolates the packed depth render + extract from sampling.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_packed_depth_buffer_roundtrips() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_features(
            instance,
            adapter,
            &[native::WGPUFeatureName_Depth32FloatStencil8],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: texture_extent(),
            format: native::WGPUTextureFormat_Depth32FloatStencil8,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!depth.is_null());
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, FRAG_DEPTH_FS);
        let pipeline = create_packed_depth_pipeline(device, vs_module, fs_module);
        let view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Clear,
            stencilStoreOp: native::WGPUStoreOp_Store,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - 0.7).abs() < 1e-5),
            "packed depth aspect not read back: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the exact CTS packed-depth staging: SAMPLE an `r32float` per
/// fragment and write it as `frag_depth` into a **packed `Depth32FloatStencil8`**
/// texture, then read the depth aspect back PER TEXEL via
/// `copyTextureToBuffer(DepthOnly)`. Unlike the constant packed probe (uniform
/// value can't catch a per-texel layout bug), this writes distinct per-texel
/// depths — mirroring `offsets_and_sizes_copy_depth_stencil` format=49 aspect=0.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampled_frag_depth_packed_per_texel() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_features(
            instance,
            adapter,
            &[native::WGPUFeatureName_Depth32FloatStencil8],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        // Mirror the CTS depthValues: index 0 → 1.0, else fmod(0.05*i, 1.0).
        let expected_depth = |col: usize, row: usize| {
            let i = col + row * WIDTH as usize;
            if i.is_multiple_of(40) {
                1.0_f32
            } else {
                (0.05_f32 * i as f32) % 1.0
            }
        };
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: texture_extent(),
            format: native::WGPUTextureFormat_Depth32FloatStencil8,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!depth.is_null());
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_FS);
        let pipeline = create_packed_depth_pipeline(device, vs_module, fs_module);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let bind_group = create_texture_bind_group(device, pipeline, input_view);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        // Mirror the CTS staging exactly: a packed format uses stencilLoadOp=Load
        // (the stencil plane is left untouched while frag_depth writes depth).
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Load,
            stencilStoreOp: native::WGPUStoreOp_Store,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let got = depths[row * WIDTH as usize + col];
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "packed sampled frag_depth not written per texel (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the EXACT failing CTS subcase
/// (`offsets_and_sizes_copy_depth_stencil` format=49 aspect=0 copyDepth=1
/// mip=0): a **3×3** packed `Depth32FloatStencil8`, sampled per-texel frag_depth,
/// then a `copyTextureToBuffer(DepthOnly)` of copySize {3,3,1} with a **tight**
/// readback buffer. The depth aspect is 4 bytes/texel; if the copy buffer-size
/// validation sizes it at the whole-format block (5), the tight buffer is
/// wrongly rejected and the copy never runs — leaving the buffer zeroed. The
/// tight buffer is load-bearing: an oversized one masks the bug.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampled_frag_depth_packed_3x3() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    const W: u32 = 3;
    const H: u32 = 3;
    const BPR: u32 = 256;

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_features(
            instance,
            adapter,
            &[native::WGPUFeatureName_Depth32FloatStencil8],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let extent = native::WGPUExtent3D {
            width: W,
            height: H,
            depthOrArrayLayers: 1,
        };
        let expected_depth = |col: usize, row: usize| {
            let i = col + row * W as usize;
            if i.is_multiple_of(40) {
                1.0_f32
            } else {
                (0.05_f32 * i as f32) % 1.0
            }
        };

        // r32float 3x3 input, written tightly (bytesPerRow = 3*4 = 12).
        let input_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: extent,
            format: native::WGPUTextureFormat_R32Float,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let input = yawgpu::wgpuDeviceCreateTexture(device, &input_desc);
        assert!(!input.is_null());
        let mut floats = vec![0u8; (W * H * 4) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let off = (row * W as usize + col) * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: W * 4,
            rowsPerImage: H,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &extent,
        );

        // Match the CTS createDepthStencilCopyTexture usage exactly.
        let depth_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: extent,
            format: native::WGPUTextureFormat_Depth32FloatStencil8,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &depth_desc);
        assert!(!depth.is_null());
        // TIGHT readback buffer (exactly the CTS size): the depth aspect of a
        // packed format is 4 bytes/texel, so the buffer needs only
        // (H-1)*bytesPerRow + W*4 bytes. An oversized buffer would mask the core
        // bug where the depth aspect was sized at the whole-format block (5).
        const READBACK_LEN: usize = (H as usize - 1) * BPR as usize + W as usize * 4;
        let readback = create_buffer(
            device,
            READBACK_LEN as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_FS);
        // Match the CTS staging exactly: an EXPLICIT pipeline layout (bind group
        // layout with UnfilterableFloat 2D texture) rather than an auto layout.
        let bgl = create_fragment_texture_bgl(device);
        let pipeline_layout_desc = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);
        assert!(!pipeline_layout.is_null());
        let pipeline =
            create_packed_depth_pipeline_with_layout(device, vs_module, fs_module, pipeline_layout);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let bind_group = create_texture_bind_group(device, pipeline, input_view);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Load,
            stencilStoreOp: native::WGPUStoreOp_Store,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        // Mirror the CTS exactly: the staging render is submitted in its OWN
        // command buffer, THEN the t2b runs in a SEPARATE command buffer.
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        submit_encoder(queue, encoder);

        let mut src = depth_texture_copy_info(depth);
        src.aspect = native::WGPUTextureAspect_DepthOnly;
        let dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BPR,
                rowsPerImage: H,
            },
            buffer: readback,
        };
        let copy_encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(copy_encoder, &src, &dst, &extent);
        submit_encoder(queue, copy_encoder);

        let mapped = read_buffer(instance, readback, 0, READBACK_LEN);
        let mut mismatches = Vec::new();
        for row in 0..H as usize {
            for col in 0..W as usize {
                let off = row * BPR as usize + col * 4;
                let got = f32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]);
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "3x3 packed depth wrong (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_packed_depth_pipeline(
    device: native::WGPUDevice,
    vs_module: native::WGPUShaderModule,
    fs_module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32FloatStencil8,
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
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fs_module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
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
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_packed_depth_pipeline_with_layout(
    device: native::WGPUDevice,
    vs_module: native::WGPUShaderModule,
    fs_module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32FloatStencil8,
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
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fs_module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
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
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
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

// Writes depth via @builtin(frag_depth) (no colour output), as the CTS T27
// image_copy depth/stencil tests stage the depth aspect. The CTS uses SEPARATE
// vertex and fragment modules (per-stage MSL path), so this probe does too.
const FRAG_DEPTH_VS: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
"#;
const FRAG_DEPTH_FS: &str = r#"
@fragment
fn fs() -> @builtin(frag_depth) f32 { return 0.7; }
"#;

/// Isolation — stage depth via a fragment shader writing `@builtin(frag_depth)`
/// (the CTS T27 `image_copy` depth/stencil staging), then read the depth aspect
/// back. Pins whether yawgpu honours `frag_depth` (the vertex z is 0.0, so a
/// non-`frag_depth` pipeline would read 0).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_frag_depth_output_writes_depth() {
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

        let vs_module = create_wgsl_module(device, FRAG_DEPTH_VS);
        let fs_module = create_wgsl_module(device, FRAG_DEPTH_FS);
        let pipeline = create_frag_depth_pipeline(device, vs_module, fs_module);
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
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - 0.7).abs() < 1e-5),
            "frag_depth output not written to the depth attachment: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_frag_depth_pipeline(
    device: native::WGPUDevice,
    vs_module: native::WGPUShaderModule,
    fs_module: native::WGPUShaderModule,
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
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fs_module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
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
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

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
    request_device_with_features(instance, adapter, &[])
}

#[cfg(feature = "metal")]
unsafe fn request_device_with_features(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    required_features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: required_features.len(),
        requiredFeatures: required_features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: unsafe { std::mem::zeroed() },
        uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
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

// ---------------------------------------------------------------------------
// F-055 isolation — separate the four stages of the CTS
// `readonly_depth_stencil:sampling_while_testing` case (init writes depth+stencil
// into a 3x3 Depth24PlusStencil8, then both aspects are sampled) so a failure
// localises to write-vs-read and depth-vs-stencil. Mirrors the CTS init exactly
// (point-list, per-instance stencil reference, frag_depth), then:
//   (1) copyTextureToBuffer(StencilOnly) -> verifies the stencil write
//       (the depth aspect of Depth24PlusStencil8 is NOT buffer-copyable per the
//        WebGPU spec, so the depth write is verified transitively by stage (2))
//   (2) sample the depth aspect (texture_2d<f32>) -> verifies the depth write+read
//   (3) sample the stencil aspect (texture_2d<u32>) -> verifies the stencil read
// ---------------------------------------------------------------------------

// CTS init: stencil(x,y)=x+1 via instance index + stencil reference, depth via
// frag_depth = (window_y_center)/10 → texture row r has depth (r+1)/10.
const DS_INIT_SHADER: &str = r#"
@vertex fn vs(@builtin(instance_index) x: u32, @builtin(vertex_index) y: u32) -> @builtin(position) vec4f {
    let texcoord = (vec2f(f32(x), f32(y)) + vec2f(0.5)) / 3.0;
    return vec4f((texcoord * 2.0) - vec2f(1.0), 0.0, 1.0);
}
@fragment fn fs_with_depth(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
    return (pos.y + 0.5) / 10.0;
}
"#;

// Full-screen triangle that samples the depth aspect and writes the depth value
// scaled by 100 (so (r+1)/10 → (r+1)*10) into an r32uint target.
const DS_SAMPLE_DEPTH_SHADER: &str = r#"
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
    let pos = array(vec2f(-3.0, -1.0), vec2f(3.0, -1.0), vec2f(0.0, 2.0));
    return vec4f(pos[id], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) u32 {
    let texel = vec2u(floor(pos.xy));
    return u32(round(textureLoad(depthTex, texel, 0).r * 100.0));
}
"#;

// Full-screen triangle that samples the stencil aspect and writes the raw
// stencil value into an r32uint target.
const DS_SAMPLE_STENCIL_SHADER: &str = r#"
@group(0) @binding(0) var stencilTex: texture_2d<u32>;
@vertex fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
    let pos = array(vec2f(-3.0, -1.0), vec2f(3.0, -1.0), vec2f(0.0, 2.0));
    return vec4f(pos[id], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) u32 {
    let texel = vec2u(floor(pos.xy));
    return textureLoad(stencilTex, texel, 0).r;
}
"#;

/// F-055 isolation across all four stages (write/read × depth/stencil).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_readonly_depth_stencil_isolation() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    const W: u32 = 3;
    const H: u32 = 3;
    const BPR: u32 = 256;
    const RB: usize = (BPR * H) as usize;

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let extent = native::WGPUExtent3D {
            width: W,
            height: H,
            depthOrArrayLayers: 1,
        };

        // 3x3 Depth24PlusStencil8, render-attachment + sampled + copy-src.
        let ds_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: extent,
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let ds = yawgpu::wgpuDeviceCreateTexture(device, &ds_desc);
        assert!(!ds.is_null());
        let ds_full_view = yawgpu::wgpuTextureCreateView(ds, std::ptr::null());

        // ----- init pipeline (point-list, depth-write, stencil replace) -----
        let init_module = create_wgsl_module(device, DS_INIT_SHADER);
        let init_ds_state = native::WGPUDepthStencilState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            depthWriteEnabled: native::WGPUOptionalBool_True,
            depthCompare: native::WGPUCompareFunction_Always,
            stencilFront: native::WGPUStencilFaceState {
                compare: native::WGPUCompareFunction_Always,
                failOp: native::WGPUStencilOperation_Keep,
                depthFailOp: native::WGPUStencilOperation_Keep,
                passOp: native::WGPUStencilOperation_Replace,
            },
            stencilBack: native::WGPUStencilFaceState {
                compare: native::WGPUCompareFunction_Always,
                failOp: native::WGPUStencilOperation_Keep,
                depthFailOp: native::WGPUStencilOperation_Keep,
                passOp: native::WGPUStencilOperation_Replace,
            },
            stencilReadMask: 0xFF,
            stencilWriteMask: 0xFF,
            depthBias: 0,
            depthBiasSlopeScale: 0.0,
            depthBiasClamp: 0.0,
        };
        let mut init_primitive = primitive_state();
        init_primitive.topology = native::WGPUPrimitiveTopology_PointList;
        let init_pipe_desc = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module: init_module,
                entryPoint: string_view("vs"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: init_primitive,
            depthStencil: &init_ds_state,
            multisample: multisample_state(),
            fragment: &native::WGPUFragmentState {
                nextInChain: std::ptr::null_mut(),
                module: init_module,
                entryPoint: string_view("fs_with_depth"),
                constantCount: 0,
                constants: std::ptr::null(),
                targetCount: 0,
                targets: std::ptr::null(),
            },
        };
        let init_pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &init_pipe_desc);
        assert!(!init_pipeline.is_null(), "init pipeline creation failed");

        // ----- Submit 1: init render pass (3 point draws, per-instance stencil) -----
        {
            let ds_att = native::WGPURenderPassDepthStencilAttachment {
                nextInChain: std::ptr::null_mut(),
                view: ds_full_view,
                depthLoadOp: native::WGPULoadOp_Clear,
                depthStoreOp: native::WGPUStoreOp_Store,
                depthClearValue: 0.0,
                depthReadOnly: 0,
                stencilLoadOp: native::WGPULoadOp_Clear,
                stencilStoreOp: native::WGPUStoreOp_Store,
                stencilClearValue: 0,
                stencilReadOnly: 0,
            };
            let pass_desc = native::WGPURenderPassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorAttachmentCount: 0,
                colorAttachments: std::ptr::null(),
                depthStencilAttachment: &ds_att,
                occlusionQuerySet: std::ptr::null(),
                timestampWrites: std::ptr::null(),
            };
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, init_pipeline);
            for i in 0..3u32 {
                yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, i + 1);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, i);
            }
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            submit_encoder(queue, encoder);
        }

        // ----- Submit 2: copy each aspect back to verify the WRITE -----
        let buf_stencil = create_buffer(
            device,
            RB as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        {
            let copy = |encoder, aspect, buffer| {
                let src = native::WGPUTexelCopyTextureInfo {
                    texture: ds,
                    mipLevel: 0,
                    origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                    aspect,
                };
                let info = native::WGPUTexelCopyBufferInfo {
                    layout: native::WGPUTexelCopyBufferLayout {
                        offset: 0,
                        bytesPerRow: BPR,
                        rowsPerImage: H,
                    },
                    buffer,
                };
                yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &info, &extent);
            };
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            // Only the stencil aspect of Depth24PlusStencil8 is buffer-copyable;
            // the depth write is verified end-to-end by the depth *sample* readback
            // (stage 3) below, mirroring the CTS case (which never copies depth).
            copy(encoder, native::WGPUTextureAspect_StencilOnly, buf_stencil);
            submit_encoder(queue, encoder);
        }
        let stencil_bytes = read_buffer(instance, buf_stencil, 0, RB);
        let mut write_stencil_bad = Vec::new();
        for row in 0..H as usize {
            for col in 0..W as usize {
                let s = stencil_bytes[row * BPR as usize + col];
                let want_s = (col + 1) as u8;
                if s != want_s {
                    write_stencil_bad.push((col, row, want_s, s));
                }
            }
        }

        // ----- Submit 3: sample each aspect into an r32uint to verify the READ -----
        let sample_read = |module_src: &str, aspect: native::WGPUTextureAspect| -> Vec<u32> {
            let module = create_wgsl_module(device, module_src);
            let color_target = native::WGPUColorTargetState {
                nextInChain: std::ptr::null_mut(),
                format: native::WGPUTextureFormat_R32Uint,
                blend: std::ptr::null(),
                writeMask: native::WGPUColorWriteMask_All,
            };
            let pipe_desc = native::WGPURenderPipelineDescriptor {
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
                depthStencil: std::ptr::null(),
                multisample: multisample_state(),
                fragment: &native::WGPUFragmentState {
                    nextInChain: std::ptr::null_mut(),
                    module,
                    entryPoint: string_view("fs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    targetCount: 1,
                    targets: &color_target,
                },
            };
            let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &pipe_desc);
            assert!(!pipeline.is_null(), "sample pipeline creation failed");

            // Aspect view + auto bind group.
            let aspect_view_desc = native::WGPUTextureViewDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                dimension: native::WGPUTextureViewDimension_2D,
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: 0,
                arrayLayerCount: 1,
                aspect,
                usage: native::WGPUTextureUsage_None,
            };
            let aspect_view = yawgpu::wgpuTextureCreateView(ds, &aspect_view_desc);
            let bgl = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
            let entry = native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 0,
                buffer: std::ptr::null(),
                offset: 0,
                size: 0,
                sampler: std::ptr::null(),
                textureView: aspect_view,
            };
            let bg_desc = native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: bgl,
                entryCount: 1,
                entries: &entry,
            };
            let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_desc);
            assert!(!bind_group.is_null(), "sample bind group creation failed");

            let result = yawgpu::wgpuDeviceCreateTexture(
                device,
                &native::WGPUTextureDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    usage: native::WGPUTextureUsage_RenderAttachment
                        | native::WGPUTextureUsage_CopySrc,
                    dimension: native::WGPUTextureDimension_2D,
                    size: extent,
                    format: native::WGPUTextureFormat_R32Uint,
                    mipLevelCount: 1,
                    sampleCount: 1,
                    viewFormatCount: 0,
                    viewFormats: std::ptr::null(),
                },
            );
            let result_view = yawgpu::wgpuTextureCreateView(result, std::ptr::null());
            let color_att = native::WGPURenderPassColorAttachment {
                nextInChain: std::ptr::null_mut(),
                view: result_view,
                depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
                resolveTarget: std::ptr::null(),
                loadOp: native::WGPULoadOp_Clear,
                storeOp: native::WGPUStoreOp_Store,
                clearValue: native::WGPUColor {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                },
            };
            let pass_desc = native::WGPURenderPassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorAttachmentCount: 1,
                colorAttachments: &color_att,
                depthStencilAttachment: std::ptr::null(),
                occlusionQuerySet: std::ptr::null(),
                timestampWrites: std::ptr::null(),
            };
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            let buf = create_buffer(
                device,
                RB as u64,
                native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            );
            let src = native::WGPUTexelCopyTextureInfo {
                texture: result,
                mipLevel: 0,
                origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                aspect: native::WGPUTextureAspect_All,
            };
            let info = native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: BPR,
                    rowsPerImage: H,
                },
                buffer: buf,
            };
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &info, &extent);
            submit_encoder(queue, encoder);
            let bytes = read_buffer(instance, buf, 0, RB);
            let mut values = Vec::new();
            for row in 0..H as usize {
                for col in 0..W as usize {
                    let off = row * BPR as usize + col * 4;
                    values.push(u32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]));
                }
            }
            yawgpu::wgpuBufferRelease(buf);
            yawgpu::wgpuTextureViewRelease(result_view);
            yawgpu::wgpuTextureRelease(result);
            yawgpu::wgpuBindGroupRelease(bind_group);
            yawgpu::wgpuBindGroupLayoutRelease(bgl);
            yawgpu::wgpuTextureViewRelease(aspect_view);
            yawgpu::wgpuRenderPipelineRelease(pipeline);
            yawgpu::wgpuShaderModuleRelease(module);
            values
        };

        let depth_read = sample_read(DS_SAMPLE_DEPTH_SHADER, native::WGPUTextureAspect_DepthOnly);
        let stencil_read = sample_read(
            DS_SAMPLE_STENCIL_SHADER,
            native::WGPUTextureAspect_StencilOnly,
        );
        let mut read_depth_bad = Vec::new();
        let mut read_stencil_bad = Vec::new();
        for row in 0..H as usize {
            for col in 0..W as usize {
                let idx = row * W as usize + col;
                let want_d = ((row + 1) * 10) as u32;
                if depth_read[idx] != want_d {
                    read_depth_bad.push((col, row, want_d, depth_read[idx]));
                }
                let want_s = (col + 1) as u32;
                if stencil_read[idx] != want_s {
                    read_stencil_bad.push((col, row, want_s, stencil_read[idx]));
                }
            }
        }

        // Report all four stages together so a single run pins every failure.
        let err = errors.lock().expect("error lock");
        assert!(
            write_stencil_bad.is_empty()
                && read_depth_bad.is_empty()
                && read_stencil_bad.is_empty()
                && err.is_empty(),
            "F-055 isolation:\n  WRITE stencil bad (col,row,want,got): {write_stencil_bad:?}\n  \
             READ depth bad (col,row,want,got): {read_depth_bad:?}\n  \
             READ stencil bad: {read_stencil_bad:?}\n  errors: {err:?}"
        );
        drop(err);

        yawgpu::wgpuBufferRelease(buf_stencil);
        yawgpu::wgpuRenderPipelineRelease(init_pipeline);
        yawgpu::wgpuShaderModuleRelease(init_module);
        yawgpu::wgpuTextureViewRelease(ds_full_view);
        yawgpu::wgpuTextureRelease(ds);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// CTS test pass: read-only DS attachment + concurrent sampling of both aspects,
// no colour output (depth/stencil tested, fragment discards on sample).
const DS_TEST_PASS_SHADER: &str = r#"
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var stencilTex: texture_2d<u32>;
@vertex fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
    let pos = array(vec2f(-3.0, -1.0), vec2f(3.0, -1.0), vec2f(0.0, 2.0));
    return vec4f(pos[id], 0.15, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) {
    let texel = vec2u(floor(pos.xy));
    if textureLoad(stencilTex, texel, 0).r > 2 { discard; }
    if textureLoad(depthTex, texel, 0).r > 0.21 { discard; }
}
"#;

// CTS check pass: re-sample both aspects, write 1 to r32uint on a match.
const DS_CHECK_PASS_SHADER: &str = r#"
@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var stencilTex: texture_2d<u32>;
@vertex fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
    let pos = array(vec2f(-3.0, -1.0), vec2f(3.0, -1.0), vec2f(0.0, 2.0));
    return vec4f(pos[id], 0.15, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) u32 {
    let texel = vec2u(floor(pos.xy));
    let initStencil = texel.x + 1;
    let initDepth = f32(texel.y + 1) / 10.0;
    if textureLoad(stencilTex, texel, 0).r != initStencil { return 0u; }
    if abs(textureLoad(depthTex, texel, 0).r - initDepth) > 0.01 { return 0u; }
    return 1u;
}
"#;

/// F-055 full reproduction — the exact CTS structure: ONE command encoder, three
/// passes (init write, read-only-DS + concurrent sample, check sample → result),
/// ONE submit. The isolation probe (separate submits) passes; this pins whether
/// the same-command-buffer multi-pass structure (read-only-DS-while-sampling +
/// cross-pass write→sample) is what fails. Captures the device error sink.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_readonly_depth_stencil_single_submit() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    const W: u32 = 3;
    const H: u32 = 3;
    const BPR: u32 = 256;
    const RB: usize = (BPR * H) as usize;

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let extent = native::WGPUExtent3D {
            width: W,
            height: H,
            depthOrArrayLayers: 1,
        };
        let ds = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_RenderAttachment
                    | native::WGPUTextureUsage_TextureBinding,
                dimension: native::WGPUTextureDimension_2D,
                size: extent,
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        let ds_full_view = yawgpu::wgpuTextureCreateView(ds, std::ptr::null());
        let aspect_view = |aspect| {
            yawgpu::wgpuTextureCreateView(
                ds,
                &native::WGPUTextureViewDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    format: native::WGPUTextureFormat_Depth24PlusStencil8,
                    dimension: native::WGPUTextureViewDimension_2D,
                    baseMipLevel: 0,
                    mipLevelCount: 1,
                    baseArrayLayer: 0,
                    arrayLayerCount: 1,
                    aspect,
                    usage: native::WGPUTextureUsage_None,
                },
            )
        };
        let depth_view = aspect_view(native::WGPUTextureAspect_DepthOnly);
        let stencil_view = aspect_view(native::WGPUTextureAspect_StencilOnly);

        let result = yawgpu::wgpuDeviceCreateTexture(
            device,
            &native::WGPUTextureDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
                dimension: native::WGPUTextureDimension_2D,
                size: extent,
                format: native::WGPUTextureFormat_R32Uint,
                mipLevelCount: 1,
                sampleCount: 1,
                viewFormatCount: 0,
                viewFormats: std::ptr::null(),
            },
        );
        let result_view = yawgpu::wgpuTextureCreateView(result, std::ptr::null());

        // ----- pipelines -----
        let init_module = create_wgsl_module(device, DS_INIT_SHADER);
        let test_module = create_wgsl_module(device, DS_TEST_PASS_SHADER);
        let check_module = create_wgsl_module(device, DS_CHECK_PASS_SHADER);

        let stencil_face = native::WGPUStencilFaceState {
            compare: native::WGPUCompareFunction_Always,
            failOp: native::WGPUStencilOperation_Keep,
            depthFailOp: native::WGPUStencilOperation_Keep,
            passOp: native::WGPUStencilOperation_Replace,
        };
        let init_ds_state = native::WGPUDepthStencilState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            depthWriteEnabled: native::WGPUOptionalBool_True,
            depthCompare: native::WGPUCompareFunction_Always,
            stencilFront: stencil_face,
            stencilBack: stencil_face,
            stencilReadMask: 0xFF,
            stencilWriteMask: 0xFF,
            depthBias: 0,
            depthBiasSlopeScale: 0.0,
            depthBiasClamp: 0.0,
        };
        let mut point_prim = primitive_state();
        point_prim.topology = native::WGPUPrimitiveTopology_PointList;
        let init_pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
            device,
            &native::WGPURenderPipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null(),
                vertex: native::WGPUVertexState {
                    nextInChain: std::ptr::null_mut(),
                    module: init_module,
                    entryPoint: string_view("vs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    bufferCount: 0,
                    buffers: std::ptr::null(),
                },
                primitive: point_prim,
                depthStencil: &init_ds_state,
                multisample: multisample_state(),
                fragment: &native::WGPUFragmentState {
                    nextInChain: std::ptr::null_mut(),
                    module: init_module,
                    entryPoint: string_view("fs_with_depth"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    targetCount: 0,
                    targets: std::ptr::null(),
                },
            },
        );

        // test pipeline: read-only DS, no colour output.
        let keep_face = native::WGPUStencilFaceState {
            compare: native::WGPUCompareFunction_LessEqual,
            failOp: native::WGPUStencilOperation_Keep,
            depthFailOp: native::WGPUStencilOperation_Keep,
            passOp: native::WGPUStencilOperation_Keep,
        };
        let test_ds_state = native::WGPUDepthStencilState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            depthWriteEnabled: native::WGPUOptionalBool_False,
            depthCompare: native::WGPUCompareFunction_LessEqual,
            stencilFront: keep_face,
            stencilBack: keep_face,
            stencilReadMask: 0xFF,
            stencilWriteMask: 0xFF,
            depthBias: 0,
            depthBiasSlopeScale: 0.0,
            depthBiasClamp: 0.0,
        };
        let test_pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
            device,
            &native::WGPURenderPipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null(),
                vertex: native::WGPUVertexState {
                    nextInChain: std::ptr::null_mut(),
                    module: test_module,
                    entryPoint: string_view("vs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    bufferCount: 0,
                    buffers: std::ptr::null(),
                },
                primitive: primitive_state(),
                depthStencil: &test_ds_state,
                multisample: multisample_state(),
                fragment: &native::WGPUFragmentState {
                    nextInChain: std::ptr::null_mut(),
                    module: test_module,
                    entryPoint: string_view("fs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    targetCount: 0,
                    targets: std::ptr::null(),
                },
            },
        );

        let check_color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_R32Uint,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let check_pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
            device,
            &native::WGPURenderPipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null(),
                vertex: native::WGPUVertexState {
                    nextInChain: std::ptr::null_mut(),
                    module: check_module,
                    entryPoint: string_view("vs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    bufferCount: 0,
                    buffers: std::ptr::null(),
                },
                primitive: primitive_state(),
                depthStencil: std::ptr::null(),
                multisample: multisample_state(),
                fragment: &native::WGPUFragmentState {
                    nextInChain: std::ptr::null_mut(),
                    module: check_module,
                    entryPoint: string_view("fs"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                    targetCount: 1,
                    targets: &check_color_target,
                },
            },
        );
        assert!(!init_pipeline.is_null() && !test_pipeline.is_null() && !check_pipeline.is_null());

        // ----- bind groups (test + check share the same aspect views) -----
        let make_bg = |pipeline| {
            let bgl = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
            let entries = [
                native::WGPUBindGroupEntry {
                    nextInChain: std::ptr::null_mut(),
                    binding: 0,
                    buffer: std::ptr::null(),
                    offset: 0,
                    size: 0,
                    sampler: std::ptr::null(),
                    textureView: depth_view,
                },
                native::WGPUBindGroupEntry {
                    nextInChain: std::ptr::null_mut(),
                    binding: 1,
                    buffer: std::ptr::null(),
                    offset: 0,
                    size: 0,
                    sampler: std::ptr::null(),
                    textureView: stencil_view,
                },
            ];
            let bg = yawgpu::wgpuDeviceCreateBindGroup(
                device,
                &native::WGPUBindGroupDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    layout: bgl,
                    entryCount: 2,
                    entries: entries.as_ptr(),
                },
            );
            yawgpu::wgpuBindGroupLayoutRelease(bgl);
            assert!(!bg.is_null());
            bg
        };
        let test_bg = make_bg(test_pipeline);
        let check_bg = make_bg(check_pipeline);

        // ----- ONE encoder, THREE passes, ONE submit -----
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        // Pass 1: init.
        {
            let ds_att = native::WGPURenderPassDepthStencilAttachment {
                nextInChain: std::ptr::null_mut(),
                view: ds_full_view,
                depthLoadOp: native::WGPULoadOp_Clear,
                depthStoreOp: native::WGPUStoreOp_Store,
                depthClearValue: 0.0,
                depthReadOnly: 0,
                stencilLoadOp: native::WGPULoadOp_Clear,
                stencilStoreOp: native::WGPUStoreOp_Store,
                stencilClearValue: 0,
                stencilReadOnly: 0,
            };
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
                encoder,
                &native::WGPURenderPassDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    colorAttachmentCount: 0,
                    colorAttachments: std::ptr::null(),
                    depthStencilAttachment: &ds_att,
                    occlusionQuerySet: std::ptr::null(),
                    timestampWrites: std::ptr::null(),
                },
            );
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, init_pipeline);
            for i in 0..3u32 {
                yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, i + 1);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, i);
            }
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
        // Pass 2: read-only DS + concurrent sample.
        {
            let ds_att = native::WGPURenderPassDepthStencilAttachment {
                nextInChain: std::ptr::null_mut(),
                view: ds_full_view,
                depthLoadOp: native::WGPULoadOp_Undefined,
                depthStoreOp: native::WGPUStoreOp_Undefined,
                depthClearValue: 0.0,
                depthReadOnly: 1,
                stencilLoadOp: native::WGPULoadOp_Undefined,
                stencilStoreOp: native::WGPUStoreOp_Undefined,
                stencilClearValue: 0,
                stencilReadOnly: 1,
            };
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
                encoder,
                &native::WGPURenderPassDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    colorAttachmentCount: 0,
                    colorAttachments: std::ptr::null(),
                    depthStencilAttachment: &ds_att,
                    occlusionQuerySet: std::ptr::null(),
                    timestampWrites: std::ptr::null(),
                },
            );
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, test_pipeline);
            yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, 2);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, test_bg, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
        // Pass 3: check → result.
        {
            let color_att = native::WGPURenderPassColorAttachment {
                nextInChain: std::ptr::null_mut(),
                view: result_view,
                depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
                resolveTarget: std::ptr::null(),
                loadOp: native::WGPULoadOp_Clear,
                storeOp: native::WGPUStoreOp_Store,
                clearValue: native::WGPUColor {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                },
            };
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
                encoder,
                &native::WGPURenderPassDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    colorAttachmentCount: 1,
                    colorAttachments: &color_att,
                    depthStencilAttachment: std::ptr::null(),
                    occlusionQuerySet: std::ptr::null(),
                    timestampWrites: std::ptr::null(),
                },
            );
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, check_pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, check_bg, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
        submit_encoder(queue, encoder);

        // Copy result → buffer and read.
        let buf = create_buffer(
            device,
            RB as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        {
            let src = native::WGPUTexelCopyTextureInfo {
                texture: result,
                mipLevel: 0,
                origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
                aspect: native::WGPUTextureAspect_All,
            };
            let info = native::WGPUTexelCopyBufferInfo {
                layout: native::WGPUTexelCopyBufferLayout {
                    offset: 0,
                    bytesPerRow: BPR,
                    rowsPerImage: H,
                },
                buffer: buf,
            };
            let enc = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &info, &extent);
            submit_encoder(queue, enc);
        }
        let bytes = read_buffer(instance, buf, 0, RB);
        let mut bad = Vec::new();
        for row in 0..H as usize {
            for col in 0..W as usize {
                let off = row * BPR as usize + col * 4;
                let v = u32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                if v != 1 {
                    bad.push((col, row, v));
                }
            }
        }
        let err = errors.lock().expect("error lock");
        assert!(
            bad.is_empty() && err.is_empty(),
            "F-055 single-submit: result texels != 1 (col,row,got): {bad:?}; errors: {err:?}"
        );
        drop(err);

        yawgpu::wgpuBufferRelease(buf);
        yawgpu::wgpuBindGroupRelease(check_bg);
        yawgpu::wgpuBindGroupRelease(test_bg);
        yawgpu::wgpuRenderPipelineRelease(check_pipeline);
        yawgpu::wgpuRenderPipelineRelease(test_pipeline);
        yawgpu::wgpuRenderPipelineRelease(init_pipeline);
        yawgpu::wgpuShaderModuleRelease(check_module);
        yawgpu::wgpuShaderModuleRelease(test_module);
        yawgpu::wgpuShaderModuleRelease(init_module);
        yawgpu::wgpuTextureViewRelease(result_view);
        yawgpu::wgpuTextureRelease(result);
        yawgpu::wgpuTextureViewRelease(stencil_view);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(ds_full_view);
        yawgpu::wgpuTextureRelease(ds);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
