#![cfg(feature = "vulkan")]
//! Block 73 E1-E10: compressed copies and sampling on a real Vulkan device.
#[path = "common/vulkan.rs"]
mod vulkan;
use vulkan::*;
use yawgpu::native;

const BC: (native::WGPUFeatureName, &str) = (
    native::WGPUFeatureName_TextureCompressionBC,
    "texture-compression-bc",
);
const BC3D: (native::WGPUFeatureName, &str) = (
    native::WGPUFeatureName_TextureCompressionBCSliced3D,
    "texture-compression-bc-sliced-3d",
);
const ETC: (native::WGPUFeatureName, &str) = (
    native::WGPUFeatureName_TextureCompressionETC2,
    "texture-compression-etc2",
);
const ASTC: (native::WGPUFeatureName, &str) = (
    native::WGPUFeatureName_TextureCompressionASTC,
    "texture-compression-astc",
);
const ASTC3D: (native::WGPUFeatureName, &str) = (
    native::WGPUFeatureName_TextureCompressionASTCSliced3D,
    "texture-compression-astc-sliced-3d",
);
const RED: [u8; 8] = [0, 0xf8, 0, 0xf8, 0, 0, 0, 0];
const GREEN: [u8; 8] = [0xe0, 7, 0xe0, 7, 0, 0, 0, 0];
const BLUE: [u8; 8] = [0x1f, 0, 0x1f, 0, 0, 0, 0, 0];

fn extent(width: u32, height: u32, depth: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth,
    }
}
fn info(
    texture: native::WGPUTexture,
    mip: u32,
    origin: [u32; 3],
) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip,
        origin: native::WGPUOrigin3D {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        aspect: native::WGPUTextureAspect_All,
    }
}
unsafe fn texture(
    ctx: &Ctx,
    format: native::WGPUTextureFormat,
    size: native::WGPUExtent3D,
    mips: u32,
    is_3d: bool,
    sampled: bool,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        usage: native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_CopyDst
            | if sampled {
                native::WGPUTextureUsage_TextureBinding
            } else {
                0
            },
        dimension: if is_3d {
            native::WGPUTextureDimension_3D
        } else {
            native::WGPUTextureDimension_2D
        },
        size,
        format,
        mipLevelCount: mips,
        sampleCount: 1,
        ..std::mem::zeroed()
    };
    let result = yawgpu::wgpuDeviceCreateTexture(ctx.device, &descriptor);
    ctx.assert_clean();
    assert!(!result.is_null());
    result
}
unsafe fn write(
    ctx: &Ctx,
    dst: native::WGPUTexelCopyTextureInfo,
    size: native::WGPUExtent3D,
    bytes: &[u8],
    row_bytes: u32,
    rows: u32,
) {
    yawgpu::wgpuQueueWriteTexture(
        ctx.queue,
        &dst,
        bytes.as_ptr().cast(),
        bytes.len(),
        &native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: row_bytes,
            rowsPerImage: rows,
        },
        &size,
    );
    ctx.assert_clean();
}
unsafe fn read(
    ctx: &Ctx,
    src: native::WGPUTexelCopyTextureInfo,
    size: native::WGPUExtent3D,
    row_bytes: usize,
    rows: u32,
) -> Vec<u8> {
    let len = 256 * rows as usize * size.depthOrArrayLayers as usize;
    let buffer = create_buffer(
        ctx.device,
        len as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
    let dst = native::WGPUTexelCopyBufferInfo {
        buffer,
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: 256,
            rowsPerImage: rows,
        },
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &size);
    ctx.submit(encoder);
    let padded = read_buffer(ctx.instance, buffer, len);
    let result = padded
        .chunks_exact(256)
        .flat_map(|row| row[..row_bytes].iter().copied())
        .collect();
    yawgpu::wgpuBufferRelease(buffer);
    result
}
unsafe fn roundtrip(
    ctx: &Ctx,
    format: native::WGPUTextureFormat,
    name: &str,
    block: u32,
    bytes: u32,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let size = extent(block * 2, block * 2, 1);
        let tex = texture(ctx, format, size, 1, false, false);
        let data: Vec<_> = (0..bytes * 4)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
            .collect();
        write(ctx, info(tex, 0, [0; 3]), size, &data, bytes * 2, 2);
        assert_eq!(
            read(ctx, info(tex, 0, [0; 3]), size, (bytes * 2) as usize, 2),
            data,
            "format {name}"
        );
        yawgpu::wgpuTextureRelease(tex);
    }));
    if let Err(error) = result {
        let detail = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        panic!("format {name}: {detail}");
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn e1_all_bc_formats_multiblock_roundtrip() {
    with_device(&[BC], |ctx| unsafe {
        use native::*;
        for (format, name, bytes) in [
            (WGPUTextureFormat_BC1RGBAUnorm, "bc1-rgba-unorm", 8),
            (WGPUTextureFormat_BC1RGBAUnormSrgb, "bc1-rgba-unorm-srgb", 8),
            (WGPUTextureFormat_BC2RGBAUnorm, "bc2-rgba-unorm", 16),
            (
                WGPUTextureFormat_BC2RGBAUnormSrgb,
                "bc2-rgba-unorm-srgb",
                16,
            ),
            (WGPUTextureFormat_BC3RGBAUnorm, "bc3-rgba-unorm", 16),
            (
                WGPUTextureFormat_BC3RGBAUnormSrgb,
                "bc3-rgba-unorm-srgb",
                16,
            ),
            (WGPUTextureFormat_BC4RUnorm, "bc4-r-unorm", 8),
            (WGPUTextureFormat_BC4RSnorm, "bc4-r-snorm", 8),
            (WGPUTextureFormat_BC5RGUnorm, "bc5-rg-unorm", 16),
            (WGPUTextureFormat_BC5RGSnorm, "bc5-rg-snorm", 16),
            (WGPUTextureFormat_BC6HRGBUfloat, "bc6h-rgb-ufloat", 16),
            (WGPUTextureFormat_BC6HRGBFloat, "bc6h-rgb-float", 16),
            (WGPUTextureFormat_BC7RGBAUnorm, "bc7-rgba-unorm", 16),
            (
                WGPUTextureFormat_BC7RGBAUnormSrgb,
                "bc7-rgba-unorm-srgb",
                16,
            ),
        ] {
            roundtrip(ctx, format, name, 4, bytes);
        }
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e2_bc1_physical_mip_chain_roundtrip() {
    with_device(&[BC], |ctx| unsafe {
        let tex = texture(
            ctx,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            extent(12, 12, 1),
            4,
            false,
            false,
        );
        let mip_data: Vec<_> = [12, 8, 4, 4]
            .into_iter()
            .enumerate()
            .map(|(mip, physical)| {
                let rows = physical / 4;
                (
                    physical,
                    rows,
                    vec![mip as u8 + 37; (rows * rows * 8) as usize],
                )
            })
            .collect();
        for (mip, &(physical, rows, ref data)) in mip_data.iter().enumerate() {
            write(
                ctx,
                info(tex, mip as u32, [0; 3]),
                extent(physical, physical, 1),
                data,
                rows * 8,
                rows,
            );
        }
        for (mip, &(physical, rows, ref data)) in mip_data.iter().enumerate() {
            assert_eq!(
                read(
                    ctx,
                    info(tex, mip as u32, [0; 3]),
                    extent(physical, physical, 1),
                    (rows * 8) as usize,
                    rows
                ),
                *data,
                "mip {mip}"
            );
        }
        yawgpu::wgpuTextureRelease(tex);
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e3_bc1_texture_copy_preserves_untouched_blocks() {
    with_device(&[BC], |ctx| unsafe {
        let size = extent(16, 16, 1);
        let src = texture(
            ctx,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            size,
            1,
            false,
            false,
        );
        let dst = texture(
            ctx,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            size,
            1,
            false,
            false,
        );
        let data: Vec<_> = (0..128).map(|i| i as u8 + 1).collect();
        write(ctx, info(src, 0, [0; 3]), size, &data, 32, 4);
        write(ctx, info(dst, 0, [0; 3]), size, &[0; 128], 32, 4);
        let enc = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(
            enc,
            &info(src, 0, [4, 4, 0]),
            &info(dst, 0, [8, 4, 0]),
            &extent(8, 8, 1),
        );
        ctx.submit(enc);
        let mut expected = vec![0; 128];
        for row in 1..3 {
            expected[row * 32 + 16..row * 32 + 32]
                .copy_from_slice(&data[row * 32 + 8..row * 32 + 24]);
        }
        assert_eq!(read(ctx, info(dst, 0, [0; 3]), size, 32, 4), expected);
        yawgpu::wgpuTextureRelease(src);
        yawgpu::wgpuTextureRelease(dst);
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e4_bc1_sampled_render() {
    sampled(false, false);
}
#[test]
#[ignore = "manual real-backend test"]
fn e5_bc1_srgb_sampled_render() {
    sampled(true, false);
}
#[test]
#[ignore = "manual real-backend test"]
fn e6_bc1_3d_write_and_buffer_copy_roundtrip() {
    with_device(&[BC, BC3D], |ctx| unsafe {
        let tex = texture(
            ctx,
            native::WGPUTextureFormat_BC1RGBAUnorm,
            extent(4, 4, 3),
            1,
            true,
            false,
        );
        let blocks = [RED, GREEN, BLUE].concat();
        write(ctx, info(tex, 0, [0; 3]), extent(4, 4, 3), &blocks, 8, 1);
        for z in 0..3 {
            assert_eq!(
                read(ctx, info(tex, 0, [0, 0, z]), extent(4, 4, 1), 8, 1),
                blocks[z as usize * 8..z as usize * 8 + 8],
                "slice {z}"
            );
        }
        // Explicit B2T with a nonzero depth origin and padded block rows.
        let mut padded = vec![0; 1024];
        padded[..8].copy_from_slice(&RED);
        padded[512..520].copy_from_slice(&GREEN);
        let upload = create_buffer(
            ctx.device,
            1024,
            native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(ctx.queue, upload, 0, padded.as_ptr().cast(), padded.len());
        let enc = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let src = native::WGPUTexelCopyBufferInfo {
            buffer: upload,
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: 256,
                rowsPerImage: 2,
            },
        };
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(
            enc,
            &src,
            &info(tex, 0, [0, 0, 1]),
            &extent(4, 4, 2),
        );
        ctx.submit(enc);
        for (z, expected) in [RED, RED, GREEN].iter().enumerate() {
            assert_eq!(
                read(ctx, info(tex, 0, [0, 0, z as u32]), extent(4, 4, 1), 8, 1),
                expected,
                "B2T slice {z}"
            );
        }
        yawgpu::wgpuBufferRelease(upload);
        yawgpu::wgpuTextureRelease(tex);
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e7_bc1_3d_sampled_slice_centres() {
    sampled(false, true);
}
#[test]
#[ignore = "manual real-backend test"]
fn e8_etc2_astc_multiblock_roundtrip() {
    with_device(&[ETC], |ctx| unsafe {
        roundtrip(
            ctx,
            native::WGPUTextureFormat_ETC2RGB8Unorm,
            "etc2-rgb8unorm",
            4,
            8,
        );
        roundtrip(
            ctx,
            native::WGPUTextureFormat_EACR11Unorm,
            "eac-r11unorm",
            4,
            8,
        );
    });
    with_device(&[ASTC], |ctx| unsafe {
        roundtrip(
            ctx,
            native::WGPUTextureFormat_ASTC4x4Unorm,
            "astc-4x4-unorm",
            4,
            16,
        );
        roundtrip(
            ctx,
            native::WGPUTextureFormat_ASTC8x8Unorm,
            "astc-8x8-unorm",
            8,
            16,
        );
        roundtrip(
            ctx,
            native::WGPUTextureFormat_ASTC12x12Unorm,
            "astc-12x12-unorm",
            12,
            16,
        );
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e9_astc_3d_roundtrip() {
    with_device(&[ASTC, ASTC3D], |ctx| unsafe {
        let tex = texture(
            ctx,
            native::WGPUTextureFormat_ASTC4x4Unorm,
            extent(8, 8, 3),
            1,
            true,
            false,
        );
        let data: Vec<_> = (0..192).map(|i| i as u8).collect();
        write(ctx, info(tex, 0, [0; 3]), extent(8, 8, 3), &data, 32, 2);
        for z in 0..3 {
            assert_eq!(
                read(ctx, info(tex, 0, [0, 0, z]), extent(8, 8, 1), 32, 2),
                data[z as usize * 64..z as usize * 64 + 64],
                "ASTC slice {z}"
            );
        }
        yawgpu::wgpuTextureRelease(tex);
    });
}
#[test]
#[ignore = "manual real-backend test"]
fn e10_bc1_texture_copy_mismatched_logical_mip_edges() {
    with_device(&[BC], |ctx| unsafe {
        for layers in [1, 3] {
            let size = extent(16, 16, layers);
            let src = texture(
                ctx,
                native::WGPUTextureFormat_BC1RGBAUnorm,
                size,
                1,
                false,
                false,
            );
            let dst = texture(
                ctx,
                native::WGPUTextureFormat_BC1RGBAUnorm,
                extent(60, 60, layers),
                3,
                false,
                false,
            );
            let data: Vec<_> = (0..128 * layers)
                .map(|i| (i as u8).wrapping_mul(17).wrapping_add((i / 128) as u8))
                .collect();
            write(ctx, info(src, 0, [0; 3]), size, &data, 32, 4);

            // The destination mip is logically 15x15 but physically 16x16.
            let enc = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
            yawgpu::wgpuCommandEncoderCopyTextureToTexture(
                enc,
                &info(src, 0, [0; 3]),
                &info(dst, 2, [0; 3]),
                &size,
            );
            ctx.submit(enc);
            assert_eq!(
                read(ctx, info(dst, 2, [0; 3]), size, 32, 4),
                data,
                "16x16 mip 0 to 60x60 mip 2, {layers} layers"
            );
            ctx.assert_clean();

            // Reverse the edge conversion into an independently zeroed image.
            write(ctx, info(src, 0, [0; 3]), size, &vec![0; data.len()], 32, 4);
            let enc = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
            yawgpu::wgpuCommandEncoderCopyTextureToTexture(
                enc,
                &info(dst, 2, [0; 3]),
                &info(src, 0, [0; 3]),
                &size,
            );
            ctx.submit(enc);
            assert_eq!(
                read(ctx, info(src, 0, [0; 3]), size, 32, 4),
                data,
                "reverse logical edge conversion, {layers} layers"
            );

            // Consecutive same-image copies exercise restoration of tracked layouts.
            write(ctx, info(dst, 2, [0; 3]), size, &vec![0; data.len()], 32, 4);
            write(ctx, info(dst, 0, [0; 3]), size, &data, 32, 4);
            let enc = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
            for (source_mip, destination_mip) in [(0, 2), (2, 0)] {
                yawgpu::wgpuCommandEncoderCopyTextureToTexture(
                    enc,
                    &info(dst, source_mip, [0; 3]),
                    &info(dst, destination_mip, [0; 3]),
                    &size,
                );
            }
            ctx.submit(enc);
            for mip in [0, 2] {
                assert_eq!(
                    read(ctx, info(dst, mip, [0; 3]), size, 32, 4),
                    data,
                    "same-image logical edge conversion, mip {mip}, {layers} layers"
                );
            }
            ctx.assert_clean();
            yawgpu::wgpuTextureRelease(src);
            yawgpu::wgpuTextureRelease(dst);
        }
    });
}

fn sampled(srgb: bool, is_3d: bool) {
    let features = if is_3d { vec![BC, BC3D] } else { vec![BC] };
    with_device(&features, |ctx| unsafe {
        let depth = if is_3d { 3 } else { 1 };
        let format = if srgb {
            native::WGPUTextureFormat_BC1RGBAUnormSrgb
        } else {
            native::WGPUTextureFormat_BC1RGBAUnorm
        };
        let tex = texture(ctx, format, extent(4, 4, depth), 1, is_3d, true);
        let blocks = if is_3d {
            [RED, GREEN, BLUE].concat()
        } else {
            RED.to_vec()
        };
        write(
            ctx,
            info(tex, 0, [0; 3]),
            extent(4, 4, depth),
            &blocks,
            8,
            1,
        );
        let view = yawgpu::wgpuTextureCreateView(tex, std::ptr::null());
        let sampler = yawgpu::wgpuDeviceCreateSampler(ctx.device, std::ptr::null());
        let dimension = if is_3d { "3d" } else { "2d" };
        let coordinate = if is_3d {
            "vec3<f32>(0.5, 0.5, (floor(position.x / 4.0) + 0.5) / 3.0)"
        } else {
            "vec2<f32>(0.5, 0.5)"
        };
        let shader = format!(
            r#"
@group(0) @binding(0) var image: texture_{dimension}<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
    let positions = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
    return vec4<f32>(positions[index],0.0,1.0);
}}
@fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {{
    return textureSampleLevel(image, image_sampler, {coordinate}, 0.0);
}}
"#
        );
        let module = create_wgsl_module(ctx.device, &shader);
        let pipeline = create_raster_pipeline(
            ctx.device,
            module,
            native::WGPUFrontFace_CCW,
            native::WGPUCullMode_None,
        );
        ctx.assert_clean();
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        let entries = [
            native::WGPUBindGroupEntry {
                binding: 0,
                textureView: view,
                ..std::mem::zeroed()
            },
            native::WGPUBindGroupEntry {
                binding: 1,
                sampler,
                ..std::mem::zeroed()
            },
        ];
        let group = yawgpu::wgpuDeviceCreateBindGroup(
            ctx.device,
            &native::WGPUBindGroupDescriptor {
                layout,
                entryCount: 2,
                entries: entries.as_ptr(),
                ..std::mem::zeroed()
            },
        );
        let size = extent(4 * depth, 4, 1);
        let target = yawgpu::wgpuDeviceCreateTexture(
            ctx.device,
            &native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_2D,
                size,
                format: native::WGPUTextureFormat_RGBA8Unorm,
                mipLevelCount: 1,
                sampleCount: 1,
                usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
                ..std::mem::zeroed()
            },
        );
        let target_view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_color_pass(encoder, target_view);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        ctx.submit(encoder);
        let pixels = read(
            ctx,
            info(target, 0, [0; 3]),
            size,
            (size.width * 4) as usize,
            4,
        );
        let colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            let slice = (index % size.width as usize) / 4;
            assert_eq!(
                pixel, colors[slice],
                "srgb={srgb}, 3d={is_3d}, pixel {index}, slice {slice}"
            );
        }
        yawgpu::wgpuTextureViewRelease(target_view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(tex);
    });
}
