//! Shared body of the Block 104 lazy zero-init e2e
//! (`specs/blocks/104-lazy-zero-init-stage2.md`), included by
//! `e2e_metal_lazy_init.rs` and `e2e_vulkan_lazy_init.rs` through `#[path]`:
//! every uninitialized subresource reads as zero — through a sampled
//! binding, for depth / stencil formats, for a multisampled attachment, for
//! a compressed format and after `storeOp: Discard` — while initialized
//! neighbours keep their canary and a partial write survives a later read
//! (the F-138 inverse-bug guard). Each scenario takes the real backend to
//! probe and the `YAWGPU_INSTANCE_BACKEND_*` selector to build the instance.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{YaWGPUInstanceBackendSelect, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const CANARY: u8 = 0xC5;

/// Loads every texel of the bound mip into a storage buffer (rgba8 as floats).
const LOAD_SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(tex);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    out[id.y * dims.x + id.x] = textureLoad(tex, vec2<i32>(id.xy), 0);
}
"#;

pub fn lazy_init_sampled_read_of_uninitialized_mip_is_zero_and_canary_mips_survive(
    backend: RealBackend,
    select: u32,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        // 8x8 rgba8unorm with 3 mips; canary written to mips 0 and 2, mip 1 untouched.
        let texture = s.create_texture(
            native::WGPUTextureFormat_RGBA8Unorm,
            8,
            8,
            3,
            1,
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        );
        s.write_canary(texture, 0, 8, 8, 4);
        s.write_canary(texture, 2, 2, 2, 4);

        let mip1 = s.load_mip_as_floats(texture, 1, 4, 4);
        assert!(
            mip1.iter().all(|&v| v == 0.0),
            "uninitialized mip 1 must read zero: {mip1:?}"
        );
        let mip0 = s.load_mip_as_floats(texture, 0, 8, 8);
        let mip2 = s.load_mip_as_floats(texture, 2, 2, 2);
        let canary = f32::from(CANARY) / 255.0;
        assert!(
            mip0.iter().all(|&v| (v - canary).abs() < 0.01),
            "mip 0 canary clobbered: {:?}",
            &mip0[..8]
        );
        assert!(
            mip2.iter().all(|&v| (v - canary).abs() < 0.01),
            "mip 2 canary clobbered: {mip2:?}"
        );
        s.assert_no_errors();
        yawgpu::wgpuTextureRelease(texture);
        s.release();
    }
}

pub fn lazy_init_depth_and_stencil_copies_read_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        // depth32float: 4 bytes/texel, copyable to a buffer.
        let depth = s.create_texture(
            native::WGPUTextureFormat_Depth32Float,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let bytes = s.copy_to_buffer(depth, native::WGPUTextureAspect_DepthOnly, 4, 4, 4);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "depth32float must read zero: {bytes:?}"
        );
        yawgpu::wgpuTextureRelease(depth);

        // stencil8: 1 byte/texel.
        let stencil = s.create_texture(
            native::WGPUTextureFormat_Stencil8,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let bytes = s.copy_to_buffer(stencil, native::WGPUTextureAspect_StencilOnly, 4, 4, 1);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "stencil8 must read zero: {bytes:?}"
        );
        yawgpu::wgpuTextureRelease(stencil);

        // depth24plus-stencil8: only the stencil aspect is copyable.
        let combined = s.create_texture(
            native::WGPUTextureFormat_Depth24PlusStencil8,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let bytes = s.copy_to_buffer(combined, native::WGPUTextureAspect_StencilOnly, 4, 4, 1);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "depth24plus-stencil8 stencil aspect must read zero: {bytes:?}"
        );
        yawgpu::wgpuTextureRelease(combined);
        s.assert_no_errors();
        s.release();
    }
}

/// Phase Review M3: a depth / stencil texture created WITHOUT
/// `RenderAttachment` cannot be cleared through a render pass, so the HAL
/// must take its blit / clear-image path; the aspects still read zero.
pub fn lazy_init_non_renderable_depth_stencil_copies_read_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let usage = native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopySrc;
        for (format, aspect, bytes_per_texel, name) in [
            (
                native::WGPUTextureFormat_Depth32Float,
                native::WGPUTextureAspect_DepthOnly,
                4,
                "depth32float",
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUTextureAspect_StencilOnly,
                1,
                "stencil8",
            ),
            (
                native::WGPUTextureFormat_Depth24PlusStencil8,
                native::WGPUTextureAspect_StencilOnly,
                1,
                "depth24plus-stencil8 stencil",
            ),
        ] {
            let texture = s.create_texture(format, 4, 4, 1, 1, usage);
            let bytes = s.copy_to_buffer(texture, aspect, 4, 4, bytes_per_texel);
            assert!(
                bytes.iter().all(|&b| b == 0),
                "non-renderable {name} must read zero: {bytes:?}"
            );
            yawgpu::wgpuTextureRelease(texture);
        }
        s.assert_no_errors();
        s.release();

        // depth32float-stencil8 (optional feature): both aspects are copyable,
        // so the depth-from-combined and stencil-from-combined paths are covered.
        let s = Session::with_features(select, &[native::WGPUFeatureName_Depth32FloatStencil8]);
        if s.device.is_null() {
            eprintln!("skipping depth32float-stencil8: feature not advertised");
            s.release();
            return;
        }
        let texture = s.create_texture(
            native::WGPUTextureFormat_Depth32FloatStencil8,
            4,
            4,
            1,
            1,
            usage,
        );
        let depth = s.copy_to_buffer(texture, native::WGPUTextureAspect_DepthOnly, 4, 4, 4);
        assert!(
            depth.iter().all(|&b| b == 0),
            "non-renderable depth32float-stencil8 depth must read zero: {depth:?}"
        );
        let stencil = s.copy_to_buffer(texture, native::WGPUTextureAspect_StencilOnly, 4, 4, 1);
        assert!(
            stencil.iter().all(|&b| b == 0),
            "non-renderable depth32float-stencil8 stencil must read zero: {stencil:?}"
        );
        yawgpu::wgpuTextureRelease(texture);
        s.assert_no_errors();
        s.release();
    }
}

pub fn lazy_init_multisampled_load_attachment_resolves_to_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let msaa = s.create_texture(
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
            4,
            1,
            4,
            native::WGPUTextureUsage_RenderAttachment,
        );
        let resolve = s.create_texture(
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst,
        );
        // Pre-fill the resolve target so a skipped resolve would be visible.
        s.write_canary(resolve, 0, 4, 4, 4);
        let msaa_view = yawgpu::wgpuTextureCreateView(msaa, std::ptr::null());
        let resolve_view = yawgpu::wgpuTextureCreateView(resolve, std::ptr::null());
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        record_render_pass(
            encoder,
            msaa_view,
            resolve_view,
            native::WGPULoadOp_Load,
            native::WGPUStoreOp_Store,
        );
        s.submit(encoder);
        let bytes = s.copy_to_buffer(resolve, native::WGPUTextureAspect_All, 4, 4, 4);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "resolving an uninitialized MSAA attachment must give zero: {bytes:?}"
        );
        s.assert_no_errors();
        yawgpu::wgpuTextureViewRelease(msaa_view);
        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureRelease(msaa);
        yawgpu::wgpuTextureRelease(resolve);
        s.release();
    }
}

pub fn lazy_init_compressed_mip_copies_zero_bytes_and_canary_mip_survives(
    backend: RealBackend,
    select: u32,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::with_features(select, &[native::WGPUFeatureName_TextureCompressionBC]);
        if s.device.is_null() {
            eprintln!("skipping: texture-compression-bc not advertised");
            return;
        }
        // bc1: 8 bytes per 4x4 block. 8x8 with 2 mips: mip 0 = 2x2 blocks, mip 1 = 1 block.
        let texture = s.create_texture(
            native::WGPUTextureFormat_BC1RGBAUnorm,
            8,
            8,
            2,
            1,
            native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
        );
        // Canary into mip 0 (4 blocks x 8 bytes = 32 bytes, 16 bytes per block row).
        let canary = vec![CANARY; 32];
        s.write_texture_bytes(texture, 0, &canary, 16, 2, 8, 8);
        let mip1 = s.copy_blocks_to_buffer(texture, 1, 1, 1, 8);
        assert!(
            mip1.iter().all(|&b| b == 0),
            "uninitialized bc1 mip must be zero bytes: {mip1:?}"
        );
        let mip0 = s.copy_blocks_to_buffer(texture, 0, 2, 2, 8);
        assert!(
            mip0.iter().all(|&b| b == CANARY),
            "bc1 canary clobbered: {mip0:?}"
        );
        s.assert_no_errors();
        yawgpu::wgpuTextureRelease(texture);
        s.release();
    }
}

pub fn lazy_init_store_op_discard_makes_the_next_read_zero(backend: RealBackend, select: u32) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let texture = s.create_texture(
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_CopySrc,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        // Pass 1: clear to red and store. Pass 2: load, then discard.
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(s.device, std::ptr::null());
        record_render_pass(
            encoder,
            view,
            std::ptr::null(),
            native::WGPULoadOp_Clear,
            native::WGPUStoreOp_Store,
        );
        record_render_pass(
            encoder,
            view,
            std::ptr::null(),
            native::WGPULoadOp_Load,
            native::WGPUStoreOp_Discard,
        );
        s.submit(encoder);
        let texels = s.load_mip_as_floats(texture, 0, 4, 4);
        assert!(
            texels.iter().all(|&v| v == 0.0),
            "after storeOp Discard the texture must read zero: {texels:?}"
        );
        let bytes = s.copy_to_buffer(texture, native::WGPUTextureAspect_All, 4, 4, 4);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "copy after discard must be zero: {bytes:?}"
        );
        s.assert_no_errors();
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        s.release();
    }
}

pub fn lazy_init_partial_write_then_sampled_read_keeps_written_texels(
    backend: RealBackend,
    select: u32,
) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let s = Session::new(select);
        let texture = s.create_texture(
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
            4,
            1,
            1,
            native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_CopyDst
                | native::WGPUTextureUsage_CopySrc,
        );
        // Write only the top-left 2x2 texels.
        let data = vec![CANARY; 2 * 2 * 4];
        s.write_texture_bytes(texture, 0, &data, 8, 2, 2, 2);
        let texels = s.load_mip_as_floats(texture, 0, 4, 4);
        let canary = f32::from(CANARY) / 255.0;
        for y in 0..4 {
            for x in 0..4 {
                let v = texels[(y * 4 + x) * 4];
                if x < 2 && y < 2 {
                    assert!(
                        (v - canary).abs() < 0.01,
                        "written texel ({x},{y}) lost: {v}"
                    );
                } else {
                    assert_eq!(v, 0.0, "unwritten texel ({x},{y}) must be zero");
                }
            }
        }
        s.assert_no_errors();
        yawgpu::wgpuTextureRelease(texture);
        s.release();
    }
}

// --- session / helpers -------------------------------------------------------

struct Session {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl Session {
    unsafe fn new(select: u32) -> Self {
        Self::with_features(select, &[])
    }

    unsafe fn with_features(select: u32, features: &[native::WGPUFeatureName]) -> Self {
        let instance = create_instance(select);
        let adapter = request_adapter(instance);
        if features
            .iter()
            .any(|&f| yawgpu::wgpuAdapterHasFeature(adapter, f) == 0)
        {
            return Self {
                instance,
                adapter,
                device: std::ptr::null(),
                queue: std::ptr::null(),
                errors: Arc::new(Mutex::new(Vec::new())),
            };
        }
        let device = request_device(instance, adapter, features);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        Self {
            instance,
            adapter,
            device,
            queue,
            errors,
        }
    }

    unsafe fn create_texture(
        &self,
        format: native::WGPUTextureFormat,
        width: u32,
        height: u32,
        mip_level_count: u32,
        sample_count: u32,
        usage: native::WGPUTextureUsage,
    ) -> native::WGPUTexture {
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width,
                height,
                depthOrArrayLayers: 1,
            },
            format,
            mipLevelCount: mip_level_count,
            sampleCount: sample_count,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(self.device, &descriptor);
        assert!(!texture.is_null());
        texture
    }

    /// Fills the whole mip with the canary byte (uncompressed, `bpp` bytes/texel).
    unsafe fn write_canary(
        &self,
        texture: native::WGPUTexture,
        mip: u32,
        w: u32,
        h: u32,
        bpp: u32,
    ) {
        let data = vec![CANARY; (w * h * bpp) as usize];
        self.write_texture_bytes(texture, mip, &data, w * bpp, h, w, h);
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn write_texture_bytes(
        &self,
        texture: native::WGPUTexture,
        mip: u32,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
        width: u32,
        height: u32,
    ) {
        let destination = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: mip,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: bytes_per_row,
            rowsPerImage: rows_per_image,
        };
        let extent = native::WGPUExtent3D {
            width,
            height,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            self.queue,
            &destination,
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &extent,
        );
    }

    /// Copies one uncompressed mip (`w`×`h`, `bpp` bytes/texel) to a buffer and returns the
    /// tightly packed bytes.
    unsafe fn copy_to_buffer(
        &self,
        texture: native::WGPUTexture,
        aspect: native::WGPUTextureAspect,
        w: u32,
        h: u32,
        bpp: u32,
    ) -> Vec<u8> {
        self.copy_region_to_buffer(texture, 0, aspect, w, h, w * bpp, h)
    }

    /// Copies `blocks_w`×`blocks_h` blocks of a compressed mip (`block_bytes` per block).
    unsafe fn copy_blocks_to_buffer(
        &self,
        texture: native::WGPUTexture,
        mip: u32,
        blocks_w: u32,
        blocks_h: u32,
        block_bytes: u32,
    ) -> Vec<u8> {
        self.copy_region_to_buffer(
            texture,
            mip,
            native::WGPUTextureAspect_All,
            blocks_w * 4,
            blocks_h * 4,
            blocks_w * block_bytes,
            blocks_h,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_region_to_buffer(
        &self,
        texture: native::WGPUTexture,
        mip: u32,
        aspect: native::WGPUTextureAspect,
        width: u32,
        height: u32,
        row_bytes: u32,
        rows: u32,
    ) -> Vec<u8> {
        let padded = 256;
        let size = u64::from(padded) * u64::from(rows);
        let readback = create_buffer(
            self.device,
            size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(self.device, std::ptr::null());
        let source = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: mip,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect,
        };
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: padded,
                rowsPerImage: rows,
            },
            buffer: readback,
        };
        let extent = native::WGPUExtent3D {
            width,
            height,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
        self.submit(encoder);
        let mapped = read_buffer(self.instance, readback, size as usize);
        let mut bytes = Vec::with_capacity((row_bytes * rows) as usize);
        for row in 0..rows as usize {
            let start = row * padded as usize;
            bytes.extend_from_slice(&mapped[start..start + row_bytes as usize]);
        }
        yawgpu::wgpuBufferRelease(readback);
        bytes
    }

    /// Runs `LOAD_SHADER` over one mip through a `texture_2d<f32>` binding and
    /// returns the texels as floats (4 per texel).
    unsafe fn load_mip_as_floats(
        &self,
        texture: native::WGPUTexture,
        mip: u32,
        w: u32,
        h: u32,
    ) -> Vec<f32> {
        let view_descriptor = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_Undefined,
            dimension: native::WGPUTextureViewDimension_2D,
            baseMipLevel: mip,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_None,
        };
        let view = yawgpu::wgpuTextureCreateView(texture, &view_descriptor);
        assert!(!view.is_null());
        let out_size = u64::from(w * h) * 16;
        let out = create_buffer(
            self.device,
            out_size,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            self.device,
            out_size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let mut entries: [native::WGPUBindGroupLayoutEntry; 2] = std::mem::zeroed();
        entries[0].binding = 0;
        entries[0].visibility = native::WGPUShaderStage_Compute;
        entries[0].texture.sampleType = native::WGPUTextureSampleType_Float;
        entries[0].texture.viewDimension = native::WGPUTextureViewDimension_2D;
        entries[1].binding = 1;
        entries[1].visibility = native::WGPUShaderStage_Compute;
        entries[1].buffer.type_ = native::WGPUBufferBindingType_Storage;
        let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(
            self.device,
            &native::WGPUBindGroupLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                entryCount: 2,
                entries: entries.as_ptr(),
            },
        );
        let layout = yawgpu::wgpuDeviceCreatePipelineLayout(
            self.device,
            &native::WGPUPipelineLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                bindGroupLayoutCount: 1,
                bindGroupLayouts: &bgl,
                immediateSize: 0,
            },
        );
        let module = create_wgsl_module(self.device, LOAD_SHADER);
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            self.device,
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout,
                compute: native::WGPUComputeState {
                    nextInChain: std::ptr::null_mut(),
                    module,
                    entryPoint: string_view("main"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                },
            },
        );
        assert!(!pipeline.is_null());
        let bind_entries = [
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 0,
                buffer: std::ptr::null(),
                offset: 0,
                size: 0,
                sampler: std::ptr::null(),
                textureView: view,
            },
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 1,
                buffer: out,
                offset: 0,
                size: out_size,
                sampler: std::ptr::null(),
                textureView: std::ptr::null(),
            },
        ];
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            self.device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: bgl,
                entryCount: 2,
                entries: bind_entries.as_ptr(),
            },
        );
        assert!(!bind_group.is_null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(self.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, w, h, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, out, 0, readback, 0, out_size);
        self.submit(encoder);
        let bytes = read_buffer(self.instance, readback, out_size as usize);
        let floats = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(out);
        yawgpu::wgpuTextureViewRelease(view);
        floats
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }

    fn assert_no_errors(&self) {
        let errors = self.errors.lock().expect("error lock");
        assert!(errors.is_empty(), "unexpected device errors: {errors:?}");
    }

    unsafe fn release(self) {
        if !self.device.is_null() {
            yawgpu::wgpuQueueRelease(self.queue);
            yawgpu::wgpuDeviceRelease(self.device);
        }
        yawgpu::wgpuAdapterRelease(self.adapter);
        yawgpu::wgpuInstanceRelease(self.instance);
    }
}

unsafe fn record_render_pass(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
    resolve_target: native::WGPUTextureView,
    load_op: native::WGPULoadOp,
    store_op: native::WGPUStoreOp,
) {
    let attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: resolve_target,
        loadOp: load_op,
        storeOp: store_op,
        clearValue: native::WGPUColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
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
    features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let mut device: native::WGPUDevice = std::ptr::null();
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

fn string_view(text: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: text.as_ptr().cast(),
        length: text.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
