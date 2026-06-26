//! Scratch repro for the HANDOFF "yawgpu retains released GPU resources" leak.
//!
//! Drives a create -> use -> release loop on a real Metal device and lets an
//! external `/usr/bin/time -l` observe peak RSS. Attribution is done by running
//! the same test with different `YAWGPU_LEAK_MODE` / `YAWGPU_LEAK_ITERS` and
//! comparing peak RSS deltas across iteration counts.
//!
//! Run:
//!   YAWGPU_LEAK_MODE=cmdsubmit YAWGPU_LEAK_ITERS=20000 \
//!   /usr/bin/time -l cargo test -p yawgpu --features metal \
//!     --test e2e_metal_leak metal_leak_loop -- --ignored --exact --nocapture
//!
//! Modes: buffer | texture | cmdsubmit | map | write | maponly | mapfail |
//!        writetexture | sampler | view | renderpass | bindgroup | drawpass | all
//!
//! NOT a regression test — manual diagnostic only.

#![cfg(any(feature = "metal", feature = "vulkan"))]

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend leak repro"]
fn metal_leak_loop() {
    let backend = std::env::var("YAWGPU_LEAK_BACKEND").unwrap_or_else(|_| "metal".to_owned());
    let (backend_id, real_backend) = match backend.as_str() {
        "vulkan" => (YAWGPU_INSTANCE_BACKEND_VULKAN, RealBackend::Vulkan),
        "metal" => (YAWGPU_INSTANCE_BACKEND_METAL, RealBackend::Metal),
        other => panic!("unknown YAWGPU_LEAK_BACKEND: {other}"),
    };
    if real_backend_skip_reason(real_backend).is_some() {
        eprintln!("metal_leak_loop: skipped (no real {backend} device)");
        return;
    }

    let mode = std::env::var("YAWGPU_LEAK_MODE").unwrap_or_else(|_| "all".to_owned());
    let iters: u64 = std::env::var("YAWGPU_LEAK_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    eprintln!("metal_leak_loop: mode={mode} iters={iters}");

    unsafe {
        let instance = create_instance(backend_id);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);

        for i in 0..iters {
            match mode.as_str() {
                "buffer" => iter_buffer(device),
                "texture" => iter_texture(device),
                "cmdsubmit" => iter_cmdsubmit(device),
                "map" => iter_map(instance, device),
                "write" => iter_write(device),
                "maponly" => iter_maponly(instance, device),
                "mapfail" => iter_mapfail(instance, device),
                "writetexture" => iter_writetexture(device),
                "sampler" => iter_sampler(device),
                "view" => iter_view(device),
                "renderpass" => iter_renderpass(device),
                "bindgroup" => iter_bindgroup(device),
                "drawpass" => iter_drawpass(instance, device),
                "distinctshader" => iter_distinctshader(device, i),
                "distinctpipeline" => iter_distinctpipeline(device, i),
                "all" => {
                    iter_buffer(device);
                    iter_texture(device);
                    iter_cmdsubmit(device);
                    iter_map(instance, device);
                }
                other => panic!("unknown YAWGPU_LEAK_MODE: {other}"),
            }
            if i % 5000 == 0 {
                eprintln!("  rss after {i}: {} KB", current_rss_kb());
            }
        }
        eprintln!("metal_leak_loop: final rss {} KB", current_rss_kb());

        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Create + release a small buffer (no submit, no map).
unsafe fn iter_buffer(device: native::WGPUDevice) {
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    yawgpu::wgpuBufferRelease(buffer);
}

/// Create + release a small 8x8 rgba8unorm texture (no submit).
unsafe fn iter_texture(device: native::WGPUDevice) {
    let texture = create_texture(device);
    yawgpu::wgpuTextureRelease(texture);
}

/// Create encoder, record a clearBuffer, finish, submit, release everything.
/// Exercises the CommandBuffer referenced-resource / command_ops retention.
unsafe fn iter_cmdsubmit(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 256);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Full write -> map -> readback -> unmap -> release. Exercises the future
/// registry / pending-callback path on every iteration.
unsafe fn iter_map(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let data = [7u8; 256];
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), data.len());

    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, 256);
    assert!(!ptr.is_null());
    yawgpu::wgpuBufferUnmap(buffer);
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Only `wgpuQueueWriteBuffer` (staging + submit), no map.
unsafe fn iter_write(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let data = [7u8; 256];
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), data.len());
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Only mapAsync + wait + getMappedRange + unmap (no writeBuffer).
unsafe fn iter_maponly(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let size: u64 = std::env::var("YAWGPU_LEAK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let buffer = create_buffer(
        device,
        size,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, 256);
    assert!(!ptr.is_null());
    yawgpu::wgpuBufferUnmap(buffer);
    yawgpu::wgpuBufferRelease(buffer);
}

/// mapAsync on a buffer WITHOUT MapRead usage: validation fails, so the
/// pending callback retains no buffer Arc, but a future is still registered.
/// Isolates the FutureRegistry registration machinery from the data path.
unsafe fn iter_mapfail(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let mut status = native::WGPUMapAsyncStatus_Success;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    yawgpu::wgpuBufferRelease(buffer);
}

/// Create texture + `wgpuQueueWriteTexture` (multi-mip upload) + release.
/// Prime suspect for the CTS texture-workload ~26 KB/subcase growth.
unsafe fn iter_writetexture(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let mips: u32 = std::env::var("YAWGPU_LEAK_MIPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let base: u32 = 64;
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: base,
            height: base,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: mips,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    for mip in 0..mips {
        let dim = (base >> mip).max(1);
        let bytes_per_row = dim * 4;
        let data = vec![7u8; (bytes_per_row * dim) as usize];
        let dest = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: mip,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: bytes_per_row,
            rowsPerImage: dim,
        };
        let size = native::WGPUExtent3D {
            width: dim,
            height: dim,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &dest,
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &size,
        );
    }
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuQueueRelease(queue);
}

/// createSampler + release.
unsafe fn iter_sampler(device: native::WGPUDevice) {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_ClampToEdge,
        addressModeV: native::WGPUAddressMode_ClampToEdge,
        addressModeW: native::WGPUAddressMode_ClampToEdge,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    };
    let sampler = yawgpu::wgpuDeviceCreateSampler(device, &descriptor);
    assert!(!sampler.is_null());
    yawgpu::wgpuSamplerRelease(sampler);
}

/// create texture + createView + release both.
unsafe fn iter_view(device: native::WGPUDevice) {
    let texture = create_texture(device);
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
}

/// renderable texture + view + render pass (clear, no pipeline) + submit + release.
unsafe fn iter_renderpass(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 64,
            height: 64,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    let color = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
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
    let attachments = [color];
    let pass_descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuQueueRelease(queue);
}

/// WGSL mirroring a CTS `textureSampleLevel` subcase: a full-screen triangle
/// from `@builtin(vertex_index)` whose fragment shader samples `t`/`s` at LOD 0
/// and tints by the uniform `u`. Bindings: 0 = sampled texture, 1 = sampler,
/// 2 = uniform vec4. The pipeline's auto bind-group layout (group 0) therefore
/// has exactly these three entries.
const SAMPLE_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> u: vec4<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VertexOut {
    var out: VertexOut;
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, in.uv, 0.0) * u;
}
"#;

/// Cached objects shared across iterations of a sampling-style mode, mirroring
/// how CTS caches the pipeline / bind-group-layout / sampler across subcases so
/// the per-iteration cost is only the bind group + its per-iter resources.
struct SampleCache {
    device: native::WGPUDevice,
    /// Kept alive so the pipeline's shader stays valid; not read after build.
    _module: native::WGPUShaderModule,
    pipeline: native::WGPURenderPipeline,
    bind_group_layout: native::WGPUBindGroupLayout,
    sampler: native::WGPUSampler,
    uniform: native::WGPUBuffer,
    /// Backing texture for `view`, cached so the view stays valid; not read after build.
    _texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

thread_local! {
    static SAMPLE_CACHE: std::cell::RefCell<Option<SampleCache>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` with the per-device sampling cache, building it on first use (and
/// rebuilding if the device pointer changed). The cached objects are never
/// released inside the leak loop, matching CTS's cross-subcase caching.
unsafe fn with_sample_cache<R>(device: native::WGPUDevice, f: impl FnOnce(&SampleCache) -> R) -> R {
    SAMPLE_CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let needs_build = match slot.as_ref() {
            Some(cache) => cache.device != device,
            None => true,
        };
        if needs_build {
            *slot = Some(build_sample_cache(device));
        }
        f(slot.as_ref().expect("sample cache built"))
    })
}

/// Build the cached pipeline / BGL / sampler / uniform / sampled texture+view.
unsafe fn build_sample_cache(device: native::WGPUDevice) -> SampleCache {
    let module = create_wgsl_module(device, SAMPLE_SHADER);
    let pipeline = create_sample_pipeline(device, module);
    let bind_group_layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    assert!(!bind_group_layout.is_null());
    let sampler = create_sample_sampler(device);
    let uniform = create_buffer(
        device,
        sample_uniform_size(),
        native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
    );
    let texture = create_sampled_texture(device);
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    SampleCache {
        device,
        _module: module,
        pipeline,
        bind_group_layout,
        sampler,
        uniform,
        _texture: texture,
        view,
    }
}

/// Render pipeline for `SAMPLE_SHADER`: full-screen triangle, no vertex buffers,
/// single rgba8unorm color target, auto pipeline layout.
/// Create a UNIQUE shader module per iter (busting the device shader cache) +
/// release it. Tests whether `WGPUDeviceImpl::shader_module_cache` evicts on
/// handle release.
unsafe fn iter_distinctshader(device: native::WGPUDevice, i: u64) {
    let source = format!("{SAMPLE_SHADER}\n// unique-iter-{i}\n");
    let module = create_wgsl_module(device, &source);
    yawgpu::wgpuShaderModuleRelease(module);
}

/// Create a UNIQUE shader module + render pipeline per iter (busting both device
/// caches) + release both. Models a CTS subcase that compiles a distinct
/// pipeline; tests whether the device pipeline/shader caches evict on release.
unsafe fn iter_distinctpipeline(device: native::WGPUDevice, i: u64) {
    let source = format!("{SAMPLE_SHADER}\n// unique-iter-{i}\n");
    let module = create_wgsl_module(device, &source);
    let pipeline = create_sample_pipeline(device, module);
    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
}

unsafe fn create_sample_pipeline(
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
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

/// Linear sampler matching a typical CTS sampling subcase.
unsafe fn create_sample_sampler(device: native::WGPUDevice) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_ClampToEdge,
        addressModeV: native::WGPUAddressMode_ClampToEdge,
        addressModeW: native::WGPUAddressMode_ClampToEdge,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    };
    let sampler = yawgpu::wgpuDeviceCreateSampler(device, &descriptor);
    assert!(!sampler.is_null());
    sampler
}

/// 8x8 rgba8unorm sampled texture (TextureBinding | CopyDst).
unsafe fn create_sampled_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 8,
            height: 8,
            depthOrArrayLayers: 1,
        },
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

/// Create a texture+sampler+uniform bind group from the cached BGL.
unsafe fn create_sample_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    view: native::WGPUTextureView,
    sampler: native::WGPUSampler,
    uniform: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entries = [
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
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler,
            textureView: std::ptr::null(),
        },
        native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 2,
            buffer: uniform,
            offset: 0,
            size: sample_uniform_size(),
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        },
    ];
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    bind_group
}

/// Isolate bind-group create/release: pipeline / BGL / sampler / uniform /
/// sampled texture+view are cached once; only a fresh bind group referencing
/// them is created and released per iteration.
unsafe fn iter_bindgroup(device: native::WGPUDevice) {
    with_sample_cache(device, |cache| {
        let bind_group = create_sample_bind_group(
            device,
            cache.bind_group_layout,
            cache.view,
            cache.sampler,
            cache.uniform,
        );
        yawgpu::wgpuBindGroupRelease(bind_group);
    });
}

/// Mirror a full CTS `textureSampleLevel` subcase: cached shader / pipeline /
/// BGL / sampler; per iteration create+upload a sampled texture, build a bind
/// group, render the sampled result into an attachment, copy it to a readback
/// buffer, map+read it back, then release every per-iter object.
unsafe fn iter_drawpass(instance: native::WGPUInstance, device: native::WGPUDevice) {
    with_sample_cache(device, |cache| {
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // Per-iter sampled texture (rgba8unorm, TextureBinding | CopyDst) + upload.
        let sampled = create_sampled_texture(device);
        let dim: u32 = 8;
        let bytes_per_row = dim * 4;
        let data = vec![7u8; (bytes_per_row * dim) as usize];
        let dest = native::WGPUTexelCopyTextureInfo {
            texture: sampled,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: bytes_per_row,
            rowsPerImage: dim,
        };
        let upload_size = native::WGPUExtent3D {
            width: dim,
            height: dim,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &dest,
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &upload_size,
        );
        let sampled_view = yawgpu::wgpuTextureCreateView(sampled, std::ptr::null());
        assert!(!sampled_view.is_null());

        // Per-iter bind group (sampled view + cached sampler + small uniform).
        let uniform = create_buffer(
            device,
            sample_uniform_size(),
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_CopyDst,
        );
        let uniform_data = [1.0f32, 1.0, 1.0, 1.0];
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            uniform,
            0,
            uniform_data.as_ptr().cast(),
            std::mem::size_of_val(&uniform_data),
        );
        let bind_group = create_sample_bind_group(
            device,
            cache.bind_group_layout,
            sampled_view,
            cache.sampler,
            uniform,
        );

        // Per-iter render-attachment texture (rgba8unorm, RenderAttachment | CopySrc).
        let attach_dim: u32 = 16;
        let attachment = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: attach_dim,
                height: attach_dim,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let attachment_texture = yawgpu::wgpuDeviceCreateTexture(device, &attachment);
        assert!(!attachment_texture.is_null());
        let attachment_view = yawgpu::wgpuTextureCreateView(attachment_texture, std::ptr::null());
        assert!(!attachment_view.is_null());

        // Per-iter readback buffer (CopyDst | MapRead). 256-byte aligned row.
        let bytes_per_row_readback: u32 = 256;
        let readback_size = (bytes_per_row_readback * attach_dim) as u64;
        let readback = create_buffer(
            device,
            readback_size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        // Encode: render pass (clear/store) -> setPipeline/setBindGroup/draw -> end.
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let color = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: attachment_view,
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
        let attachments = [color];
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, cache.pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // copyTextureToBuffer(attachment -> readback).
        let copy_source = native::WGPUTexelCopyTextureInfo {
            texture: attachment_texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let copy_dest = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: bytes_per_row_readback,
                rowsPerImage: attach_dim,
            },
            buffer: readback,
        };
        let copy_size = native::WGPUExtent3D {
            width: attach_dim,
            height: attach_dim,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &copy_source,
            &copy_dest,
            &copy_size,
        );

        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

        // Map the readback, wait, read, unmap.
        let mut status = native::WGPUMapAsyncStatus_Error;
        let callback_info = native::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(map_callback),
            userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let readback_len = usize::try_from(readback_size).expect("readback size fits in usize");
        let future = yawgpu::wgpuBufferMapAsync(
            readback,
            native::WGPUMapMode_Read,
            0,
            readback_len,
            callback_info,
        );
        wait(instance, future);
        let ptr = yawgpu::wgpuBufferGetConstMappedRange(readback, 0, readback_len);
        assert!(!ptr.is_null());
        yawgpu::wgpuBufferUnmap(readback);

        // Release every per-iter object (cached objects stay alive).
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(attachment_view);
        yawgpu::wgpuTextureRelease(attachment_texture);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBufferRelease(uniform);
        yawgpu::wgpuTextureViewRelease(sampled_view);
        yawgpu::wgpuTextureRelease(sampled);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuQueueRelease(queue);
    });
}

/// Uniform buffer size for `SAMPLE_SHADER`'s `vec4<f32>`.
fn sample_uniform_size() -> u64 {
    4 * std::mem::size_of::<f32>() as u64
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

unsafe fn create_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 8,
            height: 8,
            depthOrArrayLayers: 1,
        },
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

unsafe fn create_instance(backend_id: u32) -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: backend_id,
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

/// Current resident set size in KB via the platform RSS query
/// (`mach_task_self`/`task_info` on macOS, `/proc/self/statm` on Linux).
#[cfg(target_os = "macos")]
fn current_rss_kb() -> u64 {
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    // MACH_TASK_BASIC_INFO_COUNT = sizeof(info)/sizeof(natural_t=u32)
    const COUNT: u32 = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
    const MACH_TASK_BASIC_INFO: i32 = 20;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    unsafe {
        let mut info = std::mem::zeroed::<MachTaskBasicInfo>();
        let mut count = COUNT;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            (&mut info as *mut MachTaskBasicInfo).cast(),
            &mut count,
        );
        if kr != 0 {
            return 0;
        }
        info.resident_size / 1024
    }
}

/// Current resident set size in KB from `/proc/self/statm` (field 2 = resident
/// pages). Returns 0 on any read/parse failure. Page size is the standard 4096.
#[cfg(target_os = "linux")]
fn current_rss_kb() -> u64 {
    const PAGE_SIZE: u64 = 4096;
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    match statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse::<u64>().ok())
    {
        Some(pages) => pages * PAGE_SIZE / 1024,
        None => 0,
    }
}

/// Fallback for platforms without an RSS query wired up.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_rss_kb() -> u64 {
    0
}
