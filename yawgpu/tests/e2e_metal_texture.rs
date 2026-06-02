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
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_PIXELS_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const BUFFER_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;
#[cfg(feature = "metal")]
const TEXEL_PATTERN: [u8; 16] = [1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43];
#[cfg(feature = "metal")]
const F16_ONE: [u8; 2] = [0x00, 0x3c];
#[cfg(feature = "metal")]
const F32_ONE: [u8; 4] = [0x00, 0x00, 0x80, 0x3f];
#[cfg(feature = "metal")]
const RG11B10_ONE: [u8; 4] = rg11b10_ufloat_one().to_le_bytes();
#[cfg(feature = "metal")]
const RGB9E5_ONE: [u8; 4] = rgb9e5_ufloat_one().to_le_bytes();

#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug)]
enum FormatExpectation {
    ByteExact,
    NonZero,
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug)]
struct FormatCase {
    format: native::WGPUTextureFormat,
    bytes_per_pixel: usize,
    source: &'static [u8],
    expectation: FormatExpectation,
}

#[cfg(feature = "metal")]
const ADDED_UNCOMPRESSED_COLOR_FORMATS: &[FormatCase] = &[
    format_case_exact(native::WGPUTextureFormat_R8Snorm, 1),
    format_case_exact(native::WGPUTextureFormat_R8Uint, 1),
    format_case_exact(native::WGPUTextureFormat_R8Sint, 1),
    format_case_exact(native::WGPUTextureFormat_RG8Unorm, 2),
    format_case_exact(native::WGPUTextureFormat_RG8Snorm, 2),
    format_case_exact(native::WGPUTextureFormat_RG8Uint, 2),
    format_case_exact(native::WGPUTextureFormat_RG8Sint, 2),
    format_case_exact(native::WGPUTextureFormat_RGBA8UnormSrgb, 4),
    format_case_exact(native::WGPUTextureFormat_RGBA8Snorm, 4),
    format_case_exact(native::WGPUTextureFormat_RGBA8Sint, 4),
    format_case_exact(native::WGPUTextureFormat_BGRA8UnormSrgb, 4),
    format_case_exact(native::WGPUTextureFormat_R16Unorm, 2),
    format_case_exact(native::WGPUTextureFormat_R16Snorm, 2),
    format_case_exact(native::WGPUTextureFormat_R16Uint, 2),
    format_case_exact(native::WGPUTextureFormat_R16Sint, 2),
    format_case_non_zero(native::WGPUTextureFormat_R16Float, 2, &F16_ONE),
    format_case_exact(native::WGPUTextureFormat_RG16Unorm, 4),
    format_case_exact(native::WGPUTextureFormat_RG16Snorm, 4),
    format_case_exact(native::WGPUTextureFormat_RG16Uint, 4),
    format_case_exact(native::WGPUTextureFormat_RG16Sint, 4),
    format_case_non_zero(
        native::WGPUTextureFormat_RG16Float,
        4,
        &[0x00, 0x3c, 0x00, 0x3c],
    ),
    format_case_exact(native::WGPUTextureFormat_RGBA16Unorm, 8),
    format_case_exact(native::WGPUTextureFormat_RGBA16Snorm, 8),
    format_case_exact(native::WGPUTextureFormat_RGBA16Uint, 8),
    format_case_exact(native::WGPUTextureFormat_RGBA16Sint, 8),
    format_case_exact(native::WGPUTextureFormat_R32Uint, 4),
    format_case_exact(native::WGPUTextureFormat_R32Sint, 4),
    format_case_non_zero(native::WGPUTextureFormat_R32Float, 4, &F32_ONE),
    format_case_exact(native::WGPUTextureFormat_RG32Uint, 8),
    format_case_exact(native::WGPUTextureFormat_RG32Sint, 8),
    format_case_non_zero(
        native::WGPUTextureFormat_RG32Float,
        8,
        &[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f],
    ),
    format_case_exact(native::WGPUTextureFormat_RGBA32Uint, 16),
    format_case_exact(native::WGPUTextureFormat_RGBA32Sint, 16),
    format_case_non_zero(
        native::WGPUTextureFormat_RGBA32Float,
        16,
        &[
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00,
            0x80, 0x3f,
        ],
    ),
    format_case_exact(native::WGPUTextureFormat_RGB10A2Uint, 4),
    format_case_exact(native::WGPUTextureFormat_RGB10A2Unorm, 4),
    format_case_non_zero(native::WGPUTextureFormat_RG11B10Ufloat, 4, &RG11B10_ONE),
    format_case_non_zero(native::WGPUTextureFormat_RGB9E5Ufloat, 4, &RGB9E5_ONE),
];

#[cfg(feature = "metal")]
const fn format_case_exact(
    format: native::WGPUTextureFormat,
    bytes_per_pixel: usize,
) -> FormatCase {
    FormatCase {
        format,
        bytes_per_pixel,
        source: &TEXEL_PATTERN,
        expectation: FormatExpectation::ByteExact,
    }
}

#[cfg(feature = "metal")]
const fn format_case_non_zero(
    format: native::WGPUTextureFormat,
    bytes_per_pixel: usize,
    source: &'static [u8],
) -> FormatCase {
    FormatCase {
        format,
        bytes_per_pixel,
        source,
        expectation: FormatExpectation::NonZero,
    }
}

#[cfg(feature = "metal")]
const fn rg11b10_ufloat_one() -> u32 {
    let r = 15_u32 << 6;
    let g = 15_u32 << 6;
    let b = 15_u32 << 5;
    r | (g << 11) | (b << 22)
}

#[cfg(feature = "metal")]
const fn rgb9e5_ufloat_one() -> u32 {
    let mantissa = 256_u32;
    let exponent = 24_u32;
    mantissa | (mantissa << 9) | (mantissa << 18) | (exponent << 27)
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_buffer_texture_buffer_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let destination = run_buffer_texture_buffer_submit(device, &pixels);
        let actual = read_unpacked_texture_buffer(instance, destination);

        assert_eq!(actual, pixels);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_texture_texture_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let destination = run_texture_texture_submit(device, &pixels);
        let actual = read_unpacked_texture_buffer(instance, destination);

        assert_eq!(actual, pixels);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_added_uncompressed_color_texture_copy_round_trips_data() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let required_features = [
            native::WGPUFeatureName_TextureFormatsTier1,
            native::WGPUFeatureName_RG11B10UfloatRenderable,
        ];
        let device = request_device_with_features(instance, adapter, &required_features);
        let errors = install_error_capture(device);

        for case in ADDED_UNCOMPRESSED_COLOR_FORMATS {
            let texel = &case.source[..case.bytes_per_pixel];

            let b2t2b = run_format_buffer_texture_buffer_submit(device, *case, texel);
            assert_format_readback(instance, b2t2b, texel, *case, "B2T/T2B");
            yawgpu::wgpuBufferRelease(b2t2b);

            let b2t2t2b = run_format_texture_texture_submit(device, *case, texel);
            assert_format_readback(instance, b2t2t2b, texel, *case, "B2T/T2T/T2B");
            yawgpu::wgpuBufferRelease(b2t2t2b);
        }

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_queue_write_texture_uploads_color_data_round_trips() {
    // Regression for CTS F-025: wgpuQueueWriteTexture must actually upload the
    // supplied bytes (it used to ignore the data pointer and write zeros). Upload
    // via writeTexture, read back via copyTextureToBuffer, and require an exact
    // round-trip.
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let destination = run_write_texture_buffer_submit(device, &pixels);
        let actual = read_unpacked_texture_buffer(instance, destination);

        assert_eq!(actual, pixels);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampler_creation_has_no_device_error() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let sampler = create_sampler(device);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_depth24_plus_stencil8_texture_creation_has_no_device_error() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_TextureBinding,
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
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn default_noop_texture_and_sampler_path_has_no_device_error() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let b2t_destination = run_buffer_texture_buffer_submit(device, &pixels);
        let t2t_destination = run_texture_texture_submit(device, &pixels);
        let sampler = create_sampler(device);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuBufferRelease(t2t_destination);
        yawgpu::wgpuBufferRelease(b2t_destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_buffer_texture_buffer_submit(
    device: native::WGPUDevice,
    pixels: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_padded_pixels(queue, source, pixels);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_b2t(encoder, source, texture);
    record_t2b(encoder, texture, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

#[cfg(feature = "metal")]
unsafe fn run_write_texture_buffer_submit(
    device: native::WGPUDevice,
    pixels: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );

    // Lay the pixels out at bytesPerRow = 256 (the upload stride) and upload
    // straight to the texture with wgpuQueueWriteTexture.
    let mut padded = vec![0u8; BUFFER_SIZE];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_PIXELS_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        padded[padded_offset..padded_offset + ROW_PIXELS_BYTES]
            .copy_from_slice(&pixels[pixel_offset..pixel_offset + ROW_PIXELS_BYTES]);
    }
    let dest_info = texture_copy_info(texture);
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: BYTES_PER_ROW,
        rowsPerImage: HEIGHT,
    };
    let size = texture_extent();
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &dest_info,
        padded.as_ptr().cast(),
        padded.len(),
        &layout,
        &size,
    );

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_t2b(encoder, texture, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

unsafe fn run_texture_texture_submit(
    device: native::WGPUDevice,
    pixels: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture_a = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    let texture_b = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_padded_pixels(queue, source, pixels);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_b2t(encoder, source, texture_a);
    record_t2t(encoder, texture_a, texture_b);
    record_t2b(encoder, texture_b, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture_b);
    yawgpu::wgpuTextureRelease(texture_a);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

#[cfg(feature = "metal")]
unsafe fn run_format_buffer_texture_buffer_submit(
    device: native::WGPUDevice,
    case: FormatCase,
    texel: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture = create_format_texture(
        device,
        case.format,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_aligned_texel(queue, source, texel);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_format_b2t(encoder, source, texture);
    record_format_t2b(encoder, texture, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

#[cfg(feature = "metal")]
unsafe fn run_format_texture_texture_submit(
    device: native::WGPUDevice,
    case: FormatCase,
    texel: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture_a = create_format_texture(
        device,
        case.format,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    let texture_b = create_format_texture(
        device,
        case.format,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_aligned_texel(queue, source, texel);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_format_b2t(encoder, source, texture_a);
    record_format_t2t(encoder, texture_a, texture_b);
    record_format_t2b(encoder, texture_b, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture_b);
    yawgpu::wgpuTextureRelease(texture_a);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

unsafe fn record_b2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
) {
    let source = buffer_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &size);
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = texture_copy_info(source);
    let destination = buffer_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

unsafe fn record_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let source = texture_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
unsafe fn record_format_b2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
) {
    let source = format_buffer_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = format_extent();
    yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
unsafe fn record_format_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = texture_copy_info(source);
    let destination = format_buffer_copy_info(destination);
    let size = format_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
unsafe fn record_format_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let source = texture_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = format_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &size);
}

unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn write_padded_pixels(queue: native::WGPUQueue, buffer: native::WGPUBuffer, pixels: &[u8]) {
    let mut padded = vec![0; BUFFER_SIZE];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_PIXELS_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        padded[padded_offset..padded_offset + ROW_PIXELS_BYTES]
            .copy_from_slice(&pixels[pixel_offset..pixel_offset + ROW_PIXELS_BYTES]);
    }
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, padded.as_ptr().cast(), padded.len());
}

#[cfg(feature = "metal")]
unsafe fn write_aligned_texel(queue: native::WGPUQueue, buffer: native::WGPUBuffer, texel: &[u8]) {
    let len = texel.len().next_multiple_of(4);
    let mut aligned = [0_u8; 16];
    aligned[..texel.len()].copy_from_slice(texel);
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, aligned.as_ptr().cast(), len);
}

#[cfg(feature = "metal")]
unsafe fn assert_format_readback(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    texel: &[u8],
    case: FormatCase,
    label: &str,
) {
    // mapAsync requires a 4-byte-aligned size, so map a padded range and
    // compare only the texel's bytes (1- and 2-byte formats would otherwise
    // fail the map with a validation error).
    let mapped = read_buffer(instance, buffer, 0, texel.len().next_multiple_of(4));
    let actual = &mapped[..texel.len()];
    match case.expectation {
        FormatExpectation::ByteExact => assert_eq!(actual, texel, "{label} {:?}", case.format),
        FormatExpectation::NonZero => assert!(
            actual.iter().any(|byte| *byte != 0),
            "{label} {:?} read back all zero",
            case.format
        ),
    }
}

#[cfg(feature = "metal")]
unsafe fn read_unpacked_texture_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
) -> Vec<u8> {
    let mapped = read_buffer(instance, buffer, 0, BUFFER_SIZE);
    let mut pixels = vec![0; ROW_PIXELS_BYTES * HEIGHT as usize];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_PIXELS_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        pixels[pixel_offset..pixel_offset + ROW_PIXELS_BYTES]
            .copy_from_slice(&mapped[padded_offset..padded_offset + ROW_PIXELS_BYTES]);
    }
    pixels
}

#[cfg(feature = "metal")]
unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: u64,
    len: usize,
) -> Vec<u8> {
    let mapped_len = len.next_multiple_of(4);
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
        mapped_len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);

    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("test offset fits in usize"),
        mapped_len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), mapped_len)[..len].to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size: BUFFER_SIZE as u64,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
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

#[cfg(feature = "metal")]
unsafe fn create_format_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: format_extent(),
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_sampler(device: native::WGPUDevice) -> native::WGPUSampler {
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

#[cfg(feature = "metal")]
fn format_buffer_copy_info(buffer: native::WGPUBuffer) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: 1,
        },
        buffer,
    }
}

fn buffer_copy_info(buffer: native::WGPUBuffer) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer,
    }
}

fn texture_copy_info(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

#[cfg(feature = "metal")]
fn format_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    }
}

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

fn source_pixels() -> Vec<u8> {
    (0..ROW_PIXELS_BYTES * HEIGHT as usize)
        .map(|value| u8::try_from((value * 17 + 3) % 251).expect("test byte fits in u8"))
        .collect()
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
    request_device_with_features(instance, adapter, &[])
}

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

#[cfg(feature = "metal")]
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
