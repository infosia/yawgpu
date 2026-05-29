use std::ffi::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyMethod {
    WriteTexture,
    CopyB2T,
    CopyT2B,
}

pub const ALL_METHODS: &[CopyMethod] = &[
    CopyMethod::WriteTexture,
    CopyMethod::CopyB2T,
    CopyMethod::CopyT2B,
];

pub const BUFFER_TEXTURE_METHODS: &[CopyMethod] = &[CopyMethod::CopyB2T, CopyMethod::CopyT2B];

#[derive(Clone, Copy)]
pub struct CopyParams {
    pub texture: native::WGPUTexture,
    pub buffer: native::WGPUBuffer,
    pub layout: native::WGPUTexelCopyBufferLayout,
    pub origin: native::WGPUOrigin3D,
    pub mip_level: u32,
    pub aspect: native::WGPUTextureAspect,
    pub copy_size: native::WGPUExtent3D,
    pub data_size: u64,
    pub submit: bool,
}

impl CopyParams {
    pub fn new(
        texture: native::WGPUTexture,
        buffer: native::WGPUBuffer,
        copy_size: native::WGPUExtent3D,
        data_size: u64,
    ) -> Self {
        Self {
            texture,
            buffer,
            layout: layout_undefined(0),
            origin: origin(0, 0, 0),
            mip_level: 0,
            aspect: native::WGPUTextureAspect_Undefined,
            copy_size,
            data_size,
            submit: false,
        }
    }
}

pub unsafe fn expect_copy(
    test: &ValidationTest,
    method: CopyMethod,
    params: CopyParams,
    success: bool,
) {
    if success {
        test.expect_no_validation_error(|| run_copy(test, method, params));
    } else {
        assert_device_error!({
            run_copy(test, method, params);
        });
    }
}

pub unsafe fn run_copy(test: &ValidationTest, method: CopyMethod, params: CopyParams) {
    match method {
        CopyMethod::WriteTexture => {
            let queue = yawgpu::wgpuDeviceGetQueue(test.device());
            let destination = texture_info(
                params.texture,
                params.mip_level,
                params.origin,
                params.aspect,
            );
            let data_len = usize::try_from(params.data_size).expect("test data size fits usize");
            let data = vec![0_u8; data_len.min(4096)];
            let ptr = if data_len == 0 {
                std::ptr::null()
            } else {
                data.as_ptr().cast()
            };
            yawgpu::wgpuQueueWriteTexture(
                queue,
                &destination,
                ptr,
                data_len,
                &params.layout,
                &params.copy_size,
            );
            yawgpu::wgpuQueueRelease(queue);
        }
        CopyMethod::CopyB2T => {
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
            let source = buffer_info(params.buffer, params.layout);
            let destination = texture_info(
                params.texture,
                params.mip_level,
                params.origin,
                params.aspect,
            );
            yawgpu::wgpuCommandEncoderCopyBufferToTexture(
                encoder,
                &source,
                &destination,
                &params.copy_size,
            );
            finish_or_submit(test, encoder, params.submit);
        }
        CopyMethod::CopyT2B => {
            let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
            let source = texture_info(
                params.texture,
                params.mip_level,
                params.origin,
                params.aspect,
            );
            let destination = buffer_info(params.buffer, params.layout);
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
                encoder,
                &source,
                &destination,
                &params.copy_size,
            );
            finish_or_submit(test, encoder, params.submit);
        }
    }
}

unsafe fn finish_or_submit(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    submit: bool,
) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    if submit {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
    yawgpu::wgpuCommandBufferRelease(command_buffer);
}

pub unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: false.into(),
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_error_buffer(
    test: &ValidationTest,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let mut buffer = std::ptr::null();
    assert_device_error!({
        buffer = yawgpu::wgpuDeviceCreateBuffer(
            test.device(),
            &native::WGPUBufferDescriptor {
                size: u64::MAX,
                usage,
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                mappedAtCreation: false.into(),
            },
        );
    });
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

pub unsafe fn create_error_texture(test: &ValidationTest) -> native::WGPUTexture {
    let mut texture = std::ptr::null();
    assert_device_error!({
        texture = yawgpu::wgpuDeviceCreateTexture(
            test.device(),
            &native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_None,
                ..texture_descriptor(
                    native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureDimension_2D,
                    extent(4, 4, 1),
                    1,
                    1,
                )
            },
        );
    });
    assert!(!texture.is_null());
    texture
}

pub fn texture_descriptor(
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    dimension: native::WGPUTextureDimension,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension,
        size,
        format,
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

pub fn texture_info(
    texture: native::WGPUTexture,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip_level,
        origin,
        aspect,
    }
}

pub fn buffer_info(
    buffer: native::WGPUBuffer,
    layout: native::WGPUTexelCopyBufferLayout,
) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo { buffer, layout }
}

pub fn layout(
    offset: u64,
    bytes_per_row: u32,
    rows_per_image: u32,
) -> native::WGPUTexelCopyBufferLayout {
    native::WGPUTexelCopyBufferLayout {
        offset,
        bytesPerRow: bytes_per_row,
        rowsPerImage: rows_per_image,
    }
}

pub fn layout_undefined(offset: u64) -> native::WGPUTexelCopyBufferLayout {
    native::WGPUTexelCopyBufferLayout {
        offset,
        bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
        rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
    }
}

pub fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}

pub fn origin(x: u32, y: u32, z: u32) -> native::WGPUOrigin3D {
    native::WGPUOrigin3D { x, y, z }
}

pub unsafe fn request_device(
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

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    unsafe {
        *(userdata1 as *mut native::WGPUDevice) = device;
    }
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
