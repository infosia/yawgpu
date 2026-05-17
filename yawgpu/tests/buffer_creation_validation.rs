use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn creation_usage_must_be_non_zero() {
    let test = ValidationTest::new();
    unsafe {
        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_None, false);
        });
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn creation_map_read_usage_restrictions() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_MapRead, false);
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);

        let buffer = create_buffer(
            test.device(),
            4,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
            false,
        );
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);

        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(
                test.device(),
                4,
                native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopySrc,
                false,
            );
        });
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn creation_map_write_usage_restrictions() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_MapWrite, false);
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);

        let buffer = create_buffer(
            test.device(),
            4,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
            false,
        );
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);

        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(
                test.device(),
                4,
                native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopyDst,
                false,
            );
        });
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn creation_size_must_not_exceed_device_limit() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );

        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(
                test.device(),
                limits.maxBufferSize + 1,
                native::WGPUBufferUsage_CopySrc,
                false,
            );
        });
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn mapped_at_creation_size_must_be_four_byte_aligned() {
    let test = ValidationTest::new();
    unsafe {
        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(test.device(), 3, native::WGPUBufferUsage_CopySrc, true);
        });
        assert!(!buffer.is_null());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn mapped_at_creation_starts_mapped_for_non_mappable_usage() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopySrc, true);
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(buffer),
            native::WGPUBufferMapState_Mapped
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn destroy_and_unmap_are_safe_on_error_destroyed_and_unmapped_buffers() {
    let test = ValidationTest::new();
    unsafe {
        let mut error_buffer = std::ptr::null();
        assert_device_error!({
            error_buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_None, false);
        });
        assert!(!error_buffer.is_null());

        test.clear_errors();
        yawgpu::wgpuBufferDestroy(error_buffer);
        yawgpu::wgpuBufferUnmap(error_buffer);
        assert!(test.errors().is_empty());
        yawgpu::wgpuBufferRelease(error_buffer);

        let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopySrc, false);
        test.clear_errors();
        yawgpu::wgpuBufferDestroy(buffer);
        yawgpu::wgpuBufferDestroy(buffer);
        yawgpu::wgpuBufferUnmap(buffer);
        assert!(test.errors().is_empty());
        yawgpu::wgpuBufferRelease(buffer);

        let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopySrc, false);
        test.clear_errors();
        yawgpu::wgpuBufferUnmap(buffer);
        assert!(test.errors().is_empty());
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn error_buffer_reflects_descriptor_size_usage_and_has_queryable_map_state() {
    let test = ValidationTest::new();
    unsafe {
        let usage = native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopySrc;
        let mut buffer = std::ptr::null();
        assert_device_error!({
            buffer = create_buffer(test.device(), 12, usage, false);
        });
        assert!(!buffer.is_null());
        assert_eq!(yawgpu::wgpuBufferGetSize(buffer), 12);
        assert_eq!(yawgpu::wgpuBufferGetUsage(buffer), usage);
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(buffer),
            native::WGPUBufferMapState_Unmapped
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage,
        size,
        mappedAtCreation: u32::from(mapped_at_creation),
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}
