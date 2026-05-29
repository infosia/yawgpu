//! Ports `$CTS/src/webgpu/api/validation/buffer/create.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, cartesian2, cartesian3, ValidationTest};

const BUFFER_SIZE_ALIGNMENT: u64 = 4;
const SOME_BOGUS_BUFFER_USAGE: native::WGPUBufferUsage = 0x4000_0000;
const MAX_SAFE_MULTIPLE_OF_8: u64 = 9_007_199_254_740_984;
const ALL_BUFFER_USAGE_BITS: native::WGPUBufferUsage = native::WGPUBufferUsage_MapRead
    | native::WGPUBufferUsage_MapWrite
    | native::WGPUBufferUsage_CopySrc
    | native::WGPUBufferUsage_CopyDst
    | native::WGPUBufferUsage_Index
    | native::WGPUBufferUsage_Vertex
    | native::WGPUBufferUsage_Uniform
    | native::WGPUBufferUsage_Storage
    | native::WGPUBufferUsage_Indirect
    | native::WGPUBufferUsage_QueryResolve;
const BUFFER_USAGES: &[native::WGPUBufferUsage] = &[
    native::WGPUBufferUsage_MapRead,
    native::WGPUBufferUsage_MapWrite,
    native::WGPUBufferUsage_CopySrc,
    native::WGPUBufferUsage_CopyDst,
    native::WGPUBufferUsage_Index,
    native::WGPUBufferUsage_Vertex,
    native::WGPUBufferUsage_Uniform,
    native::WGPUBufferUsage_Storage,
    native::WGPUBufferUsage_Indirect,
    native::WGPUBufferUsage_QueryResolve,
];

#[test]
fn size() {
    let test = ValidationTest::new();
    let sizes = [
        0,
        BUFFER_SIZE_ALIGNMENT / 2,
        BUFFER_SIZE_ALIGNMENT,
        BUFFER_SIZE_ALIGNMENT + BUFFER_SIZE_ALIGNMENT / 2,
        BUFFER_SIZE_ALIGNMENT * 2,
    ];

    for (mapped_at_creation, size) in cartesian2(&[false, true], &sizes) {
        let valid = !mapped_at_creation || size % BUFFER_SIZE_ALIGNMENT == 0;
        let mut buffer = std::ptr::null();
        if valid {
            test.expect_no_validation_error(|| {
                buffer = create_buffer(
                    test.device(),
                    size,
                    native::WGPUBufferUsage_CopySrc,
                    mapped_at_creation,
                );
            });
        } else {
            assert_device_error!({
                buffer = create_buffer(
                    test.device(),
                    size,
                    native::WGPUBufferUsage_CopySrc,
                    mapped_at_creation,
                );
            });
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn limit() {
    let test = ValidationTest::new();
    let mut limits = unsafe { std::mem::zeroed::<native::WGPULimits>() };
    assert_eq!(
        unsafe { yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits) },
        native::WGPUStatus_Success
    );

    for size_addition in [-1_i64, 0, 1] {
        let size = limits.maxBufferSize.saturating_add_signed(size_addition);
        let valid = size <= limits.maxBufferSize;
        let mut buffer = std::ptr::null();
        if valid {
            test.expect_no_validation_error(|| {
                buffer = create_buffer(test.device(), size, native::WGPUBufferUsage_CopySrc, false);
            });
        } else {
            assert_device_error!({
                buffer = create_buffer(test.device(), size, native::WGPUBufferUsage_CopySrc, false);
            });
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn usage() {
    let test = ValidationTest::new();
    let usage_values = usage_values_with_zero_and_bogus();

    for (usage1, usage2, mapped_at_creation) in
        cartesian3(&usage_values, &usage_values, &[false, true])
    {
        if usage1 > usage2 {
            continue;
        }

        let usage = usage1 | usage2;
        let valid = usage != 0
            && (usage & !ALL_BUFFER_USAGE_BITS) == 0
            && ((usage & native::WGPUBufferUsage_MapRead) == 0
                || (usage & !(native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead))
                    == 0)
            && ((usage & native::WGPUBufferUsage_MapWrite) == 0
                || (usage & !(native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_MapWrite))
                    == 0);

        let mut buffer = std::ptr::null();
        if valid {
            test.expect_no_validation_error(|| {
                buffer = create_buffer(
                    test.device(),
                    BUFFER_SIZE_ALIGNMENT * 2,
                    usage,
                    mapped_at_creation,
                );
            });
        } else {
            assert_device_error!({
                buffer = create_buffer(
                    test.device(),
                    BUFFER_SIZE_ALIGNMENT * 2,
                    usage,
                    mapped_at_creation,
                );
            });
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn new_usages() {
    let test = ValidationTest::new();
    let exposed_usages = ALL_BUFFER_USAGE_BITS;

    for &usage in BUFFER_USAGES {
        let success = (usage & exposed_usages) == usage;
        let mut buffer = std::ptr::null();
        if success {
            test.expect_no_validation_error(|| {
                buffer = create_buffer(test.device(), 16, usage, false);
            });
        } else {
            assert_device_error!({
                buffer = create_buffer(test.device(), 16, usage, false);
            });
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn create_buffer_invalid_and_oom() {
    let test = ValidationTest::new();
    let cases = [
        (true, native::WGPUBufferUsage_Uniform, 16),
        (true, native::WGPUBufferUsage_Storage, 16),
        (
            false,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_Uniform,
            16,
        ),
        (
            false,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_Uniform,
            MAX_SAFE_MULTIPLE_OF_8,
        ),
        (
            false,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_Uniform,
            0x20_0000_0000,
        ),
        (
            false,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_Uniform,
            16,
        ),
        (
            false,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_Uniform,
            MAX_SAFE_MULTIPLE_OF_8,
        ),
        (
            false,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_Uniform,
            0x20_0000_0000,
        ),
    ];

    for (valid, usage, size) in cases {
        let mut buffer = std::ptr::null();
        if valid {
            test.expect_no_validation_error(|| {
                buffer = create_buffer(test.device(), size, usage, false);
            });
        } else {
            assert_device_error!({
                buffer = create_buffer(test.device(), size, usage, false);
            });
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

fn usage_values_with_zero_and_bogus() -> Vec<native::WGPUBufferUsage> {
    let mut values = Vec::with_capacity(BUFFER_USAGES.len() + 2);
    values.push(0);
    values.extend_from_slice(BUFFER_USAGES);
    values.push(SOME_BOGUS_BUFFER_USAGE);
    values
}

fn create_buffer(
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
    unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) }
}
