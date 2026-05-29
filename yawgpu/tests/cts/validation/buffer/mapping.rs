//! Ports `$CTS/src/webgpu/api/validation/buffer/mapping.spec.ts`.
//!
//! CTS `gc_behavior,*` -- N/A: JS GC / ArrayBuffer detachment, no C analogue.
//! CTS `earlyRejection` checks JS microtask ordering. The C FFI future model
//! has no microtask analogue, so these tests assert only the observable device
//! error and map callback status.

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, cartesian2, cartesian3, ValidationTest};

const OFFSET_ALIGNMENT: usize = 8;
const SIZE_ALIGNMENT: usize = 4;
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
const MAP_MODES: &[native::WGPUMapMode] = &[native::WGPUMapMode_Read, native::WGPUMapMode_Write];

#[derive(Default)]
struct MapState {
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[derive(Clone, Copy)]
struct RangeCase {
    buffer_size: u64,
    map_offset: Option<usize>,
    map_size: Option<usize>,
    offset: Option<usize>,
    size: Option<usize>,
}

#[test]
fn map_async_usage() {
    let test = ValidationTest::new();
    let cases = [
        (
            native::WGPUMapMode_Read,
            Some(native::WGPUBufferUsage_MapRead),
        ),
        (
            native::WGPUMapMode_Write,
            Some(native::WGPUBufferUsage_MapWrite),
        ),
        (native::WGPUMapMode_None, None),
    ];

    for ((map_mode, valid_usage), usage) in cartesian2(&cases, BUFFER_USAGES) {
        let buffer = create_buffer(test.device(), 16, usage, false);
        if Some(usage) == valid_usage {
            assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        } else {
            assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_invalid_buffer() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = error_buffer(&test);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_state_destroyed() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        let mut pending = MapState::default();
        let future = map_async(
            buffer,
            map_mode,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut pending,
        );

        unsafe { yawgpu::wgpuBufferDestroy(buffer) };
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
        assert_eq!(pending.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_state_mapped_at_creation() {
    let test = ValidationTest::new();
    let cases = [
        (native::WGPUMapMode_Read, native::WGPUBufferUsage_MapRead),
        (native::WGPUMapMode_Write, native::WGPUBufferUsage_MapWrite),
    ];

    for (map_mode, valid_usage) in cases {
        let buffer = create_buffer(test.device(), 16, valid_usage, true);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_state_mapped() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_state_mapping_pending() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        let mut pending0 = MapState::default();
        let future0 = map_async(
            buffer,
            map_mode,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut pending0,
        );

        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu_test::wait_for_future(test.instance(), future0) };
        assert_eq!(pending0.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_size_unspecified_oob() {
    let test = ValidationTest::new();
    let cases = [
        (0, 0),
        (0, 1),
        (0, OFFSET_ALIGNMENT),
        (16, 0),
        (16, OFFSET_ALIGNMENT),
        (16, 16),
        (16, 17),
        (16, 16 + OFFSET_ALIGNMENT),
    ];

    for (map_mode, (buffer_size, offset)) in cartesian2(MAP_MODES, &cases) {
        let buffer = create_mappable_buffer(&test, map_mode, buffer_size);
        if offset <= buffer_size as usize && offset % OFFSET_ALIGNMENT == 0 {
            assert_map_success(&test, buffer, map_mode, offset, native::WGPU_WHOLE_MAP_SIZE);
        } else {
            assert_map_validation_error(
                &test,
                buffer,
                map_mode,
                offset,
                native::WGPU_WHOLE_MAP_SIZE,
            );
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_offset_and_size_alignment() {
    let test = ValidationTest::new();

    for (map_mode, offset, size) in cartesian3(
        MAP_MODES,
        &[0, OFFSET_ALIGNMENT, OFFSET_ALIGNMENT / 2],
        &[0, SIZE_ALIGNMENT, SIZE_ALIGNMENT / 2],
    ) {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        if offset % OFFSET_ALIGNMENT == 0 && size % SIZE_ALIGNMENT == 0 {
            assert_map_success(&test, buffer, map_mode, offset, size);
        } else {
            assert_map_validation_error(&test, buffer, map_mode, offset, size);
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_offset_and_size_oob() {
    let test = ValidationTest::new();
    let cases = [
        (0, 0, 0),
        (0, 0, 4),
        (0, 8, 0),
        (16, 0, 16),
        (16, OFFSET_ALIGNMENT, 16),
        (16, 16, 0),
        (16, 16, SIZE_ALIGNMENT),
        (16, 8, 0),
        (16, 8, 8),
        (16, 8, 8 + SIZE_ALIGNMENT),
        (1024, 0, 1024),
        (1024, OFFSET_ALIGNMENT, 1024),
        (1024, 1024, 0),
        (1024, 1024, SIZE_ALIGNMENT),
        (1024, 512, 0),
        (1024, 512, 512),
        (1024, 512, 512 + SIZE_ALIGNMENT),
    ];

    for (map_mode, (buffer_size, offset, size)) in cartesian2(MAP_MODES, &cases) {
        let buffer = create_mappable_buffer(&test, map_mode, buffer_size);
        if offset + size <= buffer_size as usize {
            assert_map_success(&test, buffer, map_mode, offset, size);
        } else {
            assert_map_validation_error(&test, buffer, map_mode, offset, size);
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_early_rejection() {
    let test = ValidationTest::new();

    for (map_mode, offset2) in cartesian2(MAP_MODES, &[0, 8]) {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        let mut pending = MapState::default();
        let future = map_async(
            buffer,
            map_mode,
            0,
            8,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut pending,
        );
        assert_map_validation_error(&test, buffer, map_mode, offset2, 8);
        unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
        assert_eq!(pending.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn map_async_abort_over_invalid_error() {
    let test = ValidationTest::new();

    for (map_mode, unmap_before_resolve) in cartesian2(MAP_MODES, &[true, false]) {
        let buffer = create_buffer(test.device(), 8, native::WGPUBufferUsage_Storage, false);
        let mut state = MapState::default();
        let future = assert_map_async_call_errors(
            &test,
            buffer,
            map_mode,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
            &mut state,
        );
        if unmap_before_resolve {
            unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        }
        unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        if !unmap_before_resolve {
            unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        }
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_state_mapped() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_state_mapped_at_creation() {
    let test = ValidationTest::new();

    for (buffer_usage, map_mode) in cartesian2(BUFFER_USAGES, MAP_MODES) {
        let buffer = create_buffer(test.device(), 16, buffer_usage, true);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_state_invalid_mapped_at_creation() {
    let test = ValidationTest::new();
    let mut buffer = std::ptr::null();

    assert_device_error!({
        buffer = create_buffer(test.device(), 16, 0xffff_ffff, true);
    });
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn get_mapped_range_state_mapped_again() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_state_unmapped() {
    let test = ValidationTest::new();

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
    );
    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, true);
    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn get_mapped_range_subrange_mapped() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
        unsafe { yawgpu::wgpuBufferUnmap(buffer) };

        assert_map_success(&test, buffer, map_mode, 8, native::WGPU_WHOLE_MAP_SIZE);
        assert_mapped_range(buffer, 8, native::WGPU_WHOLE_MAP_SIZE, true);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_subrange_mapped_at_creation() {
    let test = ValidationTest::new();
    let buffer = create_buffer(
        test.device(),
        16,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        true,
    );

    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        8,
        native::WGPU_WHOLE_MAP_SIZE,
    );
    assert_mapped_range(buffer, 8, native::WGPU_WHOLE_MAP_SIZE, true);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn get_mapped_range_state_destroyed() {
    let test = ValidationTest::new();

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
    );
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, true);
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn get_mapped_range_state_mapping_pending() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        let mut pending = MapState::default();
        let future = map_async(
            buffer,
            map_mode,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut pending,
        );
        assert_map_validation_error(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, false);
        unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
        assert_eq!(pending.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        assert_mapped_range(buffer, 0, native::WGPU_WHOLE_MAP_SIZE, true);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_offset_and_size_alignment_mapped() {
    let test = ValidationTest::new();

    for (map_mode, map_offset, offset, size) in cartesian4(
        MAP_MODES,
        &[0, OFFSET_ALIGNMENT],
        &[0, OFFSET_ALIGNMENT, OFFSET_ALIGNMENT / 2],
        &[0, SIZE_ALIGNMENT, SIZE_ALIGNMENT / 2],
    ) {
        let buffer = create_mappable_buffer(&test, map_mode, 32);
        assert_map_success(
            &test,
            buffer,
            map_mode,
            map_offset,
            native::WGPU_WHOLE_MAP_SIZE,
        );
        let success = offset % OFFSET_ALIGNMENT == 0 && size % SIZE_ALIGNMENT == 0;
        assert_mapped_range(buffer, offset + map_offset, size, success);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_offset_and_size_alignment_mapped_at_creation() {
    let test = ValidationTest::new();

    for (offset, size) in cartesian2(
        &[0, OFFSET_ALIGNMENT, OFFSET_ALIGNMENT / 2],
        &[0, SIZE_ALIGNMENT, SIZE_ALIGNMENT / 2],
    ) {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, true);
        let success = offset % OFFSET_ALIGNMENT == 0 && size % SIZE_ALIGNMENT == 0;
        assert_mapped_range(buffer, offset, size, success);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_size_and_offset_oob_mapped_at_creation() {
    let test = ValidationTest::new();
    let cases = mapped_at_creation_oob_cases();

    for case in cases {
        let buffer = create_buffer(
            test.device(),
            case.buffer_size,
            native::WGPUBufferUsage_CopyDst,
            true,
        );
        let actual_offset = case.offset.unwrap_or(0);
        let actual_size = case
            .size
            .unwrap_or_else(|| (case.buffer_size as usize).saturating_sub(actual_offset));
        let success = actual_offset <= case.buffer_size as usize
            && actual_offset
                .checked_add(actual_size)
                .is_some_and(|end| end <= case.buffer_size as usize);
        assert_mapped_range(
            buffer,
            case.offset.unwrap_or(0),
            case.size.unwrap_or(native::WGPU_WHOLE_MAP_SIZE),
            success,
        );
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_size_and_offset_oob_mapped() {
    let test = ValidationTest::new();
    let cases = mapped_oob_cases();

    for (map_mode, case) in cartesian2(MAP_MODES, &cases) {
        let buffer = create_mappable_buffer(&test, map_mode, case.buffer_size);
        assert_map_success(
            &test,
            buffer,
            map_mode,
            case.map_offset.unwrap_or(0),
            case.map_size.unwrap_or(native::WGPU_WHOLE_MAP_SIZE),
        );
        let actual_map_offset = case.map_offset.unwrap_or(0);
        let actual_map_size = case
            .map_size
            .unwrap_or_else(|| (case.buffer_size as usize).saturating_sub(actual_map_offset));
        let actual_offset = case.offset.unwrap_or(0);
        let actual_size = case
            .size
            .unwrap_or_else(|| (case.buffer_size as usize).saturating_sub(actual_offset));
        let success = actual_offset >= actual_map_offset
            && actual_offset <= case.buffer_size as usize
            && actual_offset.checked_add(actual_size).is_some_and(|end| {
                actual_map_offset
                    .checked_add(actual_map_size)
                    .is_some_and(|map_end| end <= map_end)
            });
        assert_mapped_range(
            buffer,
            case.offset.unwrap_or(0),
            case.size.unwrap_or(native::WGPU_WHOLE_MAP_SIZE),
            success,
        );
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_disjoint_ranges() {
    let test = ValidationTest::new();
    let cases = [
        (8, 0, 8, 8),
        (16, 0, 8, 8),
        (8, 8, 8, 0),
        (8, 8, 16, 0),
        (0, 8, 8, 8),
        (16, 8, 8, 8),
        (8, 8, 0, 8),
        (8, 8, 16, 8),
        (16, 20, 24, 0),
        (24, 0, 16, 20),
        (16, 20, 8, 20),
        (16, 20, 32, 20),
        (0, 80, 16, 20),
        (16, 20, 0, 80),
    ];

    for (remap_between_calls, (offset1, size1, offset2, size2)) in
        cartesian2(&[false, true], &cases)
    {
        let buffer = create_buffer(test.device(), 80, native::WGPUBufferUsage_MapRead, false);
        assert_map_success(
            &test,
            buffer,
            native::WGPUMapMode_Read,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
        );
        assert_mapped_range(buffer, offset1, size1, true);

        if remap_between_calls {
            unsafe { yawgpu::wgpuBufferUnmap(buffer) };
            assert_map_success(
                &test,
                buffer,
                native::WGPUMapMode_Read,
                0,
                native::WGPU_WHOLE_MAP_SIZE,
            );
        }

        let disjoint = offset1 >= offset2 + size2 || offset2 >= offset1 + size1;
        assert_mapped_range(buffer, offset2, size2, disjoint || remap_between_calls);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn get_mapped_range_disjoint_ranges_many() {
    let test = ValidationTest::new();
    let stride = 256;
    let num_strides = 256;
    let buffer = create_buffer(
        test.device(),
        (stride * num_strides) as u64,
        native::WGPUBufferUsage_MapRead,
        false,
    );
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
    );

    for index in 0..num_strides {
        assert_mapped_range(buffer, index * stride, 8, true);
    }
    for index in 0..num_strides {
        assert_mapped_range(buffer, index * stride, stride, false);
        assert_mapped_range(buffer, index * stride + 8, stride - 8, true);
    }
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn unmap_state_unmapped() {
    let test = ValidationTest::new();

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
    );
    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, true);
    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn unmap_state_destroyed() {
    let test = ValidationTest::new();

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_mappable_buffer(&test, native::WGPUMapMode_Read, 16);
    assert_map_success(
        &test,
        buffer,
        native::WGPUMapMode_Read,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
    );
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };

    let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, true);
    unsafe { yawgpu::wgpuBufferDestroy(buffer) };
    test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn unmap_state_mapped_at_creation() {
    let test = ValidationTest::new();

    for &buffer_usage in BUFFER_USAGES {
        let buffer = create_buffer(test.device(), 16, buffer_usage, true);
        test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn unmap_state_mapped() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        assert_map_success(&test, buffer, map_mode, 0, native::WGPU_WHOLE_MAP_SIZE);
        test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn unmap_state_mapping_pending() {
    let test = ValidationTest::new();

    for &map_mode in MAP_MODES {
        let buffer = create_mappable_buffer(&test, map_mode, 16);
        let mut state = MapState::default();
        let future = map_async(
            buffer,
            map_mode,
            0,
            native::WGPU_WHOLE_MAP_SIZE,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut state,
        );
        test.expect_no_validation_error(|| unsafe { yawgpu::wgpuBufferUnmap(buffer) });
        unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

fn create_mappable_buffer(
    test: &ValidationTest,
    mode: native::WGPUMapMode,
    size: u64,
) -> native::WGPUBuffer {
    let usage = match mode {
        native::WGPUMapMode_Read => native::WGPUBufferUsage_MapRead,
        native::WGPUMapMode_Write => native::WGPUBufferUsage_MapWrite,
        _ => unreachable!("unexpected map mode"),
    };
    create_buffer(test.device(), size, usage, false)
}

fn error_buffer(test: &ValidationTest) -> native::WGPUBuffer {
    let mut buffer = std::ptr::null();
    assert_device_error!({
        buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_None, false);
    });
    buffer
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

fn assert_map_success(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
) {
    let status = map_and_wait(test, buffer, mode, offset, size);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
}

fn assert_map_validation_error(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
) {
    let mut state = MapState::default();
    let future = assert_map_async_call_errors(test, buffer, mode, offset, size, &mut state);
    unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
    assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
}

fn assert_map_async_call_errors(
    _test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    state: &mut MapState,
) -> native::WGPUFuture {
    let mut future = native::WGPUFuture { id: 0 };
    assert_device_error!({
        future = map_async(
            buffer,
            mode,
            offset,
            size,
            native::WGPUCallbackMode_WaitAnyOnly,
            state,
        );
    });
    future
}

fn map_and_wait(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
) -> native::WGPUMapAsyncStatus {
    let mut state = MapState::default();
    let future = map_async(
        buffer,
        mode,
        offset,
        size,
        native::WGPUCallbackMode_WaitAnyOnly,
        &mut state,
    );
    unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
    assert_eq!(state.statuses.len(), 1);
    state.statuses[0]
}

fn map_async(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_mode: native::WGPUCallbackMode,
    state: &mut MapState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode,
        callback: Some(map_callback),
        userdata1: (state as *mut MapState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuBufferMapAsync(buffer, mode, offset, size, callback_info) }
}

fn assert_mapped_range(buffer: native::WGPUBuffer, offset: usize, size: usize, success: bool) {
    let const_ptr = unsafe { yawgpu::wgpuBufferGetConstMappedRange(buffer, offset, size) };
    if success {
        assert!(
            !const_ptr.is_null(),
            "expected mapped range at offset {offset}, size {size}"
        );
    } else {
        let ptr = unsafe { yawgpu::wgpuBufferGetMappedRange(buffer, offset, size) };
        assert!(
            const_ptr.is_null() && ptr.is_null(),
            "expected null mapped range at offset {offset}, size {size}"
        );
    }
}

fn mapped_at_creation_oob_cases() -> Vec<RangeCase> {
    vec![
        range(0, None, None, None, None),
        range(0, None, None, None, Some(0)),
        range(0, None, None, None, Some(SIZE_ALIGNMENT)),
        range(0, None, None, Some(0), None),
        range(0, None, None, Some(0), Some(0)),
        range(0, None, None, Some(OFFSET_ALIGNMENT), None),
        range(0, None, None, Some(OFFSET_ALIGNMENT), Some(0)),
        range(80, None, None, None, Some(80)),
        range(80, None, None, None, Some(80 + SIZE_ALIGNMENT)),
        range(80, None, None, None, None),
        range(80, None, None, Some(0), None),
        range(80, None, None, Some(OFFSET_ALIGNMENT), None),
        range(80, None, None, Some(80), None),
        range(80, None, None, Some(80 + OFFSET_ALIGNMENT), None),
        range(80, None, None, Some(0), Some(80)),
        range(80, None, None, Some(0), Some(80 + SIZE_ALIGNMENT)),
        range(80, None, None, Some(OFFSET_ALIGNMENT), Some(80)),
        range(80, None, None, Some(40), Some(40)),
        range(80, None, None, Some(40 + OFFSET_ALIGNMENT), Some(40)),
        range(80, None, None, Some(40), Some(40 + SIZE_ALIGNMENT)),
    ]
}

fn mapped_oob_cases() -> Vec<RangeCase> {
    vec![
        range(0, Some(0), None, None, None),
        range(0, Some(0), None, None, Some(0)),
        range(0, Some(0), None, None, Some(SIZE_ALIGNMENT)),
        range(0, Some(0), None, Some(0), None),
        range(0, Some(0), None, Some(0), Some(0)),
        range(0, Some(0), None, Some(OFFSET_ALIGNMENT), None),
        range(0, Some(0), None, Some(OFFSET_ALIGNMENT), Some(0)),
        range(0, Some(0), Some(0), None, None),
        range(0, Some(0), Some(0), Some(0), None),
        range(0, Some(0), Some(0), Some(0), Some(0)),
        range(0, Some(0), Some(0), Some(OFFSET_ALIGNMENT), None),
        range(0, Some(0), Some(0), Some(OFFSET_ALIGNMENT), Some(0)),
        range(80, None, None, Some(0), Some(80)),
        range(80, None, None, Some(0), Some(80 + SIZE_ALIGNMENT)),
        range(80, None, None, Some(OFFSET_ALIGNMENT), Some(80)),
        range(80, Some(24), None, Some(24), Some(80 - 24)),
        range(80, Some(24), None, Some(0), Some(80 - 24 + SIZE_ALIGNMENT)),
        range(80, Some(24), None, Some(OFFSET_ALIGNMENT), Some(80 - 24)),
        range(80, Some(0), Some(80), Some(0), Some(80)),
        range(80, Some(0), Some(80), Some(OFFSET_ALIGNMENT), Some(80)),
        range(80, Some(0), Some(80), Some(0), Some(80 + SIZE_ALIGNMENT)),
        range(80, Some(0), Some(80), Some(40), Some(40)),
        range(80, Some(0), Some(80), Some(40 + OFFSET_ALIGNMENT), Some(40)),
        range(80, Some(0), Some(80), Some(40), Some(40 + SIZE_ALIGNMENT)),
        range(80, Some(24), Some(40), Some(24), Some(40)),
        range(
            80,
            Some(24),
            Some(40),
            Some(24 - OFFSET_ALIGNMENT),
            Some(40),
        ),
        range(
            80,
            Some(24),
            Some(40),
            Some(24 + OFFSET_ALIGNMENT),
            Some(40),
        ),
        range(80, Some(24), Some(40), Some(24), Some(40 + SIZE_ALIGNMENT)),
        range(80, Some(24), Some(40), None, None),
        range(80, Some(24), Some(40), Some(0), None),
        range(80, Some(24), Some(40), Some(24), None),
        range(80, Some(24), None, Some(24), None),
        range(80, Some(24), None, Some(80), None),
        range(80, Some(0), Some(64), None, None),
        range(80, Some(0), Some(64), None, Some(64)),
    ]
}

fn range(
    buffer_size: u64,
    map_offset: Option<usize>,
    map_size: Option<usize>,
    offset: Option<usize>,
    size: Option<usize>,
) -> RangeCase {
    RangeCase {
        buffer_size,
        map_offset,
        map_size,
        offset,
        size,
    }
}

fn cartesian4<A, B, C, D>(a: &[A], b: &[B], c: &[C], d: &[D]) -> Vec<(A, B, C, D)>
where
    A: Clone,
    B: Clone,
    C: Clone,
    D: Clone,
{
    a.iter()
        .flat_map(|item_a| {
            b.iter().flat_map(move |item_b| {
                c.iter().flat_map(move |item_c| {
                    d.iter().map(move |item_d| {
                        (
                            item_a.clone(),
                            item_b.clone(),
                            item_c.clone(),
                            item_d.clone(),
                        )
                    })
                })
            })
        })
        .collect()
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut MapState)).statuses.push(status);
}
