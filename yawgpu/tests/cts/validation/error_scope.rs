//! CTS port of `webgpu/api/validation/error_scope.spec.ts`.

use yawgpu::native;
use yawgpu_core::ErrorKind;
use yawgpu_test::{wait, ValidationTest};

use crate::common::{pop_and_wait, pop_error_scope, PopState};

const FILTERS: &[(native::WGPUErrorFilter, ErrorKind, native::WGPUErrorType)] = &[
    (
        native::WGPUErrorFilter_Validation,
        ErrorKind::Validation,
        native::WGPUErrorType_Validation,
    ),
    (
        native::WGPUErrorFilter_OutOfMemory,
        ErrorKind::OutOfMemory,
        native::WGPUErrorType_OutOfMemory,
    ),
    (
        native::WGPUErrorFilter_Internal,
        ErrorKind::Internal,
        native::WGPUErrorType_Internal,
    ),
];

#[test]
fn simple() {
    for &(error_filter, error_kind, expected_type) in FILTERS {
        for &(scope_filter, _, _) in FILTERS {
            let test = ValidationTest::new();
            unsafe {
                yawgpu::wgpuDevicePushErrorScope(test.device(), scope_filter);
                yawgpu::testing_dispatch_device_error(test.device(), error_kind, "generated");

                let mut state = PopState::default();
                let call = pop_and_wait(&test, &mut state);
                assert_eq!(call.status, native::WGPUPopErrorScopeStatus_Success);
                if error_filter == scope_filter {
                    assert_eq!(call.error_type, expected_type);
                    assert!(test.errors().is_empty());
                } else {
                    assert_eq!(call.error_type, native::WGPUErrorType_NoError);
                    assert_eq!(test.errors().len(), 1);
                }
            }
        }
    }
}

#[test]
fn empty() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        let call = pop_and_wait(&test, &mut state);
        assert_eq!(call.status, native::WGPUPopErrorScopeStatus_Error);
        assert_eq!(call.error_type, native::WGPUErrorType_NoError);
    }
}

#[test]
fn parent_scope() {
    for &(filter, kind, expected_type) in FILTERS {
        for stack_depth in [1_usize, 10, 100] {
            let test = ValidationTest::new();
            unsafe {
                yawgpu::wgpuDevicePushErrorScope(test.device(), filter);
                let unmatched = FILTERS
                    .iter()
                    .map(|(scope_filter, _, _)| *scope_filter)
                    .filter(|scope_filter| *scope_filter != filter)
                    .collect::<Vec<_>>();
                for index in 0..stack_depth {
                    yawgpu::wgpuDevicePushErrorScope(
                        test.device(),
                        unmatched[index % unmatched.len()],
                    );
                }

                yawgpu::testing_dispatch_device_error(test.device(), kind, "parent");

                let mut state = PopState::default();
                for _ in 0..stack_depth {
                    let call = pop_and_wait(&test, &mut state);
                    assert_eq!(call.error_type, native::WGPUErrorType_NoError);
                }
                let call = pop_and_wait(&test, &mut state);
                assert_eq!(call.error_type, expected_type);
                assert!(test.errors().is_empty());
            }
        }
    }
}

#[test]
fn current_scope() {
    for &(filter, kind, expected_type) in FILTERS {
        for stack_depth in [1_usize, 10, 100] {
            let test = ValidationTest::new();
            unsafe {
                for index in 0..stack_depth {
                    yawgpu::wgpuDevicePushErrorScope(
                        test.device(),
                        FILTERS[index % FILTERS.len()].0,
                    );
                }
                yawgpu::wgpuDevicePushErrorScope(test.device(), filter);
                yawgpu::testing_dispatch_device_error(test.device(), kind, "current");

                let mut state = PopState::default();
                let call = pop_and_wait(&test, &mut state);
                assert_eq!(call.error_type, expected_type);
                for _ in 0..stack_depth {
                    let call = pop_and_wait(&test, &mut state);
                    assert_eq!(call.error_type, native::WGPUErrorType_NoError);
                }
                assert!(test.errors().is_empty());
            }
        }
    }
}

#[test]
fn balanced_siblings() {
    for &(filter, _, _) in FILTERS {
        for count in [1_usize, 10, 100] {
            let test = ValidationTest::new();
            unsafe {
                let mut states = (0..count).map(|_| PopState::default()).collect::<Vec<_>>();
                for state in &mut states {
                    yawgpu::wgpuDevicePushErrorScope(test.device(), filter);
                    let future = pop_error_scope(test.device(), state);
                    wait(test.instance(), future);
                }

                let mut empty = PopState::default();
                let empty_call = pop_and_wait(&test, &mut empty);
                assert_eq!(empty_call.status, native::WGPUPopErrorScopeStatus_Error);
                for state in &states {
                    assert_eq!(state.calls[0].error_type, native::WGPUErrorType_NoError);
                }
            }
        }
    }
}

#[test]
fn balanced_nesting() {
    for &(filter, _, _) in FILTERS {
        for count in [1_usize, 10, 100] {
            let test = ValidationTest::new();
            unsafe {
                for _ in 0..count {
                    yawgpu::wgpuDevicePushErrorScope(test.device(), filter);
                }
                let mut state = PopState::default();
                for _ in 0..count {
                    let call = pop_and_wait(&test, &mut state);
                    assert_eq!(call.error_type, native::WGPUErrorType_NoError);
                }
                let call = pop_and_wait(&test, &mut state);
                assert_eq!(call.status, native::WGPUPopErrorScopeStatus_Error);
            }
        }
    }
}
