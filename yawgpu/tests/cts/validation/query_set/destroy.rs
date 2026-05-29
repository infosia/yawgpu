//! CTS port of `webgpu/api/validation/query_set/destroy.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{create_query_set, pop_and_wait, PopState};

#[test]
fn twice() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
        test.expect_no_validation_error(|| {
            yawgpu::wgpuQuerySetDestroy(query_set);
            yawgpu::wgpuQuerySetDestroy(query_set);
        });
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn invalid_queryset() {
    let test = ValidationTest::new();
    unsafe {
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 4097);
        let mut state = PopState::default();
        let call = pop_and_wait(&test, &mut state);
        assert_eq!(call.error_type, native::WGPUErrorType_Validation);

        test.expect_no_validation_error(|| {
            yawgpu::wgpuQuerySetDestroy(query_set);
        });
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}
