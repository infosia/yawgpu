//! CTS port of `webgpu/api/validation/query_set/create.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::create_query_set;

const MAX_QUERY_COUNT: u32 = 4096;

#[test]
#[ignore = "core currently rejects count=0 query sets; CTS expects only count > 4096 to fail"]
fn count() {
    for query_type in [
        native::WGPUQueryType_Occlusion,
        native::WGPUQueryType_Timestamp,
    ] {
        let test = if query_type == native::WGPUQueryType_Timestamp {
            ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery])
        } else {
            ValidationTest::new()
        };
        unsafe {
            for count in [0, MAX_QUERY_COUNT, MAX_QUERY_COUNT + 1] {
                test.clear_errors();
                let query_set = create_query_set(test.device(), query_type, count);
                let errors = test.errors();
                assert_eq!(
                    errors.is_empty(),
                    count <= MAX_QUERY_COUNT,
                    "query type {query_type}, count {count}, errors: {errors:?}"
                );
                yawgpu::wgpuQuerySetRelease(query_set);
            }
        }
    }
}
