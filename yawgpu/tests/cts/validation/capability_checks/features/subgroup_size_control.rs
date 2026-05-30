use yawgpu::native;

use crate::feature_common;

#[test]
#[ignore = "Noop does not advertise subgroups/subgroup-size-control"]
fn enables_subgroups() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_Subgroups);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_Subgroups);
    feature_common::assert_device_has_feature(&test, native::WGPUFeatureName_Subgroups);
}
