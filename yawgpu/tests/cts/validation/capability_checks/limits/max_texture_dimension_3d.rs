//! CTS port of `webgpu/api/validation/capability_checks/limits/maxTextureDimension3D.spec.ts`.

use crate::common;

#[test]
fn create_texture_at_over() {
    unsafe { common::assert_max_texture_dimension_3d_create_texture_at_over() };
}
