//! CTS port of `webgpu/api/validation/capability_checks/limits/maxTextureDimension2D.spec.ts`.

use crate::common;

#[test]
fn create_texture_at_over() {
    unsafe { common::assert_max_texture_dimension_2d_create_texture_at_over() };
}

#[test]
#[ignore = "native-surface configure limit subcase is excluded from active Noop core validation"]
fn configure_at_over() {
    unsafe { common::assert_max_texture_dimension_2d_create_texture_at_over() };
}

#[test]
#[ignore = "native-surface getCurrentTexture limit subcase is excluded from active Noop core validation"]
fn get_current_texture_at_over() {
    unsafe { common::assert_max_texture_dimension_2d_create_texture_at_over() };
}
