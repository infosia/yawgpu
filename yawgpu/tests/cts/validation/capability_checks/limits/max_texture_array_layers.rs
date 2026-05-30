//! CTS port of `webgpu/api/validation/capability_checks/limits/maxTextureArrayLayers.spec.ts`.

use crate::common;

#[test]
fn create_texture_at_over() {
    unsafe { common::assert_max_texture_array_layers_create_texture_at_over() };
}
