//! CTS port of `webgpu/api/validation/render_pipeline/misc.spec.ts`.

use yawgpu_test::ValidationTest;

use crate::common::{create_pipeline_layout, request_device};
use crate::render_common::{expect_render_pipeline, RenderPipelineCase};

#[test]
fn basic() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            expect_render_pipeline(&test, is_async, true, RenderPipelineCase::default());
        }
    }
}

#[test]
fn no_attachment() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_source: None,
                    fragment_has_target: false,
                    depth_stencil: false,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_state_only() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    fragment_source: None,
                    fragment_has_target: false,
                    depth_stencil: true,
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_source: None,
                    fragment_has_target: false,
                    depth_stencil: false,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn pipeline_layout_device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let own_layout = create_pipeline_layout(test.device(), &[], 0);
        let other_layout = create_pipeline_layout(other, &[], 0);
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    layout: Some(own_layout),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    layout: Some(other_layout),
                    ..Default::default()
                },
            );
        }
        yawgpu::wgpuPipelineLayoutRelease(other_layout);
        yawgpu::wgpuPipelineLayoutRelease(own_layout);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn external_texture() {
    // N/A: Web GPUExternalTexture is a browser object with no webgpu.h C analogue.
}

#[test]
fn storage_texture_format() {
    let test = ValidationTest::new();
    unsafe {
        let valid = "
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }
@fragment fn fs() -> @location(0) vec4f {
  textureStore(tex, vec2i(0), vec4f());
  return vec4f();
}";
        // `r8unorm` is a write-only storage format only with `texture-formats-tier1`,
        // which this default device does not enable — so its auto-layout storage
        // binding is rejected (`rgba8sint`, by contrast, is baseline-storage and
        // valid, per F-059).
        let invalid = "
@group(0) @binding(0) var tex: texture_storage_2d<r8unorm, write>;
@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }
@fragment fn fs() -> @location(0) vec4f {
  textureStore(tex, vec2i(0), vec4f());
  return vec4f();
}";
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    vertex_source: valid,
                    fragment_source: Some(valid),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    vertex_source: invalid,
                    fragment_source: Some(invalid),
                    ..Default::default()
                },
            );
        }
    }
}
