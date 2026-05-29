//! CTS port of `webgpu/api/validation/render_pipeline/multisample_state.spec.ts`.

use yawgpu_test::ValidationTest;

use crate::render_common::{default_multisample, expect_render_pipeline, RenderPipelineCase};

#[test]
fn count() {
    let test = ValidationTest::new();
    unsafe {
        for count in [0, 1, 2, 3, 4, 8, 16, 1024] {
            let mut multisample = default_multisample();
            multisample.count = count;
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    count == 1 || count == 4,
                    RenderPipelineCase {
                        multisample: Some(multisample),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn alpha_to_coverage_count() {
    let test = ValidationTest::new();
    unsafe {
        for alpha_to_coverage in [false, true] {
            for count in [1, 4] {
                let mut multisample = default_multisample();
                multisample.count = count;
                multisample.alphaToCoverageEnabled = u32::from(alpha_to_coverage);
                let success = if alpha_to_coverage {
                    count == 4
                } else {
                    count == 1 || count == 4
                };
                for is_async in [false, true] {
                    expect_render_pipeline(
                        &test,
                        is_async,
                        success,
                        RenderPipelineCase {
                            multisample: Some(multisample),
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn alpha_to_coverage_sample_mask() {
    let test = ValidationTest::new();
    unsafe {
        let fragment_with_sample_mask = "
struct Output {
  @builtin(sample_mask) mask_out: u32,
  @location(0) color: vec4f,
}
@fragment fn main() -> Output {
  var o: Output;
  o.mask_out = 0xffffffffu;
  o.color = vec4f();
  return o;
}";
        for alpha_to_coverage in [false, true] {
            for has_sample_mask in [false, true] {
                let mut multisample = default_multisample();
                multisample.count = 4;
                multisample.alphaToCoverageEnabled = u32::from(alpha_to_coverage);
                let success = !has_sample_mask || !alpha_to_coverage;
                for is_async in [false, true] {
                    expect_render_pipeline(
                        &test,
                        is_async,
                        success,
                        RenderPipelineCase {
                            fragment_source: if has_sample_mask {
                                Some(fragment_with_sample_mask)
                            } else {
                                Some(crate::render_common::FRAGMENT_COLOR)
                            },
                            multisample: Some(multisample),
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }
}
