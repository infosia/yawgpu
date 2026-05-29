//! CTS port of `webgpu/api/validation/render_pipeline/vertex_state.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::device_limits;
use crate::render_common::{
    empty_vertex_buffer, expect_render_pipeline, vertex_attribute, vertex_buffer,
    vertex_input_shader, RenderPipelineCase,
};

#[test]
fn max_vertex_buffer_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for count in [0, 1, limits.maxVertexBuffers, limits.maxVertexBuffers + 1] {
            let buffers = vec![empty_vertex_buffer(); count as usize];
            expect_render_pipeline(
                &test,
                false,
                count <= limits.maxVertexBuffers,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn max_vertex_attribute_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for count in [
            0,
            1,
            limits.maxVertexAttributes,
            limits.maxVertexAttributes + 1,
        ] {
            let attributes = (0..count)
                .map(|location| vertex_attribute(native::WGPUVertexFormat_Float32, 0, location))
                .collect::<Vec<_>>();
            let buffers = [vertex_buffer(4, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                count <= limits.maxVertexAttributes,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn max_vertex_buffer_array_stride_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let attribute = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        for stride in [
            0,
            4,
            256,
            u64::from(limits.maxVertexBufferArrayStride),
            u64::from(limits.maxVertexBufferArrayStride) + 4,
        ] {
            let buffers = [vertex_buffer(stride, &attribute)];
            expect_render_pipeline(
                &test,
                false,
                stride <= u64::from(limits.maxVertexBufferArrayStride),
                RenderPipelineCase {
                    vertex_source: &vertex_input_shader(0, "f32"),
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_buffer_array_stride_limit_alignment() {
    let test = ValidationTest::new();
    unsafe {
        let attribute = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        for stride in [0, 1, 2, 4, 8] {
            let buffers = [vertex_buffer(stride, &attribute)];
            expect_render_pipeline(
                &test,
                false,
                stride % 4 == 0,
                RenderPipelineCase {
                    vertex_source: &vertex_input_shader(0, "f32"),
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_attribute_shader_location_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for location in [
            0,
            1,
            limits.maxVertexAttributes - 1,
            limits.maxVertexAttributes,
        ] {
            let attributes = [vertex_attribute(
                native::WGPUVertexFormat_Float32,
                0,
                location,
            )];
            let buffers = [vertex_buffer(4, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                location < limits.maxVertexAttributes,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_attribute_shader_location_unique() {
    let test = ValidationTest::new();
    unsafe {
        let duplicate = [
            vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0),
            vertex_attribute(native::WGPUVertexFormat_Float32, 4, 0),
        ];
        let unique = [
            vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0),
            vertex_attribute(native::WGPUVertexFormat_Float32, 4, 1),
        ];
        for (attributes, success) in [(&unique[..], true), (&duplicate[..], false)] {
            let buffers = [vertex_buffer(8, attributes)];
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_shader_input_location_limit() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for location in [
            0,
            limits.maxVertexAttributes - 1,
            limits.maxVertexAttributes,
        ] {
            let shader = vertex_input_shader(location, "f32");
            let attributes = [vertex_attribute(
                native::WGPUVertexFormat_Float32,
                0,
                location,
            )];
            let buffers = [vertex_buffer(4, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                location < limits.maxVertexAttributes,
                RenderPipelineCase {
                    vertex_source: &shader,
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_shader_input_location_in_vertex_state() {
    let test = ValidationTest::new();
    unsafe {
        let shader = vertex_input_shader(2, "f32");
        let missing = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 0)];
        let present = [vertex_attribute(native::WGPUVertexFormat_Float32, 0, 2)];
        for (attributes, success) in [(&missing[..], false), (&present[..], true)] {
            let buffers = [vertex_buffer(4, attributes)];
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    vertex_source: &shader,
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_shader_type_matches_attribute_format() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (native::WGPUVertexFormat_Float32, "f32", true),
            (native::WGPUVertexFormat_Float32x3, "vec3<f32>", true),
            (native::WGPUVertexFormat_Uint32, "u32", true),
            (native::WGPUVertexFormat_Sint32, "i32", true),
            (native::WGPUVertexFormat_Float32, "i32", false),
            (native::WGPUVertexFormat_Uint32, "f32", false),
        ];
        for (format, ty, success) in cases {
            let shader = vertex_input_shader(0, ty);
            let attributes = [vertex_attribute(format, 0, 0)];
            let buffers = [vertex_buffer(16, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    vertex_source: &shader,
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_attribute_offset_alignment() {
    let test = ValidationTest::new();
    unsafe {
        for (format, offset, success) in [
            (native::WGPUVertexFormat_Float32, 0, true),
            (native::WGPUVertexFormat_Float32, 2, false),
        ] {
            let attributes = [vertex_attribute(format, offset, 0)];
            let buffers = [vertex_buffer(16, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn vertex_attribute_contained_in_stride() {
    let test = ValidationTest::new();
    unsafe {
        for (stride, offset, success) in [(16, 0, true), (8, 0, false), (0, 1024, true)] {
            let attributes = [vertex_attribute(
                native::WGPUVertexFormat_Float32x3,
                offset,
                0,
            )];
            let buffers = [vertex_buffer(stride, &attributes)];
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    buffers: &buffers,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn many_attributes_overlapping() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let attributes = (0..limits.maxVertexAttributes)
            .map(|location| {
                vertex_attribute(
                    match location % 3 {
                        0 => native::WGPUVertexFormat_Float32x4,
                        1 => native::WGPUVertexFormat_Uint32x4,
                        _ => native::WGPUVertexFormat_Sint32x4,
                    },
                    u64::from(location * 4),
                    location,
                )
            })
            .collect::<Vec<_>>();
        let buffers = [vertex_buffer(0, &attributes)];
        expect_render_pipeline(
            &test,
            false,
            true,
            RenderPipelineCase {
                buffers: &buffers,
                ..Default::default()
            },
        );
    }
}
