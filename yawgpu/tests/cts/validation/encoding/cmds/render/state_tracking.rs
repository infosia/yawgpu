//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/state_tracking.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    color_attachment, create_buffer, create_encoder, create_render_pipeline_with,
    create_render_target, default_primitive, expect_command_buffer, release_render_target,
    render_pass_descriptor, set_pipeline, set_vertex_buffer, vertex_attribute,
    vertex_buffer_layout, CommandExpectation, RenderEncoder,
};

#[test]
#[ignore = "CTS marks this g.test unimplemented; no normative all-needed vertex buffer subcases to port yet"]
fn all_needed_vertex_buffer_should_be_bound() {}

#[test]
#[ignore = "CTS marks this g.test unimplemented; no normative all-needed index buffer subcases to port yet"]
fn all_needed_index_buffer_should_be_bound() {}

#[test]
fn vertex_buffers_inherit_from_previous_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline1 = create_pipeline(&test, 1);
        let pipeline2 = create_pipeline(&test, 2);
        let vertex1 = create_buffer(test.device(), 256, native::WGPUBufferUsage_Vertex);
        let vertex2 = create_buffer(test.device(), 256, native::WGPUBufferUsage_Vertex);

        expect_one_pass(&test, CommandExpectation::FinishError, |pass| {
            let encoder = RenderEncoder::RenderPass(pass);
            set_pipeline(&encoder, pipeline1);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

        expect_one_pass(&test, CommandExpectation::Success, |pass| {
            let encoder = RenderEncoder::RenderPass(pass);
            set_pipeline(&encoder, pipeline2);
            set_vertex_buffer(&encoder, 0, vertex1, 0, u64::MAX);
            set_vertex_buffer(&encoder, 1, vertex2, 0, u64::MAX);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            set_pipeline(&encoder, pipeline1);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

        yawgpu::wgpuBufferRelease(vertex2);
        yawgpu::wgpuBufferRelease(vertex1);
        yawgpu::wgpuRenderPipelineRelease(pipeline2);
        yawgpu::wgpuRenderPipelineRelease(pipeline1);
    }
}

#[test]
fn vertex_buffers_do_not_inherit_between_render_passes() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline1 = create_pipeline(&test, 1);
        let pipeline2 = create_pipeline(&test, 2);
        let vertex1 = create_buffer(test.device(), 256, native::WGPUBufferUsage_Vertex);
        let vertex2 = create_buffer(test.device(), 256, native::WGPUBufferUsage_Vertex);

        expect_two_passes(
            &test,
            CommandExpectation::Success,
            |pass| {
                let encoder = RenderEncoder::RenderPass(pass);
                set_pipeline(&encoder, pipeline2);
                set_vertex_buffer(&encoder, 0, vertex1, 0, u64::MAX);
                set_vertex_buffer(&encoder, 1, vertex2, 0, u64::MAX);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            },
            |pass| {
                let encoder = RenderEncoder::RenderPass(pass);
                set_pipeline(&encoder, pipeline1);
                set_vertex_buffer(&encoder, 0, vertex1, 0, u64::MAX);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            },
        );

        expect_two_passes(
            &test,
            CommandExpectation::FinishError,
            |pass| {
                let encoder = RenderEncoder::RenderPass(pass);
                set_pipeline(&encoder, pipeline2);
                set_vertex_buffer(&encoder, 0, vertex1, 0, u64::MAX);
                set_vertex_buffer(&encoder, 1, vertex2, 0, u64::MAX);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            },
            |pass| {
                let encoder = RenderEncoder::RenderPass(pass);
                set_pipeline(&encoder, pipeline1);
                yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            },
        );

        yawgpu::wgpuBufferRelease(vertex2);
        yawgpu::wgpuBufferRelease(vertex1);
        yawgpu::wgpuRenderPipelineRelease(pipeline2);
        yawgpu::wgpuRenderPipelineRelease(pipeline1);
    }
}

unsafe fn create_pipeline(test: &ValidationTest, buffer_count: u32) -> native::WGPURenderPipeline {
    let attributes = (0..buffer_count)
        .map(|slot| vertex_attribute(native::WGPUVertexFormat_Float32x3, 0, slot))
        .collect::<Vec<_>>();
    let layouts = attributes
        .iter()
        .map(|attribute| {
            vertex_buffer_layout(
                native::WGPUVertexStepMode_Vertex,
                12,
                std::slice::from_ref(attribute),
            )
        })
        .collect::<Vec<_>>();
    let params = (0..buffer_count)
        .map(|slot| format!("@location({slot}) a{slot}: vec3f"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "@vertex fn vs({params}) -> @builtin(position) vec4f {{ return vec4f(0.0, 0.0, 0.0, 1.0); }}"
    );
    unsafe {
        create_render_pipeline_with(test, test.device(), &source, &layouts, default_primitive())
    }
}

unsafe fn expect_one_pass<F>(test: &ValidationTest, expectation: CommandExpectation, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    unsafe {
        let encoder = create_encoder(test.device());
        let target = create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
        let attachment = color_attachment(target.view);
        let attachments = [attachment];
        let descriptor = render_pass_descriptor(&attachments, None);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        commands(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn expect_two_passes<F, G>(
    test: &ValidationTest,
    expectation: CommandExpectation,
    first: F,
    second: G,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
    G: FnOnce(native::WGPURenderPassEncoder),
{
    unsafe {
        let encoder = create_encoder(test.device());
        let target = create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
        let attachment = color_attachment(target.view);
        let attachments = [attachment];
        let descriptor = render_pass_descriptor(&attachments, None);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        first(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        second(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}
