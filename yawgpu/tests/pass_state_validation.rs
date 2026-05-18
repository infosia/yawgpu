use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn render_draw_validates_pipeline_bind_groups_vertex_buffers_and_index_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let vertex = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 256);
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 256);
        let bind_group_layout = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Vertex, false)],
        );
        let bind_group = create_bind_group(
            test.device(),
            bind_group_layout,
            &[buffer_binding(0, uniform, 0, 256)],
        );
        let pipeline_layout = create_pipeline_layout(test.device(), &[bind_group_layout]);
        let attribute = vertex_attribute(native::WGPUVertexFormat_Float32x2, 0, 0);
        let vertex_buffer = vertex_buffer(8, &[attribute]);
        let pipeline = create_render_pipeline(
            &test,
            render_uniform_vertex_input(),
            Some(pipeline_layout),
            &[vertex_buffer],
        );

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 256);
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 256);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
        });

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(bind_group_layout);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuBufferRelease(vertex);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn render_draw_validates_incompatible_layouts_and_dynamic_offsets() {
    let test = ValidationTest::new();
    unsafe {
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let storage = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 1024);
        let dynamic_layout = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Vertex, true)],
        );
        let incompatible_layout = create_bind_group_layout(
            test.device(),
            &[storage_layout(0, native::WGPUShaderStage_Vertex)],
        );
        let dynamic_group = create_bind_group(
            test.device(),
            dynamic_layout,
            &[buffer_binding(0, uniform, 0, 512)],
        );
        let incompatible_group = create_bind_group(
            test.device(),
            incompatible_layout,
            &[buffer_binding(0, storage, 0, 512)],
        );
        let pipeline_layout = create_pipeline_layout(test.device(), &[dynamic_layout]);
        let pipeline =
            create_render_pipeline(&test, render_uniform_no_input(), Some(pipeline_layout), &[]);

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(
                pass,
                0,
                incompatible_group,
                0,
                std::ptr::null(),
            );
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, dynamic_group, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        let misaligned = [1_u32];
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(
                pass,
                0,
                dynamic_group,
                1,
                misaligned.as_ptr(),
            );
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        let aligned = [256_u32];
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, dynamic_group, 1, aligned.as_ptr());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupRelease(incompatible_group);
        yawgpu::wgpuBindGroupRelease(dynamic_group);
        yawgpu::wgpuBindGroupLayoutRelease(incompatible_layout);
        yawgpu::wgpuBindGroupLayoutRelease(dynamic_layout);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn default_render_bind_group_layouts_are_pipeline_bound_at_draw() {
    let test = ValidationTest::new();
    unsafe {
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let pipeline_a = create_render_pipeline(&test, render_auto_a(), None, &[]);
        let pipeline_b = create_render_pipeline(&test, render_auto_b(), None, &[]);
        let layout_a = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline_a, 0);
        let group_a = create_bind_group(
            test.device(),
            layout_a,
            &[buffer_binding(0, uniform, 0, 256)],
        );

        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_a);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group_a, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_b);
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group_a, 0, std::ptr::null());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });

        yawgpu::wgpuBindGroupRelease(group_a);
        yawgpu::wgpuBindGroupLayoutRelease(layout_a);
        yawgpu::wgpuRenderPipelineRelease(pipeline_b);
        yawgpu::wgpuRenderPipelineRelease(pipeline_a);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn compute_dispatch_validates_state_and_workgroup_limits() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let layout = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Compute, false)],
        );
        let group = create_bind_group(test.device(), layout, &[buffer_binding(0, uniform, 0, 256)]);
        let pipeline_layout = create_pipeline_layout(test.device(), &[layout]);
        let pipeline = create_compute_pipeline(&test, compute_uniform(), Some(pipeline_layout));

        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        });
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        });
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
            yawgpu::wgpuComputePassEncoderDispatchWorkgroups(
                pass,
                limits.maxComputeWorkgroupsPerDimension + 1,
                1,
                1,
            );
        });
        assert_compute_pass_ok(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
            yawgpu::wgpuComputePassEncoderDispatchWorkgroups(
                pass,
                limits.maxComputeWorkgroupsPerDimension,
                1,
                1,
            );
        });

        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn render_dynamic_state_commands_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, u32::MAX, 0, 1, 1);
        });
        let bad_color = native::WGPUColor {
            r: f64::NAN,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetBlendConstant(pass, &bad_color);
        });
        let good_color = native::WGPUColor {
            r: 0.0,
            g: 0.25,
            b: 0.5,
            a: 1.0,
        };
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
            yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, 4, 4);
            yawgpu::wgpuRenderPassEncoderSetBlendConstant(pass, &good_color);
            yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, 1);
        });
    }
}

#[test]
fn set_index_and_vertex_buffer_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 256);
        let vertex = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 256);
        let copy = create_buffer(test.device(), native::WGPUBufferUsage_CopySrc, 256);

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                copy,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Undefined,
                0,
                256,
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint32,
                2,
                16,
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint32,
                260,
                0,
            );
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint32,
                4,
                252,
            );
        });

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, copy, 0, 256);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 2, 16);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 260, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(
                pass,
                limits.maxVertexBuffers,
                vertex,
                0,
                16,
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(
                pass,
                limits.maxVertexBuffers,
                std::ptr::null(),
                0,
                0,
            );
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 4, 252);
        });

        yawgpu::wgpuBufferRelease(copy);
        yawgpu::wgpuBufferRelease(vertex);
        yawgpu::wgpuBufferRelease(index);
    }
}

#[test]
fn draw_vertex_instance_and_index_oob_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 24);
        let instance = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 16);
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 6);
        let vertex_attr = [vertex_attribute(native::WGPUVertexFormat_Float32x2, 0, 0)];
        let instance_attr = [vertex_attribute(native::WGPUVertexFormat_Float32x2, 0, 1)];
        let layouts = [
            vertex_buffer(8, &vertex_attr),
            vertex_buffer_with_step(8, native::WGPUVertexStepMode_Instance, &instance_attr),
        ];
        let pipeline =
            create_render_pipeline(&test, render_vertex_instance_input(), None, &layouts);

        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 24);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, instance, 0, 16);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 2, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 24);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, instance, 0, 16);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 4, 1, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 24);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, instance, 0, 16);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 3, 0, 0);
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 24);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, instance, 0, 16);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                6,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 2, 0, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 24);
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, instance, 0, 16);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                6,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 4, 1, 0, 0, 0);
        });

        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuBufferRelease(instance);
        yawgpu::wgpuBufferRelease(vertex);
    }
}

#[test]
fn indexed_strip_draw_requires_matching_pipeline_strip_index_format() {
    let test = ValidationTest::new();
    unsafe {
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 256);
        let no_strip = primitive(
            native::WGPUPrimitiveTopology_TriangleStrip,
            native::WGPUIndexFormat_Undefined,
        );
        let strip_u32 = primitive(
            native::WGPUPrimitiveTopology_TriangleStrip,
            native::WGPUIndexFormat_Uint32,
        );
        let strip_u16 = primitive(
            native::WGPUPrimitiveTopology_TriangleStrip,
            native::WGPUIndexFormat_Uint16,
        );
        let pipeline_no_strip =
            create_render_pipeline_with_primitive(&test, render_no_input(), None, &[], no_strip);
        let pipeline_u32 =
            create_render_pipeline_with_primitive(&test, render_no_input(), None, &[], strip_u32);
        let pipeline_u16 =
            create_render_pipeline_with_primitive(&test, render_no_input(), None, &[], strip_u16);

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_no_strip);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_u32);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_u16);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
        });

        yawgpu::wgpuRenderPipelineRelease(pipeline_u16);
        yawgpu::wgpuRenderPipelineRelease(pipeline_u32);
        yawgpu::wgpuRenderPipelineRelease(pipeline_no_strip);
        yawgpu::wgpuBufferRelease(index);
    }
}

#[test]
fn indirect_draw_and_dispatch_buffers_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let indirect = create_buffer(test.device(), native::WGPUBufferUsage_Indirect, 64);
        let non_indirect = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 64);
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 64);
        let render_pipeline = create_render_pipeline(&test, render_no_input(), None, &[]);
        let compute_pipeline = create_compute_pipeline(&test, compute_empty(), None);

        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
            yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, non_indirect, 0);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
            yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect, 2);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
            yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect, 52);
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
            yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect, 48);
        });
        assert_render_pass_ok(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, render_pipeline);
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                64,
            );
            yawgpu::wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect, 44);
        });
        assert_compute_pass_ok(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, compute_pipeline);
            yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 52);
        });
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, compute_pipeline);
            yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 56);
        });

        yawgpu::wgpuComputePipelineRelease(compute_pipeline);
        yawgpu::wgpuRenderPipelineRelease(render_pipeline);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuBufferRelease(non_indirect);
        yawgpu::wgpuBufferRelease(indirect);
    }
}

unsafe fn assert_render_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(test.device());
    let color_attachment = color_attachment(target.view);
    let descriptor = render_pass_descriptor(&[color_attachment]);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_render_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(test.device());
    let color_attachment = color_attachment(target.view);
    let descriptor = render_pass_descriptor(&[color_attachment]);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_compute_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_compute_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    command_buffer
}

unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

unsafe fn create_encoder(test: &ValidationTest) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    layout: Option<native::WGPUPipelineLayout>,
    vertex_buffers: &[native::WGPUVertexBufferLayout],
) -> native::WGPURenderPipeline {
    create_render_pipeline_with_primitive(
        test,
        vertex_source,
        layout,
        vertex_buffers,
        default_primitive(),
    )
}

unsafe fn create_render_pipeline_with_primitive(
    test: &ValidationTest,
    vertex_source: &str,
    layout: Option<native::WGPUPipelineLayout>,
    vertex_buffers: &[native::WGPUVertexBufferLayout],
    primitive: native::WGPUPrimitiveState,
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), fragment_source());
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: vertex_buffers.len(),
            buffers: vertex_buffers.as_ptr(),
        },
        primitive,
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
        fragment: &fragment,
    };
    test.clear_errors();
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected render pipeline errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuShaderModuleRelease(fragment_module);
    yawgpu::wgpuShaderModuleRelease(vertex_module);
    pipeline
}

unsafe fn create_compute_pipeline(
    test: &ValidationTest,
    source: &str,
    layout: Option<native::WGPUPipelineLayout>,
) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(test.device(), source);
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    test.clear_errors();
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected compute pipeline errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
}

unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: layouts.len(),
        bindGroupLayouts: layouts.as_ptr(),
        immediateSize: 0,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!group.is_null());
    group
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

struct RenderTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn create_render_target(device: native::WGPUDevice) -> RenderTarget {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    RenderTarget { texture, view }
}

unsafe fn release_render_target(target: RenderTarget) {
    yawgpu::wgpuTextureViewRelease(target.view);
    yawgpu::wgpuTextureRelease(target.texture);
}

fn render_pass_descriptor(
    attachments: &[native::WGPURenderPassColorAttachment],
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Load,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    }
}

fn buffer_binding(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn uniform_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    has_dynamic_offset: bool,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding, visibility);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.hasDynamicOffset = has_dynamic_offset.into();
    entry.buffer.minBindingSize = 16;
    entry
}

fn storage_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding, visibility);
    entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
    entry
}

fn default_bind_group_layout_entry(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    }
}

fn vertex_buffer(
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    vertex_buffer_with_step(array_stride, native::WGPUVertexStepMode_Vertex, attributes)
}

fn vertex_buffer_with_step(
    array_stride: u64,
    step_mode: native::WGPUVertexStepMode,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: step_mode,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

fn vertex_attribute(
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
) -> native::WGPUVertexAttribute {
    native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format,
        offset,
        shaderLocation: shader_location,
    }
}

fn default_primitive() -> native::WGPUPrimitiveState {
    primitive(
        native::WGPUPrimitiveTopology_TriangleList,
        native::WGPUIndexFormat_Undefined,
    )
}

fn primitive(
    topology: native::WGPUPrimitiveTopology,
    strip_index_format: native::WGPUIndexFormat,
) -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology,
        stripIndexFormat: strip_index_format,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuDeviceGetLimits(device, &mut limits),
        native::WGPUStatus_Success
    );
    limits
}

fn render_uniform_vertex_input() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
         return vec4f(pos, 0.0, 1.0) + u.value * 0.0;
     }"
}

fn render_uniform_no_input() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs() -> @builtin(position) vec4f { return u.value; }"
}

fn render_vertex_instance_input() -> &'static str {
    "@vertex fn vs(
        @location(0) pos: vec2f,
        @location(1) inst: vec2f
     ) -> @builtin(position) vec4f {
         return vec4f(pos + inst * 0.0, 0.0, 1.0);
     }"
}

fn render_no_input() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn render_auto_a() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs() -> @builtin(position) vec4f { return u.value; }"
}

fn render_auto_b() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs() -> @builtin(position) vec4f { return u.value + vec4f(0.0); }"
}

fn fragment_source() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
}

fn compute_uniform() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @compute @workgroup_size(1) fn main() { _ = u.value; }"
}

fn compute_empty() -> &'static str {
    "@compute @workgroup_size(1) fn main() {}"
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
