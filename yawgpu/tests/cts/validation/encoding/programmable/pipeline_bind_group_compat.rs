use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, color_attachment, create_buffer, create_encoder,
    create_render_target, create_wgsl_module, empty_string_view, expect_command_buffer,
    release_render_target, render_pass_descriptor, CommandExpectation,
};

#[derive(Clone, Copy)]
enum EncoderType {
    Compute,
    Render,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResourceType {
    UniformBuffer,
    Sampler,
    SampledTexture,
    StorageTexture,
}

#[test]
fn bind_groups_and_pipeline_layout_mismatch() {
    for encoder_type in [EncoderType::Compute, EncoderType::Render] {
        for dynamic in [false, true] {
            let cases = [
                ([true, true, true], true),
                ([true, true, false], true),
                ([true, false, true], false),
                ([false, true, true], false),
                ([false, false, false], false),
            ];
            for (set_groups, success) in cases {
                run_bind_group_presence_case(encoder_type, dynamic, set_groups, false, true);
                run_bind_group_presence_case(encoder_type, dynamic, set_groups, true, success);
            }
        }
    }
}

#[test]
fn buffer_binding_render_pipeline() {
    let cases = [
        (native::WGPUBufferBindingType_Undefined, true),
        (native::WGPUBufferBindingType_Uniform, true),
        (native::WGPUBufferBindingType_Storage, false),
        (native::WGPUBufferBindingType_ReadOnlyStorage, false),
    ];
    for (pipeline_type, success) in cases {
        let test = ValidationTest::new();
        unsafe {
            let buffer = create_buffer(test.device(), 256, native::WGPUBufferUsage_Uniform);
            let group_layout = create_bind_group_layout(
                test.device(),
                &[buffer_layout(
                    0,
                    native::WGPUShaderStage_Fragment,
                    native::WGPUBufferBindingType_Uniform,
                    false,
                )],
            );
            let group =
                create_bind_group(test.device(), group_layout, &[buffer_binding(0, buffer)]);
            let pipeline_layout = create_pipeline_layout(
                test.device(),
                &[create_bind_group_layout(
                    test.device(),
                    &[buffer_layout(
                        0,
                        native::WGPUShaderStage_Fragment,
                        pipeline_type,
                        false,
                    )],
                )],
            );
            let pipeline = create_render_pipeline_with_layout(&test, pipeline_layout, false);

            encode_render_draw(&test, pipeline, &[Some(group)], None, true, success);

            yawgpu::wgpuRenderPipelineRelease(pipeline);
            yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
            yawgpu::wgpuBindGroupRelease(group);
            yawgpu::wgpuBindGroupLayoutRelease(group_layout);
            yawgpu::wgpuBufferRelease(buffer);
        }
    }
}

#[test]
fn sampler_binding_render_pipeline() {
    let types = [
        native::WGPUSamplerBindingType_Filtering,
        native::WGPUSamplerBindingType_NonFiltering,
        native::WGPUSamplerBindingType_Comparison,
    ];
    for pipeline_type in types {
        for group_type in types {
            let test = ValidationTest::new();
            unsafe {
                let sampler = create_sampler(
                    test.device(),
                    if group_type == native::WGPUSamplerBindingType_Comparison {
                        native::WGPUCompareFunction_Always
                    } else {
                        native::WGPUCompareFunction_Undefined
                    },
                );
                let group_layout = create_bind_group_layout(
                    test.device(),
                    &[sampler_layout(
                        0,
                        native::WGPUShaderStage_Fragment,
                        group_type,
                    )],
                );
                let group =
                    create_bind_group(test.device(), group_layout, &[sampler_binding(0, sampler)]);
                let pipeline_layout = create_pipeline_layout(
                    test.device(),
                    &[create_bind_group_layout(
                        test.device(),
                        &[sampler_layout(
                            0,
                            native::WGPUShaderStage_Fragment,
                            pipeline_type,
                        )],
                    )],
                );
                let pipeline = create_render_pipeline_with_layout(&test, pipeline_layout, false);

                encode_render_draw(
                    &test,
                    pipeline,
                    &[Some(group)],
                    None,
                    true,
                    pipeline_type == group_type,
                );

                yawgpu::wgpuRenderPipelineRelease(pipeline);
                yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
                yawgpu::wgpuBindGroupRelease(group);
                yawgpu::wgpuBindGroupLayoutRelease(group_layout);
                yawgpu::wgpuSamplerRelease(sampler);
            }
        }
    }
}

#[test]
fn bgl_binding_mismatch() {
    let cases = [
        (&[0, 1, 2][..], &[0, 1, 2][..], true),
        (&[0, 1, 2][..], &[0, 1, 3][..], false),
        (&[0, 2][..], &[0, 2][..], true),
        (&[0, 2][..], &[2, 0][..], true),
        (&[0, 1, 2][..], &[0, 1][..], false),
        (&[0, 1][..], &[0, 1, 2][..], false),
    ];
    for encoder_type in [EncoderType::Compute, EncoderType::Render] {
        for dynamic in [false, true] {
            for (group_bindings, pipeline_bindings, success) in cases {
                run_binding_mismatch_case(
                    encoder_type,
                    dynamic,
                    group_bindings,
                    pipeline_bindings,
                    false,
                    true,
                );
                run_binding_mismatch_case(
                    encoder_type,
                    dynamic,
                    group_bindings,
                    pipeline_bindings,
                    true,
                    success,
                );
            }
        }
    }
}

#[test]
fn bgl_visibility_mismatch() {
    let group_visibilities = [
        native::WGPUShaderStage_Vertex,
        native::WGPUShaderStage_Fragment,
        native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Fragment,
        native::WGPUShaderStage_Compute,
    ];
    for encoder_type in [EncoderType::Compute, EncoderType::Render] {
        let pipeline_visibilities = match encoder_type {
            EncoderType::Compute => &[native::WGPUShaderStage_Compute][..],
            EncoderType::Render => &[
                native::WGPUShaderStage_Vertex,
                native::WGPUShaderStage_Fragment,
                native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Fragment,
            ][..],
        };
        for group_visibility in group_visibilities {
            for pipeline_visibility in pipeline_visibilities {
                run_visibility_case(
                    encoder_type,
                    group_visibility,
                    *pipeline_visibility,
                    false,
                    true,
                );
                run_visibility_case(
                    encoder_type,
                    group_visibility,
                    *pipeline_visibility,
                    true,
                    group_visibility == *pipeline_visibility,
                );
            }
        }
    }
}

#[test]
fn bgl_resource_type_mismatch() {
    let resource_types = [
        ResourceType::UniformBuffer,
        ResourceType::Sampler,
        ResourceType::SampledTexture,
        ResourceType::StorageTexture,
    ];
    for encoder_type in [EncoderType::Compute, EncoderType::Render] {
        for group_type in resource_types {
            for pipeline_type in resource_types {
                run_resource_type_case(encoder_type, group_type, pipeline_type, false, true);
                run_resource_type_case(
                    encoder_type,
                    group_type,
                    pipeline_type,
                    true,
                    group_type == pipeline_type,
                );
            }
        }
    }
}

#[test]
fn empty_bind_group_layouts_never_requires_empty_bind_groups_compute_pass() {
    let test = ValidationTest::new();
    unsafe {
        let empty = create_bind_group_layout(test.device(), &[]);
        let layouts = [empty, empty, empty, empty];
        let pipeline_layout = create_pipeline_layout(test.device(), &layouts);
        let pipeline = create_compute_pipeline_with_layout(&test, pipeline_layout);
        let empty_group = create_bind_group(test.device(), empty, &[]);

        let encoder = create_encoder(test.device());
        let pass = begin_compute_pass(encoder, None);
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        for index in 0..4 {
            yawgpu::wgpuComputePassEncoderSetBindGroup(
                pass,
                index,
                empty_group,
                0,
                std::ptr::null(),
            );
        }
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 0, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        expect_command_buffer(&test, encoder, CommandExpectation::Success);

        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuBindGroupRelease(empty_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(empty);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn empty_bind_group_layouts_never_requires_empty_bind_groups_render_pass() {
    let test = ValidationTest::new();
    unsafe {
        let empty = create_bind_group_layout(test.device(), &[]);
        let layouts = [empty, empty, empty, empty];
        let pipeline_layout = create_pipeline_layout(test.device(), &layouts);
        let pipeline = create_render_pipeline_with_layout(&test, pipeline_layout, false);
        let empty_group = create_bind_group(test.device(), empty, &[]);

        encode_render_draw(&test, pipeline, &[Some(empty_group); 4], None, true, true);

        yawgpu::wgpuBindGroupRelease(empty_group);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(empty);
    }
}

#[test]
fn default_bind_group_layouts_never_match_compute_pass() {
    run_default_layout_case(EncoderType::Compute, false);
    run_default_layout_case(EncoderType::Compute, true);
}

#[test]
fn default_bind_group_layouts_never_match_render_pass() {
    run_default_layout_case(EncoderType::Render, false);
    run_default_layout_case(EncoderType::Render, true);
}

fn run_bind_group_presence_case(
    encoder_type: EncoderType,
    dynamic: bool,
    set_groups: [bool; 3],
    with_dispatch_or_draw: bool,
    success: bool,
) {
    let test = ValidationTest::new();
    unsafe {
        let visibility = stage_for_encoder(encoder_type);
        let entries = [buffer_layout(
            0,
            visibility,
            native::WGPUBufferBindingType_Uniform,
            dynamic,
        )];
        let layout0 = create_bind_group_layout(test.device(), &entries);
        let layout1 = create_bind_group_layout(test.device(), &entries);
        let pipeline_layout = create_pipeline_layout(test.device(), &[layout0, layout1]);
        let pipeline = create_pipeline(&test, encoder_type, pipeline_layout);
        let group0 = create_buffer_group(&test, layout0, &entries);
        let group1 = create_buffer_group(&test, layout1, &entries);
        let group2 = create_buffer_group(&test, layout1, &entries);
        let groups = [
            set_groups[0].then_some(group0),
            set_groups[1].then_some(group1),
            set_groups[2].then_some(group2),
        ];
        let offsets = dynamic.then_some(vec![0_u32]);

        encode_programmable(
            &test,
            encoder_type,
            pipeline,
            &groups,
            offsets.as_deref(),
            with_dispatch_or_draw,
            success,
        );

        release_pipeline(encoder_type, pipeline);
        for group in [group0, group1, group2] {
            yawgpu::wgpuBindGroupRelease(group);
        }
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(layout1);
        yawgpu::wgpuBindGroupLayoutRelease(layout0);
    }
}

fn run_binding_mismatch_case(
    encoder_type: EncoderType,
    dynamic: bool,
    group_bindings: &[u32],
    pipeline_bindings: &[u32],
    with_dispatch_or_draw: bool,
    success: bool,
) {
    let test = ValidationTest::new();
    unsafe {
        let visibility = stage_for_encoder(encoder_type);
        let group_entries = group_bindings
            .iter()
            .map(|binding| {
                buffer_layout(
                    *binding,
                    visibility,
                    native::WGPUBufferBindingType_Uniform,
                    dynamic,
                )
            })
            .collect::<Vec<_>>();
        let pipeline_entries = pipeline_bindings
            .iter()
            .map(|binding| {
                buffer_layout(
                    *binding,
                    visibility,
                    native::WGPUBufferBindingType_Uniform,
                    dynamic,
                )
            })
            .collect::<Vec<_>>();
        let group_layout = create_bind_group_layout(test.device(), &group_entries);
        let pipeline_group_layout = create_bind_group_layout(test.device(), &pipeline_entries);
        let pipeline_layout = create_pipeline_layout(test.device(), &[pipeline_group_layout]);
        let pipeline = create_pipeline(&test, encoder_type, pipeline_layout);
        let group = create_buffer_group(&test, group_layout, &group_entries);
        let offsets = dynamic.then(|| vec![0_u32; group_bindings.len()]);

        encode_programmable(
            &test,
            encoder_type,
            pipeline,
            &[Some(group)],
            offsets.as_deref(),
            with_dispatch_or_draw,
            success,
        );

        yawgpu::wgpuBindGroupRelease(group);
        release_pipeline(encoder_type, pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(pipeline_group_layout);
        yawgpu::wgpuBindGroupLayoutRelease(group_layout);
    }
}

fn run_visibility_case(
    encoder_type: EncoderType,
    group_visibility: native::WGPUShaderStage,
    pipeline_visibility: native::WGPUShaderStage,
    with_dispatch_or_draw: bool,
    success: bool,
) {
    let test = ValidationTest::new();
    unsafe {
        let group_entry = buffer_layout(
            0,
            group_visibility,
            native::WGPUBufferBindingType_Uniform,
            false,
        );
        let pipeline_entry = buffer_layout(
            0,
            pipeline_visibility,
            native::WGPUBufferBindingType_Uniform,
            false,
        );
        let group_layout = create_bind_group_layout(test.device(), &[group_entry]);
        let pipeline_group_layout = create_bind_group_layout(test.device(), &[pipeline_entry]);
        let pipeline_layout = create_pipeline_layout(test.device(), &[pipeline_group_layout]);
        let pipeline = create_pipeline(&test, encoder_type, pipeline_layout);
        let group = create_buffer_group(&test, group_layout, &[group_entry]);

        encode_programmable(
            &test,
            encoder_type,
            pipeline,
            &[Some(group)],
            None,
            with_dispatch_or_draw,
            success,
        );

        yawgpu::wgpuBindGroupRelease(group);
        release_pipeline(encoder_type, pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(pipeline_group_layout);
        yawgpu::wgpuBindGroupLayoutRelease(group_layout);
    }
}

fn run_resource_type_case(
    encoder_type: EncoderType,
    group_type: ResourceType,
    pipeline_type: ResourceType,
    with_dispatch_or_draw: bool,
    success: bool,
) {
    let test = ValidationTest::new();
    unsafe {
        let visibility = stage_for_encoder(encoder_type);
        let group_entry = resource_layout(0, visibility, group_type);
        let pipeline_entry = resource_layout(0, visibility, pipeline_type);
        let group_layout = create_bind_group_layout(test.device(), &[group_entry]);
        let pipeline_group_layout = create_bind_group_layout(test.device(), &[pipeline_entry]);
        let pipeline_layout = create_pipeline_layout(test.device(), &[pipeline_group_layout]);
        let pipeline = create_pipeline(&test, encoder_type, pipeline_layout);
        let resource = create_resource(test.device(), group_type);
        let binding = resource.binding(0);
        let group = create_bind_group(test.device(), group_layout, &[binding]);

        encode_programmable(
            &test,
            encoder_type,
            pipeline,
            &[Some(group)],
            None,
            with_dispatch_or_draw,
            success,
        );

        yawgpu::wgpuBindGroupRelease(group);
        resource.release();
        release_pipeline(encoder_type, pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(pipeline_group_layout);
        yawgpu::wgpuBindGroupLayoutRelease(group_layout);
    }
}

fn run_default_layout_case(encoder_type: EncoderType, empty: bool) {
    let test = ValidationTest::new();
    unsafe {
        let auto_a = create_auto_pipeline(&test, encoder_type);
        let auto_b = create_auto_pipeline(&test, encoder_type);
        let explicit_empty = create_bind_group_layout(test.device(), &[]);
        let explicit_entry = buffer_layout(
            0,
            stage_for_encoder(encoder_type),
            native::WGPUBufferBindingType_Uniform,
            false,
        );
        let explicit_non_empty = create_bind_group_layout(test.device(), &[explicit_entry]);
        let explicit_layouts = [
            explicit_empty,
            explicit_empty,
            explicit_non_empty,
            explicit_non_empty,
        ];
        let explicit_pipeline_layout = create_pipeline_layout(test.device(), &explicit_layouts);
        let explicit_pipeline = create_pipeline(&test, encoder_type, explicit_pipeline_layout);

        if empty {
            let empty0_layout = get_bind_group_layout(encoder_type, auto_a, 0);
            let empty1_layout = get_bind_group_layout(encoder_type, auto_a, 1);
            let empty0 = create_bind_group(test.device(), empty0_layout, &[]);
            let empty1 = create_bind_group(test.device(), empty1_layout, &[]);

            let auto_a_group2 = create_group_from_pipeline(&test, encoder_type, auto_a, 2);
            let auto_a_group3 = create_group_from_pipeline(&test, encoder_type, auto_a, 3);
            encode_programmable(
                &test,
                encoder_type,
                auto_a,
                &[
                    Some(empty0),
                    Some(empty1),
                    Some(auto_a_group2),
                    Some(auto_a_group3),
                ],
                None,
                true,
                true,
            );
            yawgpu::wgpuBindGroupRelease(auto_a_group3);
            yawgpu::wgpuBindGroupRelease(auto_a_group2);

            let explicit_group2 = create_buffer_group(&test, explicit_non_empty, &[explicit_entry]);
            let explicit_group3 = create_buffer_group(&test, explicit_non_empty, &[explicit_entry]);
            encode_programmable(
                &test,
                encoder_type,
                explicit_pipeline,
                &[
                    Some(empty0),
                    Some(empty1),
                    Some(explicit_group2),
                    Some(explicit_group3),
                ],
                None,
                true,
                true,
            );
            yawgpu::wgpuBindGroupRelease(explicit_group3);
            yawgpu::wgpuBindGroupRelease(explicit_group2);

            let auto_b_group2 = create_group_from_pipeline(&test, encoder_type, auto_b, 2);
            let auto_b_group3 = create_group_from_pipeline(&test, encoder_type, auto_b, 3);
            encode_programmable(
                &test,
                encoder_type,
                auto_b,
                &[
                    Some(empty0),
                    Some(empty1),
                    Some(auto_b_group2),
                    Some(auto_b_group3),
                ],
                None,
                true,
                true,
            );
            yawgpu::wgpuBindGroupRelease(auto_b_group3);
            yawgpu::wgpuBindGroupRelease(auto_b_group2);

            yawgpu::wgpuBindGroupRelease(empty1);
            yawgpu::wgpuBindGroupRelease(empty0);
            yawgpu::wgpuBindGroupLayoutRelease(empty1_layout);
            yawgpu::wgpuBindGroupLayoutRelease(empty0_layout);
        } else {
            let empty_group = create_bind_group(test.device(), explicit_empty, &[]);
            let source_group2 = create_group_from_pipeline(&test, encoder_type, auto_a, 2);
            let source_group3 = create_group_from_pipeline(&test, encoder_type, auto_a, 3);
            encode_programmable(
                &test,
                encoder_type,
                auto_a,
                &[
                    Some(empty_group),
                    Some(empty_group),
                    Some(source_group2),
                    Some(source_group3),
                ],
                None,
                true,
                true,
            );
            encode_programmable(
                &test,
                encoder_type,
                explicit_pipeline,
                &[
                    Some(empty_group),
                    Some(empty_group),
                    Some(source_group2),
                    Some(source_group3),
                ],
                None,
                true,
                false,
            );
            encode_programmable(
                &test,
                encoder_type,
                auto_b,
                &[
                    Some(empty_group),
                    Some(empty_group),
                    Some(source_group2),
                    Some(source_group3),
                ],
                None,
                true,
                false,
            );
            yawgpu::wgpuBindGroupRelease(source_group3);
            yawgpu::wgpuBindGroupRelease(source_group2);
            yawgpu::wgpuBindGroupRelease(empty_group);
        }
        release_pipeline(encoder_type, explicit_pipeline);
        yawgpu::wgpuPipelineLayoutRelease(explicit_pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(explicit_non_empty);
        yawgpu::wgpuBindGroupLayoutRelease(explicit_empty);
        release_pipeline(encoder_type, auto_b);
        release_pipeline(encoder_type, auto_a);
    }
}

unsafe fn create_group_from_pipeline(
    test: &ValidationTest,
    encoder_type: EncoderType,
    pipeline: *const std::ffi::c_void,
    index: u32,
) -> native::WGPUBindGroup {
    let layout = unsafe { get_bind_group_layout(encoder_type, pipeline, index) };
    let entry = buffer_layout(
        0,
        stage_for_encoder(encoder_type),
        native::WGPUBufferBindingType_Uniform,
        false,
    );
    let group = unsafe { create_buffer_group(test, layout, &[entry]) };
    unsafe { yawgpu::wgpuBindGroupLayoutRelease(layout) };
    group
}

unsafe fn encode_programmable(
    test: &ValidationTest,
    encoder_type: EncoderType,
    pipeline: *const std::ffi::c_void,
    groups: &[Option<native::WGPUBindGroup>],
    dynamic_offsets: Option<&[u32]>,
    with_dispatch_or_draw: bool,
    success: bool,
) {
    match encoder_type {
        EncoderType::Compute => unsafe {
            let encoder = create_encoder(test.device());
            let pass = begin_compute_pass(encoder, None);
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline.cast());
            set_compute_bind_groups(pass, groups, dynamic_offsets);
            if with_dispatch_or_draw {
                yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 0, 1, 1);
            }
            yawgpu::wgpuComputePassEncoderEnd(pass);
            expect_encoding_result(test, encoder, success);
            yawgpu::wgpuComputePassEncoderRelease(pass);
            yawgpu::wgpuCommandEncoderRelease(encoder);
        },
        EncoderType::Render => unsafe {
            encode_render_draw(
                test,
                pipeline.cast(),
                groups,
                dynamic_offsets,
                with_dispatch_or_draw,
                success,
            );
        },
    }
}

unsafe fn encode_render_draw(
    test: &ValidationTest,
    pipeline: native::WGPURenderPipeline,
    groups: &[Option<native::WGPUBindGroup>],
    dynamic_offsets: Option<&[u32]>,
    with_draw: bool,
    success: bool,
) {
    let encoder = unsafe { create_encoder(test.device()) };
    let target =
        unsafe { create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1) };
    let attachment = color_attachment(target.view);
    let descriptor = render_pass_descriptor(&[attachment], None);
    let pass = unsafe { begin_render_pass(encoder, &descriptor) };
    unsafe {
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        set_render_bind_groups(pass, groups, dynamic_offsets);
        if with_draw {
            yawgpu::wgpuRenderPassEncoderDraw(pass, 0, 1, 0, 0);
        }
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_encoding_result(test, encoder, success);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn expect_encoding_result(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    success: bool,
) {
    if success {
        unsafe { expect_command_buffer(test, encoder, CommandExpectation::Success) };
        return;
    }

    if test.errors().is_empty() {
        unsafe { expect_command_buffer(test, encoder, CommandExpectation::FinishError) };
    } else {
        test.clear_errors();
        unsafe { expect_command_buffer(test, encoder, CommandExpectation::Success) };
    }
}

unsafe fn set_compute_bind_groups(
    pass: native::WGPUComputePassEncoder,
    groups: &[Option<native::WGPUBindGroup>],
    dynamic_offsets: Option<&[u32]>,
) {
    for (index, group) in groups.iter().enumerate() {
        let Some(group) = group else {
            break;
        };
        let (count, offsets) = offsets(dynamic_offsets);
        unsafe {
            yawgpu::wgpuComputePassEncoderSetBindGroup(pass, index as u32, *group, count, offsets);
        }
    }
}

unsafe fn set_render_bind_groups(
    pass: native::WGPURenderPassEncoder,
    groups: &[Option<native::WGPUBindGroup>],
    dynamic_offsets: Option<&[u32]>,
) {
    for (index, group) in groups.iter().enumerate() {
        let Some(group) = group else {
            break;
        };
        let (count, offsets) = offsets(dynamic_offsets);
        unsafe {
            yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, index as u32, *group, count, offsets);
        }
    }
}

fn offsets(dynamic_offsets: Option<&[u32]>) -> (usize, *const u32) {
    dynamic_offsets.map_or((0, std::ptr::null()), |offsets| {
        (offsets.len(), offsets.as_ptr())
    })
}

unsafe fn create_pipeline(
    test: &ValidationTest,
    encoder_type: EncoderType,
    layout: native::WGPUPipelineLayout,
) -> *const std::ffi::c_void {
    match encoder_type {
        EncoderType::Compute => unsafe { create_compute_pipeline_with_layout(test, layout).cast() },
        EncoderType::Render => unsafe {
            create_render_pipeline_with_layout(test, layout, false).cast()
        },
    }
}

unsafe fn release_pipeline(encoder_type: EncoderType, pipeline: *const std::ffi::c_void) {
    unsafe {
        match encoder_type {
            EncoderType::Compute => yawgpu::wgpuComputePipelineRelease(pipeline.cast()),
            EncoderType::Render => yawgpu::wgpuRenderPipelineRelease(pipeline.cast()),
        }
    }
}

unsafe fn create_auto_pipeline(
    test: &ValidationTest,
    encoder_type: EncoderType,
) -> *const std::ffi::c_void {
    match encoder_type {
        EncoderType::Compute => unsafe {
            let module = create_wgsl_module(
                test.device(),
                r#"
@group(2) @binding(0) var<uniform> u1: vec4f;
@group(3) @binding(0) var<uniform> u2: vec4f;
@compute @workgroup_size(1) fn main() { _ = u1; _ = u2; }
"#,
            );
            let descriptor = native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null(),
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
            assert!(
                test.errors().is_empty(),
                "unexpected auto compute pipeline errors: {:?}",
                test.errors()
            );
            yawgpu::wgpuShaderModuleRelease(module);
            assert!(!pipeline.is_null());
            pipeline.cast()
        },
        EncoderType::Render => unsafe {
            create_render_pipeline_with_layout(test, std::ptr::null(), true).cast()
        },
    }
}

unsafe fn get_bind_group_layout(
    encoder_type: EncoderType,
    pipeline: *const std::ffi::c_void,
    index: u32,
) -> native::WGPUBindGroupLayout {
    unsafe {
        match encoder_type {
            EncoderType::Compute => {
                yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline.cast(), index)
            }
            EncoderType::Render => {
                yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline.cast(), index)
            }
        }
    }
}

unsafe fn create_compute_pipeline_with_layout(
    test: &ValidationTest,
    layout: native::WGPUPipelineLayout,
) -> native::WGPUComputePipeline {
    let module =
        unsafe { create_wgsl_module(test.device(), "@compute @workgroup_size(1) fn main() {}") };
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    test.clear_errors();
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor) };
    assert!(
        test.errors().is_empty(),
        "unexpected compute pipeline errors: {:?}",
        test.errors()
    );
    assert!(!pipeline.is_null());
    unsafe { yawgpu::wgpuShaderModuleRelease(module) };
    pipeline
}

unsafe fn create_render_pipeline_with_layout(
    test: &ValidationTest,
    layout: native::WGPUPipelineLayout,
    auto_resources: bool,
) -> native::WGPURenderPipeline {
    let vertex_source = if auto_resources {
        r#"
@group(2) @binding(0) var<uniform> u1: vec4f;
@group(3) @binding(0) var<uniform> u2: vec4f;
@vertex fn main() -> @builtin(position) vec4f { return u1 + u2; }
"#
    } else {
        "@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }"
    };
    let vertex = unsafe { create_wgsl_module(test.device(), vertex_source) };
    let fragment = unsafe { create_wgsl_module(test.device(), "@fragment fn main() {}") };
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_None,
    };
    let fragment_state = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment_state,
    };
    test.clear_errors();
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) };
    assert!(
        test.errors().is_empty(),
        "unexpected render pipeline errors: {:?}",
        test.errors()
    );
    assert!(!pipeline.is_null());
    unsafe {
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
    pipeline
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
    let layout = unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) };
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
    let layout = unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) };
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
    let group = unsafe { yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor) };
    assert!(!group.is_null());
    group
}

unsafe fn create_buffer_group(
    test: &ValidationTest,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroup {
    let buffer = unsafe { create_buffer(test.device(), 1024, native::WGPUBufferUsage_Uniform) };
    let bindings = entries
        .iter()
        .map(|entry| buffer_binding(entry.binding, buffer))
        .collect::<Vec<_>>();
    unsafe { create_bind_group(test.device(), layout, &bindings) }
}

fn stage_for_encoder(encoder_type: EncoderType) -> native::WGPUShaderStage {
    match encoder_type {
        EncoderType::Compute => native::WGPUShaderStage_Compute,
        EncoderType::Render => native::WGPUShaderStage_Vertex,
    }
}

fn resource_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    resource_type: ResourceType,
) -> native::WGPUBindGroupLayoutEntry {
    match resource_type {
        ResourceType::UniformBuffer => buffer_layout(
            binding,
            visibility,
            native::WGPUBufferBindingType_Uniform,
            false,
        ),
        ResourceType::Sampler => sampler_layout(
            binding,
            visibility,
            native::WGPUSamplerBindingType_Filtering,
        ),
        ResourceType::SampledTexture => texture_layout(binding, visibility),
        ResourceType::StorageTexture => storage_texture_layout(binding, visibility),
    }
}

fn buffer_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUBufferBindingType,
    dynamic: bool,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = unused_layout(binding, visibility);
    entry.buffer.type_ = if ty == native::WGPUBufferBindingType_Undefined {
        native::WGPUBufferBindingType_Uniform
    } else {
        ty
    };
    entry.buffer.hasDynamicOffset = u32::from(dynamic);
    entry
}

fn sampler_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
    ty: native::WGPUSamplerBindingType,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = unused_layout(binding, visibility);
    entry.sampler.type_ = ty;
    entry
}

fn texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = unused_layout(binding, visibility);
    entry.texture.sampleType = native::WGPUTextureSampleType_Float;
    entry.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn storage_texture_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = unused_layout(binding, visibility);
    entry.storageTexture.access = if visibility & native::WGPUShaderStage_Vertex != 0 {
        native::WGPUStorageTextureAccess_ReadOnly
    } else {
        native::WGPUStorageTextureAccess_WriteOnly
    };
    entry.storageTexture.format = native::WGPUTextureFormat_R32Float;
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn unused_layout(
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

fn buffer_binding(binding: u32, buffer: native::WGPUBuffer) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset: 0,
        size: 256,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn sampler_binding(binding: u32, sampler: native::WGPUSampler) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler,
        textureView: std::ptr::null(),
    }
}

fn texture_binding(binding: u32, view: native::WGPUTextureView) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: u64::MAX,
        sampler: std::ptr::null(),
        textureView: view,
    }
}

unsafe fn create_sampler(
    device: native::WGPUDevice,
    compare: native::WGPUCompareFunction,
) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_Undefined,
        addressModeV: native::WGPUAddressMode_Undefined,
        addressModeW: native::WGPUAddressMode_Undefined,
        magFilter: native::WGPUFilterMode_Undefined,
        minFilter: native::WGPUFilterMode_Undefined,
        mipmapFilter: native::WGPUMipmapFilterMode_Undefined,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare,
        maxAnisotropy: 1,
    };
    let sampler = unsafe { yawgpu::wgpuDeviceCreateSampler(device, &descriptor) };
    assert!(!sampler.is_null());
    sampler
}

enum Resource {
    Buffer(native::WGPUBuffer),
    Sampler(native::WGPUSampler),
    Texture(native::WGPUTexture, native::WGPUTextureView),
}

impl Resource {
    fn binding(&self, binding: u32) -> native::WGPUBindGroupEntry {
        match *self {
            Self::Buffer(buffer) => buffer_binding(binding, buffer),
            Self::Sampler(sampler) => sampler_binding(binding, sampler),
            Self::Texture(_, view) => texture_binding(binding, view),
        }
    }

    unsafe fn release(self) {
        unsafe {
            match self {
                Self::Buffer(buffer) => yawgpu::wgpuBufferRelease(buffer),
                Self::Sampler(sampler) => yawgpu::wgpuSamplerRelease(sampler),
                Self::Texture(texture, view) => {
                    yawgpu::wgpuTextureViewRelease(view);
                    yawgpu::wgpuTextureRelease(texture);
                }
            }
        }
    }
}

unsafe fn create_resource(device: native::WGPUDevice, resource_type: ResourceType) -> Resource {
    match resource_type {
        ResourceType::UniformBuffer => unsafe {
            Resource::Buffer(create_buffer(device, 256, native::WGPUBufferUsage_Uniform))
        },
        ResourceType::Sampler => unsafe {
            Resource::Sampler(create_sampler(
                device,
                native::WGPUCompareFunction_Undefined,
            ))
        },
        ResourceType::SampledTexture => unsafe {
            create_texture_resource(
                device,
                native::WGPUTextureUsage_TextureBinding,
                native::WGPUTextureFormat_RGBA8Unorm,
            )
        },
        ResourceType::StorageTexture => unsafe {
            create_texture_resource(
                device,
                native::WGPUTextureUsage_StorageBinding,
                native::WGPUTextureFormat_R32Float,
            )
        },
    }
}

unsafe fn create_texture_resource(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
) -> Resource {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    Resource::Texture(texture, view)
}
