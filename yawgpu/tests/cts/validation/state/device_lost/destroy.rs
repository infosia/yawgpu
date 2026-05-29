//! CTS port of `webgpu/api/validation/state/device_lost/destroy.spec.ts`.
//!
//! N/A web-only cases in this CTS file:
//! - `importExternalTexture`
//! - `queue,copyExternalImageToTexture,canvas`
//! - `queue,copyExternalImageToTexture,imageBitmap`

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

use crate::common::{self, ComputePipelineAsyncState, RenderPipelineAsyncState};

#[derive(Clone, Copy)]
enum CommandStage {
    Finish,
    Submit,
}

#[test]
fn create_buffer() {
    for usage in [
        native::WGPUBufferUsage_CopySrc,
        native::WGPUBufferUsage_CopyDst,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
        native::WGPUBufferUsage_Uniform,
    ] {
        for mapped_at_creation in [false, true] {
            for await_lost in [false, true] {
                let test = ValidationTest::new();
                unsafe {
                    execute_after_destroy(&test, await_lost, || {
                        let buffer =
                            common::create_buffer(test.device(), 16, usage, mapped_at_creation);
                        yawgpu::wgpuBufferRelease(buffer);
                    });
                }
            }
        }
    }
}

#[test]
fn create_texture_2d_uncompressed_format() {
    for (format, usage) in [
        (
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_TextureBinding,
        ),
        (
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_CopyDst,
        ),
        (
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_RenderAttachment,
        ),
        (
            native::WGPUTextureFormat_R32Float,
            native::WGPUTextureUsage_TextureBinding,
        ),
    ] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                execute_after_destroy(&test, await_lost, || {
                    let texture = common::create_texture(test.device(), format, usage, 4, 4);
                    yawgpu::wgpuTextureRelease(texture);
                });
            }
        }
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression features; CTS compressed-format device-lost subcases require a compressed texture format"]
fn create_texture_2d_compressed_format() {}

#[test]
fn create_view_2d_uncompressed_format() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let texture = common::create_texture(
                test.device(),
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_TextureBinding,
                4,
                4,
            );
            execute_after_destroy(&test, await_lost, || {
                let view = common::create_texture_view(texture);
                yawgpu::wgpuTextureViewRelease(view);
            });
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression features; CTS compressed-format device-lost subcases require a compressed texture format"]
fn create_view_2d_compressed_format() {}

#[test]
fn create_sampler() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            execute_after_destroy(&test, await_lost, || {
                let sampler = common::create_sampler(test.device());
                yawgpu::wgpuSamplerRelease(sampler);
            });
        }
    }
}

#[test]
fn create_bind_group_layout() {
    for visibility in [
        native::WGPUShaderStage_Vertex,
        native::WGPUShaderStage_Fragment,
        native::WGPUShaderStage_Compute,
    ] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                execute_after_destroy(&test, await_lost, || {
                    let entry = common::uniform_layout_entry(visibility);
                    let layout = common::create_bind_group_layout(test.device(), &[entry]);
                    yawgpu::wgpuBindGroupLayoutRelease(layout);
                });
            }
        }
    }
}

#[test]
fn create_bind_group() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let entry = common::uniform_layout_entry(native::WGPUShaderStage_Compute);
            let layout = common::create_bind_group_layout(test.device(), &[entry]);
            let buffer =
                common::create_buffer(test.device(), 16, native::WGPUBufferUsage_Uniform, false);
            execute_after_destroy(&test, await_lost, || {
                let binding = common::buffer_binding(0, buffer);
                let group = common::create_bind_group(test.device(), layout, &[binding]);
                yawgpu::wgpuBindGroupRelease(group);
            });
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
fn create_pipeline_layout() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let entry = common::uniform_layout_entry(native::WGPUShaderStage_Compute);
            let bind_group_layout = common::create_bind_group_layout(test.device(), &[entry]);
            execute_after_destroy(&test, await_lost, || {
                let layout = common::create_pipeline_layout(test.device(), &[bind_group_layout]);
                yawgpu::wgpuPipelineLayoutRelease(layout);
            });
            yawgpu::wgpuBindGroupLayoutRelease(bind_group_layout);
        }
    }
}

#[test]
fn create_shader_module() {
    for source in [
        "@compute @workgroup_size(1) fn main() {}",
        "@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }",
        "@fragment fn main() {}",
    ] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                execute_after_destroy(&test, await_lost, || {
                    let module = common::create_wgsl_module(test.device(), source);
                    yawgpu::wgpuShaderModuleRelease(module);
                });
            }
        }
    }
}

#[test]
fn create_compute_pipeline() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let module = create_shader_module_for_stage(test.device(), ShaderStage::Compute);
            execute_after_destroy(&test, await_lost, || {
                let pipeline = common::create_compute_pipeline(test.device(), module, "main");
                yawgpu::wgpuComputePipelineRelease(pipeline);
            });
            yawgpu::wgpuShaderModuleRelease(module);
        }
    }
}

#[test]
fn create_render_pipeline() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let vertex = create_shader_module_for_stage(test.device(), ShaderStage::Vertex);
            let fragment = create_shader_module_for_stage(test.device(), ShaderStage::Fragment);
            execute_after_destroy(&test, await_lost, || {
                let pipeline =
                    common::create_render_pipeline(test.device(), vertex, fragment, "main");
                yawgpu::wgpuRenderPipelineRelease(pipeline);
            });
            yawgpu::wgpuShaderModuleRelease(fragment);
            yawgpu::wgpuShaderModuleRelease(vertex);
        }
    }
}

#[test]
#[ignore = "core currently resolves createComputePipelineAsync with ValidationError; CTS expects device-lost async pipeline creation to complete without validation error"]
fn create_compute_pipeline_async() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let module = create_shader_module_for_stage(test.device(), ShaderStage::Compute);
            let descriptor = common::compute_pipeline_descriptor(module, "main");
            execute_after_destroy(&test, await_lost, || {
                let mut state = ComputePipelineAsyncState::default();
                let future =
                    common::create_compute_pipeline_async(test.device(), &descriptor, &mut state);
                wait(test.instance(), future);
                assert_eq!(state.calls, 1);
                assert_eq!(state.status, native::WGPUCreatePipelineAsyncStatus_Success);
                if !state.pipeline.is_null() {
                    yawgpu::wgpuComputePipelineRelease(state.pipeline);
                }
            });
            yawgpu::wgpuShaderModuleRelease(module);
        }
    }
}

#[test]
#[ignore = "core currently resolves createRenderPipelineAsync with ValidationError; CTS expects device-lost async pipeline creation to complete without validation error"]
fn create_render_pipeline_async() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let vertex = create_shader_module_for_stage(test.device(), ShaderStage::Vertex);
            let fragment = create_shader_module_for_stage(test.device(), ShaderStage::Fragment);
            let target = crate::common::color_target();
            let fragment_state = common::fragment_state(fragment, "main", &target);
            let descriptor = common::render_pipeline_descriptor(vertex, &fragment_state);
            execute_after_destroy(&test, await_lost, || {
                let mut state = RenderPipelineAsyncState::default();
                let future =
                    common::create_render_pipeline_async(test.device(), &descriptor, &mut state);
                wait(test.instance(), future);
                assert_eq!(state.calls, 1);
                assert_eq!(state.status, native::WGPUCreatePipelineAsyncStatus_Success);
                if !state.pipeline.is_null() {
                    yawgpu::wgpuRenderPipelineRelease(state.pipeline);
                }
            });
            yawgpu::wgpuShaderModuleRelease(fragment);
            yawgpu::wgpuShaderModuleRelease(vertex);
        }
    }
}

#[test]
fn create_command_encoder() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            execute_after_destroy(&test, await_lost, || {
                let encoder = common::create_command_encoder(test.device());
                yawgpu::wgpuCommandEncoderRelease(encoder);
            });
        }
    }
}

#[test]
fn create_render_bundle_encoder() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            execute_after_destroy(&test, await_lost, || {
                let encoder = common::create_render_bundle_encoder(test.device());
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
            });
        }
    }
}

#[test]
fn create_query_set() {
    for query_type in [
        native::WGPUQueryType_Occlusion,
        native::WGPUQueryType_Timestamp,
    ] {
        for await_lost in [false, true] {
            let test = if query_type == native::WGPUQueryType_Timestamp {
                ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery])
            } else {
                ValidationTest::new()
            };
            unsafe {
                execute_after_destroy(&test, await_lost, || {
                    let query_set = common::create_query_set(test.device(), query_type, 4);
                    yawgpu::wgpuQuerySetRelease(query_set);
                });
            }
        }
    }
}

#[test]
fn command_copy_buffer_to_buffer() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let source = common::create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopySrc,
                    false,
                );
                let destination = common::create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopyDst,
                    false,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
                        encoder,
                        source,
                        0,
                        destination,
                        0,
                        16,
                    );
                });
                yawgpu::wgpuBufferRelease(destination);
                yawgpu::wgpuBufferRelease(source);
            }
        }
    }
}

#[test]
fn command_copy_buffer_to_texture() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let buffer = common::create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_CopySrc,
                    false,
                );
                let texture = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopyDst,
                    1,
                    1,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    copy_buffer_to_texture(encoder, buffer, texture);
                });
                yawgpu::wgpuTextureRelease(texture);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn command_copy_texture_to_buffer() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let texture = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopySrc,
                    1,
                    1,
                );
                let buffer = common::create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_CopyDst,
                    false,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    copy_texture_to_buffer(encoder, texture, buffer);
                });
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuTextureRelease(texture);
            }
        }
    }
}

#[test]
fn command_copy_texture_to_texture() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let source = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopySrc,
                    1,
                    1,
                );
                let destination = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopyDst,
                    1,
                    1,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    copy_texture_to_texture(encoder, source, destination);
                });
                yawgpu::wgpuTextureRelease(destination);
                yawgpu::wgpuTextureRelease(source);
            }
        }
    }
}

#[test]
fn command_clear_buffer() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let buffer = common::create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_CopyDst,
                    false,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 16);
                });
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn command_resolve_query_set() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let query_set =
                    common::create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
                let destination = common::create_buffer(
                    test.device(),
                    16,
                    native::WGPUBufferUsage_QueryResolve,
                    false,
                );
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    yawgpu::wgpuCommandEncoderResolveQuerySet(
                        encoder,
                        query_set,
                        0,
                        1,
                        destination,
                        0,
                    );
                });
                yawgpu::wgpuBufferRelease(destination);
                yawgpu::wgpuQuerySetRelease(query_set);
            }
        }
    }
}

#[test]
fn command_compute_pass_dispatch() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let module = create_shader_module_for_stage(test.device(), ShaderStage::Compute);
                let pipeline = common::create_compute_pipeline(test.device(), module, "main");
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    let pass =
                        yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
                    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
                    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
                    yawgpu::wgpuComputePassEncoderEnd(pass);
                    yawgpu::wgpuComputePassEncoderRelease(pass);
                });
                yawgpu::wgpuComputePipelineRelease(pipeline);
                yawgpu::wgpuShaderModuleRelease(module);
            }
        }
    }
}

#[test]
fn command_render_pass_draw() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let vertex = create_shader_module_for_stage(test.device(), ShaderStage::Vertex);
                let fragment = create_shader_module_for_stage(test.device(), ShaderStage::Fragment);
                let pipeline =
                    common::create_render_pipeline(test.device(), vertex, fragment, "main");
                let target = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_RenderAttachment,
                    4,
                    4,
                );
                let view = common::create_texture_view(target);
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    encode_render_pass_draw(encoder, view, pipeline);
                });
                yawgpu::wgpuTextureViewRelease(view);
                yawgpu::wgpuTextureRelease(target);
                yawgpu::wgpuRenderPipelineRelease(pipeline);
                yawgpu::wgpuShaderModuleRelease(fragment);
                yawgpu::wgpuShaderModuleRelease(vertex);
            }
        }
    }
}

#[test]
fn command_render_pass_render_bundle() {
    for stage in [CommandStage::Finish, CommandStage::Submit] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let vertex = create_shader_module_for_stage(test.device(), ShaderStage::Vertex);
                let fragment = create_shader_module_for_stage(test.device(), ShaderStage::Fragment);
                let pipeline =
                    common::create_render_pipeline(test.device(), vertex, fragment, "main");
                let bundle_encoder = common::create_render_bundle_encoder(test.device());
                yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline);
                yawgpu::wgpuRenderBundleEncoderDraw(bundle_encoder, 0, 1, 0, 0);
                let bundle =
                    yawgpu::wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null());
                assert!(!bundle.is_null());
                let target = common::create_texture(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_RenderAttachment,
                    4,
                    4,
                );
                let view = common::create_texture_view(target);
                execute_commands_after_destroy(&test, stage, await_lost, |encoder| {
                    encode_render_pass_bundle(encoder, view, bundle);
                });
                yawgpu::wgpuTextureViewRelease(view);
                yawgpu::wgpuTextureRelease(target);
                yawgpu::wgpuRenderBundleRelease(bundle);
                yawgpu::wgpuRenderBundleEncoderRelease(bundle_encoder);
                yawgpu::wgpuRenderPipelineRelease(pipeline);
                yawgpu::wgpuShaderModuleRelease(fragment);
                yawgpu::wgpuShaderModuleRelease(vertex);
            }
        }
    }
}

#[test]
fn queue_write_buffer() {
    for size in [4_usize, 8, 16] {
        for await_lost in [false, true] {
            let test = ValidationTest::new();
            unsafe {
                let buffer = common::create_buffer(
                    test.device(),
                    size as u64,
                    native::WGPUBufferUsage_CopyDst,
                    false,
                );
                let queue = yawgpu::wgpuDeviceGetQueue(test.device());
                let data = vec![0_u8; size];
                execute_after_destroy(&test, await_lost, || {
                    yawgpu::wgpuQueueWriteBuffer(
                        queue,
                        buffer,
                        0,
                        data.as_ptr().cast(),
                        data.len(),
                    );
                });
                yawgpu::wgpuQueueRelease(queue);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
    }
}

#[test]
fn queue_write_texture_2d_uncompressed_format() {
    for await_lost in [false, true] {
        let test = ValidationTest::new();
        unsafe {
            let texture = common::create_texture(
                test.device(),
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_CopyDst,
                1,
                1,
            );
            let queue = yawgpu::wgpuDeviceGetQueue(test.device());
            let data = [0_u8; 4];
            execute_after_destroy(&test, await_lost, || {
                common::write_texture(queue, texture, &data, 1, 1, 4);
            });
            yawgpu::wgpuQueueRelease(queue);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression features; CTS compressed-format device-lost subcases require a compressed texture format"]
fn queue_write_texture_2d_compressed_format() {}

#[derive(Clone, Copy)]
enum ShaderStage {
    Compute,
    Vertex,
    Fragment,
}

unsafe fn create_shader_module_for_stage(
    device: native::WGPUDevice,
    stage: ShaderStage,
) -> native::WGPUShaderModule {
    let source = match stage {
        ShaderStage::Compute => "@compute @workgroup_size(1) fn main() {}",
        ShaderStage::Vertex => {
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }"
        }
        ShaderStage::Fragment => "@fragment fn main() {}",
    };
    unsafe { common::create_wgsl_module(device, source) }
}

unsafe fn execute_after_destroy<F>(test: &ValidationTest, await_lost: bool, mut action: F)
where
    F: FnMut(),
{
    test.expect_no_validation_error(&mut action);
    let lost = unsafe { yawgpu::wgpuDeviceGetLostFuture(test.device()) };
    unsafe { yawgpu::wgpuDeviceDestroy(test.device()) };
    if await_lost {
        unsafe { wait(test.instance(), lost) };
    }
    test.expect_no_validation_error(action);
}

unsafe fn execute_commands_after_destroy<F>(
    test: &ValidationTest,
    stage: CommandStage,
    await_lost: bool,
    mut encode: F,
) where
    F: FnMut(native::WGPUCommandEncoder),
{
    let queue = unsafe { yawgpu::wgpuDeviceGetQueue(test.device()) };

    test.expect_no_validation_error(|| unsafe {
        let encoder = common::create_command_encoder(test.device());
        encode(encoder);
        let command_buffer = common::finish_command_encoder(encoder);
        if matches!(stage, CommandStage::Submit) {
            yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        }
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    });

    let encoder = unsafe { common::create_command_encoder(test.device()) };
    encode(encoder);
    let command_buffer = if matches!(stage, CommandStage::Submit) {
        let command_buffer = unsafe { common::finish_command_encoder(encoder) };
        Some(command_buffer)
    } else {
        None
    };

    let lost = unsafe { yawgpu::wgpuDeviceGetLostFuture(test.device()) };
    unsafe { yawgpu::wgpuDeviceDestroy(test.device()) };
    if await_lost {
        unsafe { wait(test.instance(), lost) };
    }

    test.expect_no_validation_error(|| unsafe {
        match stage {
            CommandStage::Finish => {
                let command_buffer = common::finish_command_encoder(encoder);
                yawgpu::wgpuCommandBufferRelease(command_buffer);
            }
            CommandStage::Submit => {
                let command_buffer = command_buffer.expect("submit stage has a command buffer");
                yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
                yawgpu::wgpuCommandBufferRelease(command_buffer);
            }
        }
    });
    unsafe {
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuQueueRelease(queue);
    }
}

unsafe fn copy_buffer_to_texture(
    encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    texture: native::WGPUTexture,
) {
    let source = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: 256,
            rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
        },
        buffer,
    };
    let destination = texture_copy(texture);
    let size = copy_size();
    unsafe { yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &size) };
}

unsafe fn copy_texture_to_buffer(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
) {
    let source = texture_copy(texture);
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: 256,
            rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
        },
        buffer,
    };
    let size = copy_size();
    unsafe { yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size) };
}

unsafe fn copy_texture_to_texture(
    encoder: native::WGPUCommandEncoder,
    source_texture: native::WGPUTexture,
    destination_texture: native::WGPUTexture,
) {
    let source = texture_copy(source_texture);
    let destination = texture_copy(destination_texture);
    let size = copy_size();
    unsafe {
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &size);
    }
}

fn texture_copy(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

fn copy_size() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    }
}

unsafe fn encode_render_pass_draw(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
    pipeline: native::WGPURenderPipeline,
) {
    let attachment = common::color_attachment(view);
    let descriptor = common::render_pass_descriptor(&attachment);
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor) };
    assert!(!pass.is_null());
    unsafe {
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 0, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
    }
}

unsafe fn encode_render_pass_bundle(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
    bundle: native::WGPURenderBundle,
) {
    let attachment = common::color_attachment(view);
    let descriptor = common::render_pass_descriptor(&attachment);
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor) };
    assert!(!pass.is_null());
    unsafe {
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
    }
}
