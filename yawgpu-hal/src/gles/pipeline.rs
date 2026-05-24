use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::map_vertex_format;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalDescriptorBinding, HalError, HalPrimitiveTopology, HalRenderPipelineDescriptor,
    HalTextureFormat, HalVertexBufferLayout,
};

struct GlesComputePipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    workgroup_size: (u32, u32, u32),
    bindings: Vec<HalDescriptorBinding>,
}

impl Drop for GlesComputePipelineInner {
    fn drop(&mut self) {
        if let Ok(program) = self.program.as_ref() {
            let program = *program;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_program(program);
            });
        }
    }
}

/// Stores GLES compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesComputePipeline {
    inner: Arc<GlesComputePipelineInner>,
}

// SAFETY: `GlesComputePipeline` owns only GL object names and uses the owning
// device inner to serialize all GL access.
unsafe impl Send for GlesComputePipeline {}
// SAFETY: See the `Send` impl; shared access is synchronized by the device.
unsafe impl Sync for GlesComputePipeline {}

impl std::fmt::Debug for GlesComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesComputePipeline")
            .field("workgroup_size", &self.inner.workgroup_size)
            .field("bindings", &self.inner.bindings)
            .finish()
    }
}

impl GlesComputePipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        source: String,
        workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<Self, HalError> {
        let program = build_compute_program(&device, &source)?;
        Ok(Self {
            inner: Arc::new(GlesComputePipelineInner {
                device,
                program: Ok(program),
                workgroup_size,
                bindings: bindings.to_vec(),
            }),
        })
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Program, HalError> {
        self.inner
            .program
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }

    #[must_use]
    pub(super) fn bindings(&self) -> &[HalDescriptorBinding] {
        &self.inner.bindings
    }
}

struct GlesRenderPipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    vertex_buffers: Vec<HalVertexBufferLayout>,
    primitive_topology: HalPrimitiveTopology,
    bindings: Vec<HalDescriptorBinding>,
    first_instance_location: Option<glow::UniformLocation>,
}

impl Drop for GlesRenderPipelineInner {
    fn drop(&mut self) {
        if let Ok(program) = self.program.as_ref() {
            let program = *program;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_program(program);
            });
        }
    }
}

/// Stores GLES render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesRenderPipeline {
    inner: Arc<GlesRenderPipelineInner>,
}

// SAFETY: `GlesRenderPipeline` owns only GL object names and uses the owning
// device inner to serialize all GL access.
unsafe impl Send for GlesRenderPipeline {}
// SAFETY: See the `Send` impl; shared access is synchronized by the device.
unsafe impl Sync for GlesRenderPipeline {}

impl std::fmt::Debug for GlesRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesRenderPipeline")
            .field("vertex_buffers", &self.inner.vertex_buffers)
            .field("primitive_topology", &self.inner.primitive_topology)
            .field("bindings", &self.inner.bindings)
            .finish()
    }
}

impl GlesRenderPipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        vertex_source: String,
        fragment_source: String,
        descriptor: HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<Self, HalError> {
        validate_render_pipeline_descriptor(&descriptor)?;
        let (program, first_instance_location) =
            build_render_program(&device, &vertex_source, &fragment_source)?;
        Ok(Self {
            inner: Arc::new(GlesRenderPipelineInner {
                device,
                program: Ok(program),
                vertex_buffers: descriptor.vertex_buffers,
                primitive_topology: descriptor.primitive_topology,
                bindings: bindings.to_vec(),
                first_instance_location,
            }),
        })
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Program, HalError> {
        self.inner
            .program
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }

    #[must_use]
    pub(super) fn vertex_buffers(&self) -> &[HalVertexBufferLayout] {
        &self.inner.vertex_buffers
    }

    #[must_use]
    pub(super) fn primitive_topology(&self) -> HalPrimitiveTopology {
        self.inner.primitive_topology
    }

    #[must_use]
    pub(super) fn bindings(&self) -> &[HalDescriptorBinding] {
        &self.inner.bindings
    }

    #[must_use]
    pub(super) fn first_instance_location(&self) -> Option<&glow::UniformLocation> {
        self.inner.first_instance_location.as_ref()
    }
}

fn build_compute_program(
    device: &Arc<GlesDeviceInner>,
    source: &str,
) -> Result<glow::Program, HalError> {
    device
        .with_current_context(|gl| unsafe {
            let shader = gl.create_shader(glow::COMPUTE_SHADER).map_err(|_| {
                HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: "glCreateShader failed".to_owned(),
                }
            })?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES compute shader compilation failed: {log}"),
                });
            }

            let program = match gl.create_program() {
                Ok(program) => program,
                Err(_) => {
                    gl.delete_shader(shader);
                    return Err(HalError::ShaderCompilationFailed {
                        backend: BACKEND,
                        message: "glCreateProgram failed".to_owned(),
                    });
                }
            };
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES compute program link failed: {log}"),
                });
            }
            Ok(program)
        })
        .and_then(|result| result)
}

fn validate_render_pipeline_descriptor(
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<(), HalError> {
    if descriptor.color_formats.len() != 1
        || descriptor.color_formats[0] != HalTextureFormat::Rgba8Unorm
    {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES P15.5 render pipeline supports exactly one RGBA8Unorm color target",
        });
    }
    if descriptor.depth_stencil.is_some() {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES P15.5 render pipeline does not support depth/stencil",
        });
    }
    for layout in &descriptor.vertex_buffers {
        for attribute in &layout.attributes {
            map_vertex_format(attribute.format)?;
        }
    }
    Ok(())
}

fn build_render_program(
    device: &Arc<GlesDeviceInner>,
    vertex_source: &str,
    fragment_source: &str,
) -> Result<(glow::Program, Option<glow::UniformLocation>), HalError> {
    device
        .with_current_context(|gl| unsafe {
            let vertex = compile_shader(gl, glow::VERTEX_SHADER, vertex_source, "vertex")?;
            let fragment =
                match compile_shader(gl, glow::FRAGMENT_SHADER, fragment_source, "fragment") {
                    Ok(fragment) => fragment,
                    Err(error) => {
                        gl.delete_shader(vertex);
                        return Err(error);
                    }
                };
            let program = match gl.create_program() {
                Ok(program) => program,
                Err(_) => {
                    gl.delete_shader(vertex);
                    gl.delete_shader(fragment);
                    return Err(HalError::ShaderCompilationFailed {
                        backend: BACKEND,
                        message: "glCreateProgram failed".to_owned(),
                    });
                }
            };
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            gl.detach_shader(program, vertex);
            gl.detach_shader(program, fragment);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES render program link failed: {log}"),
                });
            }
            let first_instance_location =
                gl.get_uniform_location(program, "naga_vs_first_instance");
            Ok((program, first_instance_location))
        })
        .and_then(|result| result)
}

fn compile_shader(
    gl: &glow::Context,
    stage: u32,
    source: &str,
    label: &'static str,
) -> Result<glow::Shader, HalError> {
    unsafe {
        let shader = gl
            .create_shader(stage)
            .map_err(|_| HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: "glCreateShader failed".to_owned(),
            })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: format!("GLES {label} shader compilation failed: {log}"),
            });
        }
        Ok(shader)
    }
}
