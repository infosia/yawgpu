use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::{rebuild_hal_error, BACKEND};
use crate::{HalDescriptorBinding, HalError};

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

/// Stores GLES render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesRenderPipeline;

impl std::fmt::Debug for GlesRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesRenderPipeline").finish()
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
