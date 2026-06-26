//! Tint shader frontend skeleton for the feature-selected frontend facade.
#![allow(dead_code)]

use crate::shader::CompilationMessage;
pub(crate) use crate::shader_types::*;

const NOT_IMPLEMENTED: &str = "shader_tint: not yet implemented (P2b/P2c)";

/// Stores reflected shader module data used by validation and backend submission.
#[derive(Debug)]
pub struct ReflectedModule {
    /// Tint program.
    pub program: yawgpu_tint::Program,
    /// Non-fatal compilation warnings.
    pub(crate) warnings: Vec<CompilationMessage>,
}

// SAFETY: Phase 2a does not call into the Tint program after parsing; the handle
// is stored only to preserve the future frontend shape. Later Tint slices must
// revisit this when reflection and code generation start using the handle.
unsafe impl Send for ReflectedModule {}

// SAFETY: See the `Send` impl above.
unsafe impl Sync for ReflectedModule {}

/// Returns parse and validate wgsl.
pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ReflectedModule, String> {
    parse_and_validate_wgsl_gated(src, true)
}

/// Returns parse and validate wgsl using the supplied `shader-f16` gate.
pub(crate) fn parse_and_validate_wgsl_gated(
    src: &str,
    shader_f16: bool,
) -> Result<ReflectedModule, String> {
    Ok(ReflectedModule {
        program: yawgpu_tint::Program::parse(src, shader_f16)?,
        warnings: Vec::new(),
    })
}

impl ReflectedModule {
    /// Generates spirv for the validated shader module.
    pub(crate) fn generate_spirv(
        &self,
        _entry_name: &str,
        _stage: ShaderStage,
        _pipeline_constants: &PipelineConstants,
        _unchecked_buffer_bounds: bool,
    ) -> Result<Vec<u32>, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates GLSL ES for the validated shader module.
    #[cfg(feature = "gles")]
    pub(crate) fn generate_glsl(
        &self,
        _entry_name: &str,
        _stage: ShaderStage,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedGlsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates msl for the validated shader module.
    pub(crate) fn generate_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates render vertex MSL for a validated shader module.
    pub(crate) fn generate_render_vertex_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _vertex_buffers: &[MslVertexBufferBinding],
        _force_point_size: bool,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates render fragment MSL for a validated shader module.
    pub(crate) fn generate_render_fragment_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _pipeline_constants: &PipelineConstants,
        _sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates render msl for the validated shader module.
    pub(crate) fn generate_render_msl(
        &self,
        _vertex_entry_name: &str,
        _fragment_entry_name: Option<&str>,
        _binding_map: &MslBindingMap,
        _vertex_buffers: &[MslVertexBufferBinding],
        _force_point_size: bool,
    ) -> Result<GeneratedRenderMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Returns entry points reflected by the validated shader module.
    pub(crate) fn entry_points(&self) -> Vec<ReflectedEntryPoint> {
        Vec::new()
    }

    /// Returns compute workgroup size reflected by the validated shader module.
    pub(crate) fn compute_workgroup_size(
        &self,
        _entry_point: &str,
    ) -> Result<Option<ReflectedWorkgroupSize>, String> {
        Ok(None)
    }

    /// Returns compute workgroup size after resolving pipeline constants.
    pub(crate) fn resolved_compute_workgroup_size(
        &self,
        _entry_point: &str,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<ReflectedWorkgroupSize, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Returns entry point io reflected by the validated shader module.
    pub(crate) fn entry_point_io(&self) -> Vec<ReflectedEntryPointIo> {
        Vec::new()
    }

    /// Returns resource bindings reflected by the validated shader module.
    pub(crate) fn resource_bindings(&self) -> Vec<ReflectedResourceBinding> {
        Vec::new()
    }

    /// Returns resource bindings for entry reflected by the validated shader module.
    pub(crate) fn resource_bindings_for_entry(
        &self,
        _entry_point: &str,
    ) -> Result<Vec<ReflectedResourceBinding>, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Returns storage buffer bindings that populate MSL `_mslBufferSizes`.
    pub(crate) fn msl_buffer_size_bindings_for_entry(
        &self,
        _entry_point: &str,
    ) -> Result<Vec<MslBufferSizeBinding>, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Returns fragment builtins reflected by the validated shader module.
    pub(crate) fn fragment_builtins(&self) -> Vec<ReflectedFragmentBuiltins> {
        Vec::new()
    }

    /// Returns overrides reflected by the validated shader module.
    pub(crate) fn overrides(&self) -> Vec<ReflectedOverride> {
        Vec::new()
    }
}
