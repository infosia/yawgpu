use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::{is_color_renderable_with, map_vertex_format, GlesColorRenderCaps};
use super::texture::validate_sample_count;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalColorTargetState, HalCombinedSampler, HalCullMode, HalDepthStencilState,
    HalDescriptorBinding, HalError, HalFrontFace, HalGlesBindingRemap, HalPrimitiveTopology,
    HalRenderPipelineDescriptor, HalTextureMetadataSlot, HalVertexBufferLayout,
};

/// The GLSL uniform name Tint's GLSL writer emits for the first element of
/// its "immediate data" emulation array when `first_instance_offset` is
/// requested (`yawgpu-core`'s `tint_bindings_for_glsl`/`generate_glsl`
/// request offset `0`, and GLES has no other internal immediate sharing the
/// struct, so element `0` is always `tint_first_instance`). See
/// `tint_shim.h`'s `yawgpu_tint_generate_glsl` docs and the
/// `generate_glsl_first_instance_offset_only_applied_when_requested` pin in
/// `yawgpu-tint` for the exact GLSL text this name is queried against (F2,
/// specs/tracking/tint-integration-refactor.md slice R6).
const FIRST_INSTANCE_UNIFORM_NAME: &str = "tint_immediates[0]";

#[derive(Debug, Clone)]
pub(super) struct GlesResolvedCombinedSampler {
    pub(super) unit: u32,
    pub(super) uniform_name: String,
    pub(super) texture_group: u32,
    pub(super) texture_binding: u32,
    pub(super) sampler_group: u32,
    pub(super) sampler_binding: u32,
    pub(super) uses_placeholder_sampler: bool,
}

pub(super) struct GlesPipelineResourceBindings {
    pub(super) combined_samplers: Vec<HalCombinedSampler>,
    pub(super) texture_metadata_slots: Vec<HalTextureMetadataSlot>,
    pub(super) binding_remaps: Vec<HalGlesBindingRemap>,
    pub(super) texture_metadata_ubo_binding: Option<u32>,
}

struct GlesComputePipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    workgroup_size: (u32, u32, u32),
    bindings: Vec<HalDescriptorBinding>,
    combined_samplers: Vec<GlesResolvedCombinedSampler>,
    texture_metadata_slots: Vec<HalTextureMetadataSlot>,
    binding_remaps: Vec<HalGlesBindingRemap>,
    texture_metadata_ubo_binding: Option<u32>,
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
            .field("combined_samplers", &self.inner.combined_samplers)
            .finish()
    }
}

impl GlesComputePipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        source: String,
        workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
        resource_bindings: GlesPipelineResourceBindings,
    ) -> Result<Self, HalError> {
        let program = build_compute_program(&device, &source)?;
        let combined_samplers =
            resolve_combined_samplers(&device, program, &resource_bindings.combined_samplers)?;
        Ok(Self {
            inner: Arc::new(GlesComputePipelineInner {
                device,
                program: Ok(program),
                workgroup_size,
                bindings: bindings.to_vec(),
                combined_samplers,
                texture_metadata_slots: resource_bindings.texture_metadata_slots,
                binding_remaps: resource_bindings.binding_remaps,
                texture_metadata_ubo_binding: resource_bindings.texture_metadata_ubo_binding,
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

    #[must_use]
    pub(super) fn combined_samplers(&self) -> &[GlesResolvedCombinedSampler] {
        &self.inner.combined_samplers
    }

    #[must_use]
    pub(super) fn binding_remaps(&self) -> &[HalGlesBindingRemap] {
        &self.inner.binding_remaps
    }

    #[must_use]
    pub(super) fn texture_metadata_slots(&self) -> &[HalTextureMetadataSlot] {
        &self.inner.texture_metadata_slots
    }

    #[must_use]
    pub(super) fn texture_metadata_ubo_binding(&self) -> Option<u32> {
        self.inner.texture_metadata_ubo_binding
    }
}

struct GlesRenderPipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    vertex_buffers: Vec<HalVertexBufferLayout>,
    color_target: Option<HalColorTargetState>,
    depth_stencil: Option<HalDepthStencilState>,
    primitive_topology: HalPrimitiveTopology,
    front_face: HalFrontFace,
    cull_mode: HalCullMode,
    sample_mask: u32,
    alpha_to_coverage_enabled: bool,
    bindings: Vec<HalDescriptorBinding>,
    combined_samplers: Vec<GlesResolvedCombinedSampler>,
    texture_metadata_slots: Vec<HalTextureMetadataSlot>,
    binding_remaps: Vec<HalGlesBindingRemap>,
    texture_metadata_ubo_binding: Option<u32>,
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
            .field("combined_samplers", &self.inner.combined_samplers)
            .finish()
    }
}

impl GlesRenderPipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        vertex_source: String,
        fragment_source: Option<String>,
        descriptor: HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
        resource_bindings: GlesPipelineResourceBindings,
    ) -> Result<Self, HalError> {
        validate_render_pipeline_descriptor(
            &descriptor,
            device.color_render_caps(),
            device.max_samples(),
        )?;
        let (program, first_instance_location) =
            build_render_program(&device, &vertex_source, fragment_source.as_deref())?;
        let combined_samplers =
            resolve_combined_samplers(&device, program, &resource_bindings.combined_samplers)?;
        Ok(Self {
            inner: Arc::new(GlesRenderPipelineInner {
                device,
                program: Ok(program),
                vertex_buffers: descriptor.vertex_buffers,
                // T-G8 (MRT): validation above guarantees every color target
                // shares one write mask and one blend state, so the first
                // `Some` target fully describes the global GL color state
                // applied at draw time (GLES 3.1 has no indexed variants).
                color_target: descriptor.color_targets.iter().copied().flatten().next(),
                depth_stencil: descriptor.depth_stencil,
                primitive_topology: descriptor.primitive_topology,
                front_face: descriptor.front_face,
                cull_mode: descriptor.cull_mode,
                sample_mask: descriptor.sample_mask,
                alpha_to_coverage_enabled: descriptor.alpha_to_coverage_enabled,
                bindings: bindings.to_vec(),
                combined_samplers,
                texture_metadata_slots: resource_bindings.texture_metadata_slots,
                binding_remaps: resource_bindings.binding_remaps,
                texture_metadata_ubo_binding: resource_bindings.texture_metadata_ubo_binding,
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

    /// Returns the shared color-target state (first `Some` target).
    /// Descriptor validation rejects divergent per-target write masks and
    /// blend state on GLES 3.1, so this single state drives the global
    /// `glColorMask`/`glBlendFunc*` calls for every attachment.
    #[must_use]
    pub(super) fn color_target(&self) -> Option<HalColorTargetState> {
        self.inner.color_target
    }

    #[must_use]
    pub(super) fn depth_stencil(&self) -> Option<HalDepthStencilState> {
        self.inner.depth_stencil
    }

    #[must_use]
    pub(super) fn primitive_topology(&self) -> HalPrimitiveTopology {
        self.inner.primitive_topology
    }

    #[must_use]
    pub(super) fn front_face(&self) -> HalFrontFace {
        self.inner.front_face
    }

    #[must_use]
    pub(super) fn cull_mode(&self) -> HalCullMode {
        self.inner.cull_mode
    }

    #[must_use]
    pub(super) fn sample_mask(&self) -> u32 {
        self.inner.sample_mask
    }

    #[must_use]
    pub(super) fn alpha_to_coverage_enabled(&self) -> bool {
        self.inner.alpha_to_coverage_enabled
    }

    #[must_use]
    pub(super) fn bindings(&self) -> &[HalDescriptorBinding] {
        &self.inner.bindings
    }

    #[must_use]
    pub(super) fn first_instance_location(&self) -> Option<&glow::UniformLocation> {
        self.inner.first_instance_location.as_ref()
    }

    #[must_use]
    pub(super) fn combined_samplers(&self) -> &[GlesResolvedCombinedSampler] {
        &self.inner.combined_samplers
    }

    #[must_use]
    pub(super) fn binding_remaps(&self) -> &[HalGlesBindingRemap] {
        &self.inner.binding_remaps
    }

    #[must_use]
    pub(super) fn texture_metadata_slots(&self) -> &[HalTextureMetadataSlot] {
        &self.inner.texture_metadata_slots
    }

    #[must_use]
    pub(super) fn texture_metadata_ubo_binding(&self) -> Option<u32> {
        self.inner.texture_metadata_ubo_binding
    }
}

fn resolve_combined_samplers(
    device: &Arc<GlesDeviceInner>,
    program: glow::Program,
    combined_samplers: &[HalCombinedSampler],
) -> Result<Vec<GlesResolvedCombinedSampler>, HalError> {
    device
        .with_current_context(|gl| unsafe {
            let mut resolved = Vec::new();
            gl.use_program(Some(program));
            for combined in combined_samplers {
                let Some(location) = gl.get_uniform_location(program, &combined.glsl_uniform_name)
                else {
                    continue;
                };
                let unit =
                    u32::try_from(resolved.len()).map_err(|_| HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "GLES combined sampler unit index exceeds limit",
                    })?;
                let unit_i32 =
                    i32::try_from(unit).map_err(|_| HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "GLES combined sampler unit index exceeds limit",
                    })?;
                gl.uniform_1_i32(Some(&location), unit_i32);
                resolved.push(GlesResolvedCombinedSampler {
                    unit,
                    uniform_name: combined.glsl_uniform_name.clone(),
                    texture_group: combined.texture_group,
                    texture_binding: combined.texture_binding,
                    sampler_group: combined.sampler_group,
                    sampler_binding: combined.sampler_binding,
                    uses_placeholder_sampler: combined.uses_placeholder_sampler,
                });
            }
            gl.use_program(None);
            Ok(resolved)
        })
        .and_then(|result| result)
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
    color_render_caps: GlesColorRenderCaps,
    max_samples: i32,
) -> Result<(), HalError> {
    validate_sample_count(descriptor.sample_count, max_samples)?;
    if descriptor.unclipped_depth {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline does not support unclipped depth",
        });
    }
    // T-G8 (MRT): GLES 3.1 core has no indexed per-draw-buffer state --
    // `glColorMaski` / `glBlendFunci` / `glEnablei(GL_BLEND, i)` arrive only
    // with ES 3.2 / EXT_draw_buffers_indexed -- so write mask and blend state
    // apply globally to every draw buffer. Pipelines whose color targets all
    // share one write mask and one blend state map cleanly; divergent
    // per-target state is a Tier-2 HAL rejection (catalogued in
    // specs/blocks/67-gles-backend.md, "WebGPU x GLES mapping matrix").
    let first_target = descriptor.color_targets.iter().flatten().next();
    for target in descriptor.color_targets.iter().flatten() {
        // T-G12: float formats become color-renderable when the device
        // advertises `EXT_color_buffer_float` / `EXT_color_buffer_half_float`
        // (caps detected once at device creation).
        if !is_color_renderable_with(target.format, color_render_caps) {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message:
                    "GLES render pipeline color target format is not color-renderable on GLES 3.1",
            });
        }
        if target.blend.is_some_and(gles_blend_uses_dual_source) {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES render pipeline does not support dual-source blend factors",
            });
        }
        if let Some(first) = first_target {
            if target.write_mask != first.write_mask {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES 3.1 cannot apply per-target write masks",
                });
            }
            if target.blend != first.blend {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES 3.1 cannot apply per-target blend state",
                });
            }
        }
    }
    if !descriptor.color_targets.iter().any(Option::is_some) && descriptor.depth_stencil.is_none() {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline requires a color target or depth-stencil state",
        });
    }
    for layout in &descriptor.vertex_buffers {
        for attribute in &layout.attributes {
            map_vertex_format(attribute.format)?;
        }
    }
    Ok(())
}

fn gles_blend_uses_dual_source(blend: crate::HalBlendState) -> bool {
    gles_blend_component_uses_dual_source(blend.color)
        || gles_blend_component_uses_dual_source(blend.alpha)
}

fn gles_blend_component_uses_dual_source(component: crate::HalBlendComponent) -> bool {
    gles_blend_factor_is_dual_source(component.src_factor)
        || gles_blend_factor_is_dual_source(component.dst_factor)
}

fn gles_blend_factor_is_dual_source(factor: crate::HalBlendFactor) -> bool {
    matches!(
        factor,
        crate::HalBlendFactor::Src1
            | crate::HalBlendFactor::OneMinusSrc1
            | crate::HalBlendFactor::Src1Alpha
            | crate::HalBlendFactor::OneMinusSrc1Alpha
    )
}

/// Stub fragment shader linked into fragment-less render pipelines. Mesa
/// rejects fragment-less program links ("program lacks a fragment shader")
/// while ANGLE accepts them; the stub makes vertex-only
/// (depth/stencil-only) pipelines link on every driver.
const FRAGMENTLESS_STUB_FS: &str = "#version 310 es\nvoid main() {}\n";

fn build_render_program(
    device: &Arc<GlesDeviceInner>,
    vertex_source: &str,
    fragment_source: Option<&str>,
) -> Result<(glow::Program, Option<glow::UniformLocation>), HalError> {
    device
        .with_current_context(|gl| unsafe {
            let vertex = compile_shader(gl, glow::VERTEX_SHADER, vertex_source, "vertex")?;
            // Fragment-less pipelines still attach a stub fragment stage so
            // the program links on drivers (Mesa) that reject vertex-only
            // links; see `FRAGMENTLESS_STUB_FS`.
            let fragment = match compile_shader(
                gl,
                glow::FRAGMENT_SHADER,
                fragment_source.unwrap_or(FRAGMENTLESS_STUB_FS),
                "fragment",
            ) {
                Ok(fragment) => Some(fragment),
                Err(error) => {
                    gl.delete_shader(vertex);
                    return Err(error);
                }
            };
            let program = match gl.create_program() {
                Ok(program) => program,
                Err(_) => {
                    gl.delete_shader(vertex);
                    if let Some(fragment) = fragment {
                        gl.delete_shader(fragment);
                    }
                    return Err(HalError::ShaderCompilationFailed {
                        backend: BACKEND,
                        message: "glCreateProgram failed".to_owned(),
                    });
                }
            };
            gl.attach_shader(program, vertex);
            if let Some(fragment) = fragment {
                gl.attach_shader(program, fragment);
            }
            gl.link_program(program);
            gl.detach_shader(program, vertex);
            if let Some(fragment) = fragment {
                gl.detach_shader(program, fragment);
            }
            gl.delete_shader(vertex);
            if let Some(fragment) = fragment {
                gl.delete_shader(fragment);
            }
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES render program link failed: {log}"),
                });
            }
            // Tint's GLSL writer emits buffer blocks with an explicit
            // `layout(binding = N)` matching the WGSL @binding number
            // directly (see `yawgpu-core`'s `tint_bindings_for_glsl`, which
            // supplies an identity remap so this always holds) -- GLES 3.1
            // core honors that at link time, so no
            // `glUniformBlockBinding`/`glShaderStorageBlockBinding` runtime
            // remap is needed (F2, specs/tracking/tint-integration-refactor.md
            // slice R6; this replaced a naga-era `_block_N`-suffix name
            // parser that no longer matched Tint's block-naming scheme and
            // would have silently mis-bound buffers).
            //
            // First-instance offset: when the vertex stage reads
            // `@builtin(instance_index)`, `yawgpu-core` requests Tint's
            // `first_instance_offset`, which emits a single
            // `layout(location = 0) uniform uint tint_immediates[1]` array
            // (see `tint_shim.h`'s `yawgpu_tint_generate_glsl` docs). Query
            // its location by the GLSL array-element name; `queue.rs`
            // writes the WebGPU `firstInstance` draw parameter there before
            // each draw.
            let first_instance_location =
                gl.get_uniform_location(program, FIRST_INSTANCE_UNIFORM_NAME);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HalTextureFormat;
    const TEST_MAX_SAMPLES: i32 = 4;

    fn color_target() -> HalColorTargetState {
        HalColorTargetState {
            format: HalTextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: 0xf,
        }
    }

    fn render_descriptor(
        color_targets: Vec<Option<HalColorTargetState>>,
    ) -> HalRenderPipelineDescriptor {
        HalRenderPipelineDescriptor {
            sample_count: 1,
            sample_mask: u32::MAX,
            alpha_to_coverage_enabled: false,
            color_targets,
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            primitive_topology: crate::HalPrimitiveTopology::TriangleList,
            front_face: crate::HalFrontFace::Ccw,
            cull_mode: crate::HalCullMode::None,
            unclipped_depth: false,
            needs_frag_depth_range_push_constant: false,
            user_immediate_size: 0,
        }
    }

    #[test]
    fn validate_render_pipeline_descriptor_accepts_multiple_and_sparse_color_targets() {
        // T-G8 (MRT): multiple color targets and sparse (None) slots are
        // mappable via glDrawBuffers on GLES 3.1 and must validate.
        assert!(validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(color_target())]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .is_ok());
        assert!(validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(color_target()), Some(color_target())]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .is_ok());
        assert!(validate_render_pipeline_descriptor(
            &render_descriptor(vec![None, Some(color_target())]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .is_ok());
        assert!(validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(color_target()), None, Some(color_target())]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .is_ok());
    }

    #[test]
    fn validate_render_pipeline_descriptor_rejects_divergent_per_target_state() {
        // T-G8 (MRT): GLES 3.1 has no indexed glColorMaski/glBlendFunci, so
        // write mask and blend state must be identical across all color
        // targets; divergence is a Tier-2 HAL rejection.
        let mut masked = color_target();
        masked.write_mask = 0x7;
        let error = validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(color_target()), Some(masked)]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .expect_err("divergent per-target write masks must be rejected");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES 3.1 cannot apply per-target write masks",
            }
        ));

        let mut blended = color_target();
        blended.blend = Some(crate::HalBlendState {
            color: crate::HalBlendComponent {
                operation: crate::HalBlendOperation::Add,
                src_factor: crate::HalBlendFactor::One,
                dst_factor: crate::HalBlendFactor::Zero,
            },
            alpha: crate::HalBlendComponent {
                operation: crate::HalBlendOperation::Add,
                src_factor: crate::HalBlendFactor::One,
                dst_factor: crate::HalBlendFactor::Zero,
            },
        });
        let error = validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(color_target()), Some(blended)]),
            GlesColorRenderCaps::default(),
            TEST_MAX_SAMPLES,
        )
        .expect_err("divergent per-target blend state must be rejected");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES 3.1 cannot apply per-target blend state",
            }
        ));
    }

    #[test]
    fn validate_render_pipeline_descriptor_accepts_color_renderable_formats() {
        // T-G7: the color-target whitelist widened from RGBA8/BGRA8 to the
        // GLES 3.1 core color-renderable set, including integer formats.
        let accepted = [
            HalTextureFormat::R8Unorm,
            HalTextureFormat::Rg8Unorm,
            HalTextureFormat::Rgba8Unorm,
            HalTextureFormat::Rgba8UnormSrgb,
            HalTextureFormat::Bgra8Unorm,
            HalTextureFormat::Rgb10a2Unorm,
            HalTextureFormat::R32Uint,
            HalTextureFormat::Rgba16Sint,
            HalTextureFormat::Rgba32Uint,
            HalTextureFormat::Rgb10a2Uint,
        ];
        for format in accepted {
            let mut target = color_target();
            target.format = format;
            assert!(
                validate_render_pipeline_descriptor(
                    &render_descriptor(vec![Some(target)]),
                    GlesColorRenderCaps::default(),
                    TEST_MAX_SAMPLES,
                )
                .is_ok(),
                "{format:?} must be accepted as a GLES color target"
            );
        }

        let rejected = [
            HalTextureFormat::Rgba8Snorm,
            HalTextureFormat::R16Float,
            HalTextureFormat::Rgba32Float,
        ];
        for format in rejected {
            let mut target = color_target();
            target.format = format;
            let error = validate_render_pipeline_descriptor(
                &render_descriptor(vec![Some(target)]),
                GlesColorRenderCaps::default(),
                TEST_MAX_SAMPLES,
            )
            .expect_err("non-color-renderable format must be rejected");
            assert!(
                matches!(
                    error,
                    HalError::BufferOperationFailed {
                        backend: "gles",
                        message:
                            "GLES render pipeline color target format is not color-renderable on GLES 3.1",
                    }
                ),
                "{format:?}"
            );
        }
    }

    #[test]
    fn validate_render_pipeline_descriptor_gates_float_targets_on_extension_caps() {
        // T-G12: float color targets are accepted exactly when the device
        // caps say the corresponding extension is present.
        let float_caps = GlesColorRenderCaps {
            color_buffer_float: true,
            color_buffer_half_float: false,
        };
        for format in [
            HalTextureFormat::R16Float,
            HalTextureFormat::Rgba16Float,
            HalTextureFormat::R32Float,
            HalTextureFormat::Rgba32Float,
            HalTextureFormat::Rg11b10Ufloat,
        ] {
            let mut target = color_target();
            target.format = format;
            assert!(
                validate_render_pipeline_descriptor(
                    &render_descriptor(vec![Some(target)]),
                    float_caps,
                    TEST_MAX_SAMPLES,
                )
                .is_ok(),
                "{format:?} must be accepted with EXT_color_buffer_float"
            );
        }

        // EXT_color_buffer_half_float alone unlocks float16 targets only.
        let half_float_caps = GlesColorRenderCaps {
            color_buffer_float: false,
            color_buffer_half_float: true,
        };
        let mut half_float_target = color_target();
        half_float_target.format = HalTextureFormat::Rgba16Float;
        assert!(validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(half_float_target)]),
            half_float_caps,
            TEST_MAX_SAMPLES,
        )
        .is_ok());
        let mut float32_target = color_target();
        float32_target.format = HalTextureFormat::R32Float;
        let error = validate_render_pipeline_descriptor(
            &render_descriptor(vec![Some(float32_target)]),
            half_float_caps,
            TEST_MAX_SAMPLES,
        )
        .expect_err("float32 target must stay rejected without EXT_color_buffer_float");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message:
                    "GLES render pipeline color target format is not color-renderable on GLES 3.1",
            }
        ));
    }

    #[test]
    fn validate_render_pipeline_descriptor_rejects_sample_count_above_max_samples() {
        let mut descriptor = render_descriptor(vec![Some(color_target())]);
        descriptor.sample_count = 4;
        assert!(validate_render_pipeline_descriptor(
            &descriptor,
            GlesColorRenderCaps::default(),
            4,
        )
        .is_ok());
        let error =
            validate_render_pipeline_descriptor(&descriptor, GlesColorRenderCaps::default(), 1)
                .expect_err("sample count above GL_MAX_SAMPLES must be rejected");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "texture sample count exceeds GL_MAX_SAMPLES",
            }
        ));
    }

    /// F2 (specs/tracking/tint-integration-refactor.md, slice R6): pins the
    /// GLSL uniform name this HAL queries against the exact array-element
    /// syntax Tint's GLSL writer emits (see
    /// `yawgpu-tint`'s `generate_glsl_first_instance_offset_only_applied_when_requested`,
    /// which pins the shim side of this same contract: `layout(location = 0)
    /// uniform uint tint_immediates[1];`).
    #[test]
    fn first_instance_uniform_name_matches_tint_immediate_array_element_zero() {
        assert_eq!(FIRST_INSTANCE_UNIFORM_NAME, "tint_immediates[0]");
    }

    fn stencil_face_state() -> crate::HalStencilFaceState {
        crate::HalStencilFaceState {
            compare: crate::HalCompareFunction::Always,
            fail_op: crate::HalStencilOperation::Keep,
            depth_fail_op: crate::HalStencilOperation::Keep,
            pass_op: crate::HalStencilOperation::Keep,
        }
    }

    #[test]
    fn create_render_pipeline_without_fragment_stage_links() {
        // Regression test for T-G6 (Finding G-6,
        // specs/tracking/cts-gles-sweep-0705.md): Mesa (crocus/llvmpipe)
        // rejects fragment-less program links ("error: program lacks a
        // fragment shader"), so fragment-less (depth/stencil-only) render
        // pipelines must link a stub fragment stage. Pre-fix this failed at
        // link time on Mesa; ANGLE accepted fragment-less links, which is
        // why the gap only surfaced on this host.
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!(
                    "skipping GLES fragment-less pipeline test; backend unavailable: {error:?}"
                );
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES fragment-less pipeline test; no adapter available");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!(
                    "skipping GLES fragment-less pipeline test; device unavailable: {error:?}"
                );
                return;
            }
        };

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n"
                        .to_owned(),
                    fragment: None,
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                None,
                &HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: Vec::new(),
                    depth_stencil: Some(HalDepthStencilState {
                        format: HalTextureFormat::Depth24PlusStencil8,
                        depth_write_enabled: true,
                        depth_compare: crate::HalCompareFunction::Always,
                        stencil_front: stencil_face_state(),
                        stencil_back: stencil_face_state(),
                        stencil_read_mask: 0xff,
                        stencil_write_mask: 0xff,
                        depth_bias: 0,
                        depth_bias_slope_scale: 0.0,
                        depth_bias_clamp: 0.0,
                    }),
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: crate::HalFrontFace::Ccw,
                    cull_mode: crate::HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("fragment-less GLES render pipeline creation must link with the stub fragment stage");
        assert!(
            pipeline.raw_or_err().is_ok(),
            "fragment-less pipeline must hold a linked GL program"
        );
    }

    #[test]
    fn create_render_pipeline_accepts_integer_color_target_and_rejects_snorm() {
        // T-G7: an Rgba16Sint color target is core color-renderable on GLES
        // 3.1 and must create, while Rgba8Snorm is not color-renderable and
        // must still error.
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!(
                    "skipping GLES integer-color-target pipeline test; backend unavailable: {error:?}"
                );
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES integer-color-target pipeline test; no adapter available");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!(
                    "skipping GLES integer-color-target pipeline test; device unavailable: {error:?}"
                );
                return;
            }
        };

        let descriptor_for = |format: HalTextureFormat| HalRenderPipelineDescriptor {
            sample_count: 1,
            sample_mask: u32::MAX,
            alpha_to_coverage_enabled: false,
            color_targets: vec![Some(HalColorTargetState {
                format,
                blend: None,
                write_mask: 0xf,
            })],
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            primitive_topology: crate::HalPrimitiveTopology::TriangleList,
            front_face: crate::HalFrontFace::Ccw,
            cull_mode: crate::HalCullMode::None,
            unclipped_depth: false,
            needs_frag_depth_range_push_constant: false,
            user_immediate_size: 0,
        };
        let shader_source = || crate::HalShaderSource::GlslStages {
            vertex: "#version 310 es\n\
                     void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n"
                .to_owned(),
            fragment: Some(
                "#version 310 es\n\
                 precision highp int;\n\
                 layout(location = 0) out highp ivec4 frag_color;\n\
                 void main() { frag_color = ivec4(1); }\n"
                    .to_owned(),
            ),
            combined_samplers: Vec::new(),
            texture_metadata_slots: Vec::new(),
            binding_remaps: Vec::new(),
            texture_metadata_ubo_binding: None,
        };

        let pipeline = device
            .create_render_pipeline(
                shader_source(),
                "main",
                Some("main"),
                &descriptor_for(HalTextureFormat::Rgba16Sint),
                &[],
            )
            .expect("Rgba16Sint color target must be accepted on GLES 3.1");
        assert!(
            pipeline.raw_or_err().is_ok(),
            "integer-color-target pipeline must hold a linked GL program"
        );

        let error = device
            .create_render_pipeline(
                shader_source(),
                "main",
                Some("main"),
                &descriptor_for(HalTextureFormat::Rgba8Snorm),
                &[],
            )
            .expect_err("Rgba8Snorm color target must still be rejected on GLES");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message:
                    "GLES render pipeline color target format is not color-renderable on GLES 3.1",
            }
        ));
    }
}
