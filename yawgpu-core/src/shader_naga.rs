#![allow(dead_code)]
// P5.0 intentionally lands reflection helpers before pipeline creation uses
// them. Later Phase-5 slices consume these crate-private APIs.

use std::collections::BTreeMap;

/// Stores reflected shader module data used by validation and backend submission.
#[derive(Debug)]
pub struct ReflectedModule {
    /// Module.
    pub module: naga::Module,
    /// Info.
    pub info: naga::valid::ModuleInfo,
}

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MslBindingMap {
    /// Resources.
    pub resources: Vec<MslResourceBinding>,
}

/// Stores one MSL buffer-size entry binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslBufferSizeBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
}

/// Stores MSL resource binding metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslResourceBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Metal index.
    pub metal_index: u32,
    /// Kind.
    pub kind: MslResourceBindingKind,
}

/// Enumerates MSL resource binding kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslResourceBindingKind {
    /// Buffer variant.
    Buffer,
    /// Texture variant.
    Texture,
    /// Sampler variant.
    Sampler,
    /// External texture variant.
    ExternalTexture,
}

/// Stores generated shader source for generated MSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedMsl {
    /// Source.
    pub source: String,
    /// Entry point.
    pub entry_point: String,
    /// Reserved MSL buffer slot for `_mslBufferSizes`.
    pub buffer_sizes_slot: Option<u32>,
    /// Bindings whose byte lengths populate `_mslBufferSizes`.
    pub buffer_size_bindings: Vec<MslBufferSizeBinding>,
    /// Reserved fragment immediate slot for the frag-depth clamp range.
    pub frag_depth_clamp_slot: Option<u32>,
    /// Per-argument threadgroup memory allocation sizes (bytes, rounded up to a
    /// multiple of 16) for compute shaders that use `var<workgroup>` globals.
    ///
    /// naga's MSL backend emits each workgroup variable as an entry-point
    /// argument annotated with `[[threadgroup(N)]]`, where N is the 0-based
    /// declaration index among all workgroup globals used by that entry point.
    /// Metal requires the compute encoder to call
    /// `setThreadgroupMemoryLength:atIndex:` for each such slot before dispatch;
    /// without this the slots read as zeros.  The vec is empty for compute shaders
    /// that have no workgroup variables, and is also empty for render shaders
    /// (which cannot use workgroup memory through this path).
    pub workgroup_memory_sizes: Vec<u32>,
}

/// Stores generated shader source for generated GLSL.
#[cfg(feature = "gles")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedGlsl {
    /// Source.
    pub source: String,
    /// Entry point.
    pub entry_point: String,
}

/// Stores generated shader source for generated render MSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedRenderMsl {
    /// Source.
    pub source: String,
    /// Vertex entry point.
    pub vertex_entry_point: String,
    /// Optional fragment entry point.
    pub fragment_entry_point: Option<String>,
    /// Reserved vertex-stage MSL buffer slot for `_mslBufferSizes`.
    pub vertex_buffer_sizes_slot: Option<u32>,
    /// Vertex-stage bindings whose byte lengths populate `_mslBufferSizes`.
    pub vertex_buffer_size_bindings: Vec<MslBufferSizeBinding>,
    /// Reserved fragment-stage MSL buffer slot for `_mslBufferSizes`.
    pub fragment_buffer_sizes_slot: Option<u32>,
    /// Fragment-stage bindings whose byte lengths populate `_mslBufferSizes`.
    pub fragment_buffer_size_bindings: Vec<MslBufferSizeBinding>,
}

struct RenderMslStageOptions<'a> {
    vertex_buffer_mappings: Vec<naga::back::msl::VertexBufferMapping>,
    force_point_size: bool,
    subpass_color_slots: &'a [((u32, u32), u32)],
    pipeline_constants: &'a naga::back::PipelineConstants,
    sample_mask: u32,
}

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MslVertexBufferBinding {
    /// Slot.
    pub slot: u32,
    /// Metal index.
    pub metal_index: u32,
    /// Array stride.
    pub array_stride: u64,
    /// Step mode.
    pub step_mode: MslVertexStepMode,
    /// Attributes.
    pub attributes: Vec<MslVertexAttribute>,
}

/// Enumerates msl vertex step mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslVertexStepMode {
    /// Vertex variant.
    Vertex,
    /// Instance variant.
    Instance,
}

/// Stores attribute metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslVertexAttribute {
    /// Shader location.
    pub shader_location: u32,
    /// Offset.
    pub offset: u64,
    /// Format.
    pub format: MslVertexFormat,
}

/// Enumerates msl vertex format values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslVertexFormat {
    /// Uint8 variant.
    Uint8,
    /// Uint8x2 variant.
    Uint8x2,
    /// Uint8x4 variant.
    Uint8x4,
    /// Sint8 variant.
    Sint8,
    /// Sint8x2 variant.
    Sint8x2,
    /// Sint8x4 variant.
    Sint8x4,
    /// Unorm8 variant.
    Unorm8,
    /// Unorm8x2 variant.
    Unorm8x2,
    /// Unorm8x4 variant.
    Unorm8x4,
    /// Snorm8 variant.
    Snorm8,
    /// Snorm8x2 variant.
    Snorm8x2,
    /// Snorm8x4 variant.
    Snorm8x4,
    /// Uint16 variant.
    Uint16,
    /// Uint16x2 variant.
    Uint16x2,
    /// Uint16x4 variant.
    Uint16x4,
    /// Sint16 variant.
    Sint16,
    /// Sint16x2 variant.
    Sint16x2,
    /// Sint16x4 variant.
    Sint16x4,
    /// Unorm16 variant.
    Unorm16,
    /// Unorm16x2 variant.
    Unorm16x2,
    /// Unorm16x4 variant.
    Unorm16x4,
    /// Snorm16 variant.
    Snorm16,
    /// Snorm16x2 variant.
    Snorm16x2,
    /// Snorm16x4 variant.
    Snorm16x4,
    /// Float16 variant.
    Float16,
    /// Float16x2 variant.
    Float16x2,
    /// Float16x4 variant.
    Float16x4,
    /// Float32 variant.
    Float32,
    /// Float32x2 variant.
    Float32x2,
    /// Float32x3 variant.
    Float32x3,
    /// Float32x4 variant.
    Float32x4,
    /// Uint32 variant.
    Uint32,
    /// Uint32x2 variant.
    Uint32x2,
    /// Uint32x3 variant.
    Uint32x3,
    /// Uint32x4 variant.
    Uint32x4,
    /// Sint32 variant.
    Sint32,
    /// Sint32x2 variant.
    Sint32x2,
    /// Sint32x3 variant.
    Sint32x3,
    /// Sint32x4 variant.
    Sint32x4,
    /// Unorm10_10_10_2 variant.
    Unorm10_10_10_2,
    /// Unorm8x4 BGRA variant.
    Unorm8x4Bgra,
}

/// Enumerates reflected shader stage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedShaderStage {
    /// Vertex variant.
    Vertex,
    /// Fragment variant.
    Fragment,
    /// Compute variant.
    Compute,
}

/// Stores entry metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPoint {
    /// Name.
    pub name: String,
    /// Stage.
    pub stage: ReflectedShaderStage,
}

/// Enumerates reflected type scalar class values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTypeScalarClass {
    /// Float variant.
    Float,
    /// Sint variant.
    Sint,
    /// Uint variant.
    Uint,
    /// Bool variant.
    Bool,
}

/// Stores reflection data for reflected type class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReflectedTypeClass {
    /// Scalar.
    pub scalar: ReflectedTypeScalarClass,
    /// Components.
    pub components: u8,
    /// Width.
    pub width: u8,
}

/// Stores reflection data for reflected io location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedIoLocation {
    /// Location.
    pub location: u32,
    /// Ty.
    pub ty: ReflectedTypeClass,
    /// Interpolation.
    pub interpolation: Option<ReflectedInterpolation>,
    /// Sampling.
    pub sampling: Option<ReflectedSampling>,
}

/// Enumerates reflected interpolation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedInterpolation {
    /// Perspective variant.
    Perspective,
    /// Linear variant.
    Linear,
    /// Flat variant.
    Flat,
    /// Per-vertex variant.
    PerVertex,
}

/// Enumerates reflected interpolation sampling values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedSampling {
    /// Center variant.
    Center,
    /// Centroid variant.
    Centroid,
    /// Sample variant.
    Sample,
    /// First variant.
    First,
    /// Either variant.
    Either,
}

/// Stores entry metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPointIo {
    /// Entry point.
    pub entry_point: String,
    /// Inputs.
    pub inputs: Vec<ReflectedIoLocation>,
    /// Outputs.
    pub outputs: Vec<ReflectedIoLocation>,
    /// Number of stage-input `@builtin`s that consume an inter-stage shader
    /// variable slot toward `maxInterStageShaderVariables` (fragment
    /// `front_facing` / `sample_index` / `sample_mask` / `primitive_index` /
    /// `subgroup_invocation_id` / `subgroup_size`). `@builtin(position)` does
    /// not count.
    pub input_inter_stage_builtins: u32,
}

/// Identifies reflected override key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedOverrideKey {
    /// Name.
    pub name: Option<String>,
    /// Id.
    pub id: Option<u16>,
}

/// Stores reflection data for reflected workgroup size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedWorkgroupSize {
    /// Entry point.
    pub entry_point: String,
    /// Literal size.
    pub literal_size: [u32; 3],
    /// Per-dimension override keys for `@workgroup_size(x, y, z)`.
    ///
    /// Naga already stores the literal fallback in `literal_size`; when a
    /// dimension is override-driven, this key lets pipeline validation apply
    /// pipeline constants before enforcing compute limits.
    pub override_keys: [Option<ReflectedOverrideKey>; 3],
    /// Workgroup storage size.
    pub workgroup_storage_size: u64,
}

/// Enumerates reflected buffer type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedBufferType {
    /// Uniform variant.
    Uniform,
    /// Storage variant.
    Storage,
    /// Read only storage variant.
    ReadOnlyStorage,
}

/// Enumerates reflected texture sample usage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTextureSampleUsage {
    /// Sample variant.
    Sample,
    /// Gather variant.
    Gather,
    /// Load variant.
    Load,
}

/// Stores reflection data for reflected storage texture access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedStorageTextureAccess {
    /// Read.
    pub read: bool,
    /// Write.
    pub write: bool,
}

/// Enumerates reflected resource binding kind values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReflectedResourceBindingKind {
    /// Buffer variant.
    Buffer(ReflectedBufferType),
    /// Sampler variant.
    Sampler {
        /// Comparison variant.
        comparison: bool,
    },
    /// Texture variant.
    Texture {
        /// Sampled variant.
        sampled: bool,
        /// Sample kind variant.
        sample_kind: Option<ReflectedTypeScalarClass>,
        /// Sample usage variant.
        sample_usage: ReflectedTextureSampleUsage,
        /// View dimension variant.
        view_dimension: ReflectedTextureViewDimension,
        /// Multisampled variant.
        multisampled: bool,
    },
    /// Input attachment variant.
    #[cfg(feature = "tiled")]
    InputAttachment {
        /// Sample kind variant.
        sample_kind: ReflectedTypeScalarClass,
        /// Multisampled variant.
        multisampled: bool,
    },
    /// Storage texture variant.
    StorageTexture {
        /// Format variant.
        format: String,
        /// Access variant.
        access: ReflectedStorageTextureAccess,
        /// View dimension variant.
        view_dimension: ReflectedTextureViewDimension,
    },
    /// External texture variant.
    ExternalTexture,
}

/// Enumerates reflected texture view dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTextureViewDimension {
    /// D1 variant.
    D1,
    /// D2 variant.
    D2,
    /// D2 array variant.
    D2Array,
    /// Cube variant.
    Cube,
    /// Cube array variant.
    CubeArray,
    /// D3 variant.
    D3,
}

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedResourceBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Kind.
    pub kind: ReflectedResourceBindingKind,
    /// Min binding size.
    pub min_binding_size: u64,
    /// Statically used.
    pub statically_used: bool,
}

/// Stores reflection data for reflected fragment builtins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedFragmentBuiltins {
    /// Entry point.
    pub entry_point: String,
    /// Frag depth.
    pub frag_depth: bool,
    /// Sample mask.
    pub sample_mask: bool,
}

/// Stores reflection data for reflected override.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReflectedOverride {
    /// Name.
    pub name: Option<String>,
    /// Id.
    pub id: Option<u16>,
    /// Ty.
    pub ty: ReflectedTypeClass,
    /// Has default.
    pub has_default: bool,
    /// Default value.
    pub default_value: Option<ReflectedOverrideValue>,
}

/// Enumerates reflected override value values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReflectedOverrideValue {
    /// Number variant.
    Number(f64),
    /// Bool variant.
    Bool(bool),
}

/// Returns parse and validate wgsl.
pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ReflectedModule, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|error| error.to_string())?;
    validate_module(module)
}

/// Reflects a SPIR-V module without re-emitting the input words.
#[cfg(feature = "shader-passthrough")]
pub(crate) fn reflect_spirv(words: &[u32]) -> Result<ReflectedModule, String> {
    let options = naga::front::spv::Options::default();
    let module = naga::front::spv::Frontend::new(words.iter().copied(), &options)
        .parse()
        .map_err(|error| error.to_string())?;
    validate_module(module)
}

fn validate_module(module: naga::Module) -> Result<ReflectedModule, String> {
    let capabilities = naga::valid::Capabilities::SHADER_FLOAT16
        | naga::valid::Capabilities::CUBE_ARRAY_TEXTURES
        | naga::valid::Capabilities::MULTISAMPLED_SHADING
        | naga::valid::Capabilities::STORAGE_TEXTURE_16BIT_NORM_FORMATS
        | naga::valid::Capabilities::TEXTURE_EXTERNAL;
    // Enabled capabilities:
    // - SHADER_FLOAT16: Phase-5 overridable-constant validation needs WGSL
    //   `enable f16; override x: f16;` shaders from Dawn.
    // - CUBE_ARRAY_TEXTURES + MULTISAMPLED_SHADING: WebGPU-baseline sampled
    //   texture types (`texture_cube_array<T>`, `texture_multisampled_2d<T>`).
    //   naga gates both behind a capability; omitting them turned every such
    //   shader into an error module (F-057). wgpu enables both for any WebGPU
    //   device (the `CUBE_ARRAY_TEXTURES` / `MULTISAMPLED_SHADING` downlevel
    //   flags are baseline on Metal/Vulkan).
    // - STORAGE_TEXTURE_16BIT_NORM_FORMATS: the 16-bit-norm storage formats
    //   (`r16unorm`, `rgba16snorm`, …) are baseline-storage in WebGPU; naga gates
    //   the WGSL `texture_storage_*<r16unorm, …>` types behind this (F-059).
    // - TEXTURE_EXTERNAL: WebGPU baseline external textures (`texture_external`).
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities);
    let info = validator
        .validate(&module)
        .map_err(|error| error.to_string())?;
    Ok(ReflectedModule { module, info })
}

impl ReflectedModule {
    /// Generates spirv for the validated shader module.
    pub(crate) fn generate_spirv(
        &self,
        entry_name: &str,
        stage: naga::ShaderStage,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<Vec<u32>, String> {
        let (module, info) =
            self.process_overrides_for_entry(entry_name, stage, pipeline_constants)?;
        // naga's SPIR-V backend does not implement `ImageClass::External`, and
        // wgpu likewise leaves external textures unimplemented on Vulkan (both
        // naga-SPIR-V and wgpu-hal/vulkan). Reject external-texture pipelines on
        // the Vulkan backend with a clean error (NOT a panic, and NOT a fake
        // texture_2d rewrite that would silently mis-sample) — yawgpu's external
        // texture support is Metal-only.
        if module_has_external_texture(&module) {
            return Err("external textures are not supported on the Vulkan backend".to_owned());
        }
        let options = naga::back::spv::Options {
            fake_missing_bindings: true,
            ..Default::default()
        };
        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: entry_name.to_owned(),
        };
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
            .map_err(|error| error.to_string())
    }

    /// Generates GLSL ES for the validated shader module.
    #[cfg(feature = "gles")]
    pub(crate) fn generate_glsl(
        &self,
        entry_name: &str,
        stage: naga::ShaderStage,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<GeneratedGlsl, String> {
        let (module, info) =
            self.process_overrides_for_entry(entry_name, stage, pipeline_constants)?;
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            },
            writer_flags: naga::back::glsl::WriterFlags::empty(),
            binding_map: naga::back::glsl::BindingMap::default(),
            zero_initialize_workgroup_memory: true,
            use_framebuffer_fetch: false,
        };
        let pipeline_options = naga::back::glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry_name.to_owned(),
            multiview: None,
        };
        let mut source = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut source,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .map_err(|error| error.to_string())?;
        let _reflection = writer.write().map_err(|error| error.to_string())?;
        Ok(GeneratedGlsl {
            source,
            entry_point: entry_name.to_owned(),
        })
    }

    /// Generates msl for the validated shader module.
    pub(crate) fn generate_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        let (module, info) = self.process_overrides_for_entry(
            entry_name,
            naga::ShaderStage::Compute,
            pipeline_constants,
        )?;
        let resources = msl_resources(binding_map)?;
        let buffer_size_bindings = msl_buffer_size_bindings_for_entry(&module, entry_name)?;
        let buffer_sizes_slot = msl_buffer_sizes_slot(binding_map, &buffer_size_bindings, &[])?;
        let workgroup_memory_sizes =
            collect_workgroup_memory_sizes(&module, &info, entry_name)?;
        let mut per_entry_point_map = BTreeMap::new();
        per_entry_point_map.insert(
            entry_name.to_owned(),
            naga::back::msl::EntryPointResources {
                resources,
                sizes_buffer: buffer_sizes_slot,
                ..Default::default()
            },
        );
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            per_entry_point_map,
            fake_missing_bindings: false,
            bounds_check_policies: msl_bounds_check_policies(),
            ..Default::default()
        };
        let pipeline_options = naga::back::msl::PipelineOptions {
            entry_point: Some((naga::ShaderStage::Compute, entry_name.to_owned())),
            ..Default::default()
        };
        let (source, write_info) =
            naga::back::msl::write_string(&module, &info, &options, &pipeline_options)
                .map_err(|error| error.to_string())?;
        let entry_point =
            emitted_entry_point_name(&module, &write_info, naga::ShaderStage::Compute, entry_name)?;
        Ok(GeneratedMsl {
            source,
            entry_point,
            buffer_sizes_slot: buffer_sizes_slot.map(u32::from),
            buffer_size_bindings,
            frag_depth_clamp_slot: None,
            workgroup_memory_sizes,
        })
    }

    /// Generates render vertex MSL for a validated shader module.
    pub(crate) fn generate_render_vertex_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        vertex_buffers: &[MslVertexBufferBinding],
        force_point_size: bool,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        self.generate_render_stage_msl(
            entry_name,
            naga::ShaderStage::Vertex,
            binding_map,
            RenderMslStageOptions {
                vertex_buffer_mappings: msl_vertex_buffer_mappings(vertex_buffers)?,
                force_point_size,
                subpass_color_slots: &[],
                pipeline_constants,
                sample_mask: u32::MAX,
            },
        )
    }

    /// Generates render fragment MSL for a validated shader module.
    pub(crate) fn generate_render_fragment_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        subpass_color_slots: &[((u32, u32), u32)],
        pipeline_constants: &naga::back::PipelineConstants,
        sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        self.generate_render_stage_msl(
            entry_name,
            naga::ShaderStage::Fragment,
            binding_map,
            RenderMslStageOptions {
                vertex_buffer_mappings: Vec::new(),
                force_point_size: false,
                subpass_color_slots,
                pipeline_constants,
                sample_mask,
            },
        )
    }

    fn generate_render_stage_msl(
        &self,
        entry_name: &str,
        stage: naga::ShaderStage,
        binding_map: &MslBindingMap,
        stage_options: RenderMslStageOptions<'_>,
    ) -> Result<GeneratedMsl, String> {
        let (module, info) =
            self.process_overrides_for_entry(entry_name, stage, stage_options.pipeline_constants)?;
        let needs_frag_depth_clamp = stage == naga::ShaderStage::Fragment
            && fragment_entry_writes_frag_depth(&module, entry_name);
        let (module, info) = if needs_frag_depth_clamp {
            naga::back::clamp_frag_depth::clamp_frag_depth(
                &module,
                &info,
                (naga::ShaderStage::Fragment, entry_name),
            )
            .map_err(|error| error.to_string())?
        } else {
            (module, info)
        };
        let (module, info) =
            if stage == naga::ShaderStage::Fragment && stage_options.sample_mask != u32::MAX {
                naga::back::sample_mask::apply_sample_mask(
                    &module,
                    &info,
                    (naga::ShaderStage::Fragment, entry_name),
                    stage_options.sample_mask,
                )
                .map_err(|error| error.to_string())?
            } else {
                (module, info)
            };
        let resources = msl_resources(binding_map)?;
        let buffer_size_bindings = msl_buffer_size_bindings_for_entry(&module, entry_name)?;
        let vertex_buffer_indices = stage_options
            .vertex_buffer_mappings
            .iter()
            .map(|mapping| mapping.id)
            .collect::<Vec<_>>();
        let buffer_sizes_slot =
            msl_buffer_sizes_slot(binding_map, &buffer_size_bindings, &vertex_buffer_indices)?;
        let frag_depth_clamp_slot = if needs_frag_depth_clamp {
            let size_slot = buffer_sizes_slot
                .map(u32::from)
                .into_iter()
                .collect::<Vec<_>>();
            Some(msl_next_buffer_slot(binding_map, &size_slot)?)
        } else {
            None
        };
        let mut per_entry_point_map = BTreeMap::new();
        per_entry_point_map.insert(
            entry_name.to_owned(),
            naga::back::msl::EntryPointResources {
                resources,
                sizes_buffer: buffer_sizes_slot,
                immediates_buffer: frag_depth_clamp_slot,
            },
        );
        let mut color_slot_map = naga::FastHashMap::default();
        for &(key, slot) in stage_options.subpass_color_slots {
            color_slot_map.insert(key, slot);
        }
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            per_entry_point_map,
            fake_missing_bindings: false,
            subpass_color_slots: color_slot_map,
            bounds_check_policies: msl_bounds_check_policies(),
            ..Default::default()
        };
        let pipeline_options = naga::back::msl::PipelineOptions {
            entry_point: Some((stage, entry_name.to_owned())),
            vertex_buffer_mappings: stage_options.vertex_buffer_mappings,
            allow_and_force_point_size: stage_options.force_point_size,
            ..Default::default()
        };
        let (source, write_info) =
            naga::back::msl::write_string(&module, &info, &options, &pipeline_options)
                .map_err(|error| error.to_string())?;
        let entry_point = emitted_entry_point_name(&module, &write_info, stage, entry_name)?;
        Ok(GeneratedMsl {
            source,
            entry_point,
            buffer_sizes_slot: buffer_sizes_slot.map(u32::from),
            buffer_size_bindings,
            frag_depth_clamp_slot: frag_depth_clamp_slot.map(u32::from),
            // Render stages (vertex/fragment) cannot use var<workgroup> through this
            // path; workgroup_memory_sizes only applies to compute pipelines.
            workgroup_memory_sizes: Vec::new(),
        })
    }

    /// Generates render msl for the validated shader module.
    pub(crate) fn generate_render_msl(
        &self,
        vertex_entry_name: &str,
        fragment_entry_name: Option<&str>,
        binding_map: &MslBindingMap,
        vertex_buffers: &[MslVertexBufferBinding],
        subpass_color_slots: &[((u32, u32), u32)],
        force_point_size: bool,
    ) -> Result<GeneratedRenderMsl, String> {
        let empty_pipeline_constants = naga::back::PipelineConstants::default();
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &self.module,
            &self.info,
            None,
            &empty_pipeline_constants,
        )
        .map_err(|error| error.to_string())?;
        let resources = msl_resources(binding_map)?;
        let vertex_buffer_mappings = msl_vertex_buffer_mappings(vertex_buffers)?;
        let vertex_buffer_indices = vertex_buffer_mappings
            .iter()
            .map(|mapping| mapping.id)
            .collect::<Vec<_>>();
        let vertex_buffer_size_bindings =
            msl_buffer_size_bindings_for_entry(&module, vertex_entry_name)?;
        let vertex_buffer_sizes_slot = msl_buffer_sizes_slot(
            binding_map,
            &vertex_buffer_size_bindings,
            &vertex_buffer_indices,
        )?;
        let fragment_buffer_size_bindings = fragment_entry_name
            .map(|entry| msl_buffer_size_bindings_for_entry(&module, entry))
            .transpose()?
            .unwrap_or_default();
        let fragment_buffer_sizes_slot =
            msl_buffer_sizes_slot(binding_map, &fragment_buffer_size_bindings, &[])?;
        let mut per_entry_point_map = BTreeMap::new();
        per_entry_point_map.insert(
            vertex_entry_name.to_owned(),
            naga::back::msl::EntryPointResources {
                resources: resources.clone(),
                sizes_buffer: vertex_buffer_sizes_slot,
                ..Default::default()
            },
        );
        if let Some(fragment_entry_name) = fragment_entry_name {
            per_entry_point_map.insert(
                fragment_entry_name.to_owned(),
                naga::back::msl::EntryPointResources {
                    resources,
                    sizes_buffer: fragment_buffer_sizes_slot,
                    ..Default::default()
                },
            );
        }
        let mut color_slot_map = naga::FastHashMap::default();
        for &(key, slot) in subpass_color_slots {
            color_slot_map.insert(key, slot);
        }
        let options = naga::back::msl::Options {
            lang_version: (2, 4),
            per_entry_point_map,
            fake_missing_bindings: false,
            subpass_color_slots: color_slot_map,
            bounds_check_policies: msl_bounds_check_policies(),
            ..Default::default()
        };
        let pipeline_options = naga::back::msl::PipelineOptions {
            entry_point: None,
            vertex_buffer_mappings,
            allow_and_force_point_size: force_point_size,
            ..Default::default()
        };
        let (source, info) =
            naga::back::msl::write_string(&module, &info, &options, &pipeline_options)
                .map_err(|error| error.to_string())?;
        let vertex_entry_point =
            emitted_entry_point_name(&module, &info, naga::ShaderStage::Vertex, vertex_entry_name)?;
        let fragment_entry_point = fragment_entry_name
            .map(|fragment_entry_name| {
                emitted_entry_point_name(
                    &module,
                    &info,
                    naga::ShaderStage::Fragment,
                    fragment_entry_name,
                )
            })
            .transpose()?;
        Ok(GeneratedRenderMsl {
            source,
            vertex_entry_point,
            fragment_entry_point,
            vertex_buffer_sizes_slot: vertex_buffer_sizes_slot.map(u32::from),
            vertex_buffer_size_bindings,
            fragment_buffer_sizes_slot: fragment_buffer_sizes_slot.map(u32::from),
            fragment_buffer_size_bindings,
        })
    }

    /// Returns entry points reflected by the validated shader module.
    pub(crate) fn entry_points(&self) -> Vec<ReflectedEntryPoint> {
        self.module
            .entry_points
            .iter()
            .filter_map(|entry| {
                Some(ReflectedEntryPoint {
                    name: entry.name.clone(),
                    stage: map_shader_stage(entry.stage)?,
                })
            })
            .collect()
    }

    /// Returns compute workgroup size reflected by the validated shader module.
    pub(crate) fn compute_workgroup_size(
        &self,
        entry_point: &str,
    ) -> Result<Option<ReflectedWorkgroupSize>, String> {
        let Some((entry_index, entry)) =
            self.module
                .entry_points
                .iter()
                .enumerate()
                .find(|(_, entry)| {
                    entry.name == entry_point && entry.stage == naga::ShaderStage::Compute
                })
        else {
            return Ok(None);
        };

        let mut override_keys: [Option<ReflectedOverrideKey>; 3] = [None, None, None];
        if let Some(overrides) = entry.workgroup_size_overrides {
            for (index, expression) in overrides.into_iter().enumerate() {
                override_keys[index] = expression
                    .and_then(|expression| override_key_from_expression(&self.module, expression));
            }
        }

        Ok(Some(ReflectedWorkgroupSize {
            entry_point: entry.name.clone(),
            literal_size: entry.workgroup_size,
            override_keys,
            workgroup_storage_size: self.workgroup_storage_size_for_entry(entry_index)?,
        }))
    }

    /// Returns compute workgroup size after resolving pipeline constants.
    pub(crate) fn resolved_compute_workgroup_size(
        &self,
        entry_point: &str,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<ReflectedWorkgroupSize, String> {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &self.module,
            &self.info,
            Some((naga::ShaderStage::Compute, entry_point)),
            pipeline_constants,
        )
        .map_err(|error| error.to_string())?;
        resolved_workgroup_size(&module, &info, entry_point)
    }

    fn process_overrides_for_entry<'a>(
        &'a self,
        entry_name: &str,
        stage: naga::ShaderStage,
        pipeline_constants: &naga::back::PipelineConstants,
    ) -> Result<
        (
            std::borrow::Cow<'a, naga::Module>,
            std::borrow::Cow<'a, naga::valid::ModuleInfo>,
        ),
        String,
    > {
        naga::back::pipeline_constants::process_overrides(
            &self.module,
            &self.info,
            Some((stage, entry_name)),
            pipeline_constants,
        )
        .map_err(|error| error.to_string())
    }

    /// Returns entry point io reflected by the validated shader module.
    pub(crate) fn entry_point_io(&self) -> Vec<ReflectedEntryPointIo> {
        self.module
            .entry_points
            .iter()
            .filter_map(|entry| {
                let stage = map_shader_stage(entry.stage)?;
                Some(ReflectedEntryPointIo {
                    entry_point: entry.name.clone(),
                    inputs: collect_function_inputs(&self.module, &entry.function, stage),
                    outputs: collect_function_outputs(&self.module, &entry.function, stage),
                    input_inter_stage_builtins: count_inter_stage_input_builtins(
                        &self.module,
                        &entry.function,
                        stage,
                    ),
                })
            })
            .collect()
    }

    /// Returns resource bindings reflected by the validated shader module.
    pub(crate) fn resource_bindings(&self) -> Vec<ReflectedResourceBinding> {
        let mut layouter = naga::proc::Layouter::default();
        let layout_ready = layouter.update(self.module.to_ctx()).is_ok();
        self.module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                let binding = global.binding?;
                let kind = resource_binding_kind(&self.module, global, handle, None)?;
                let min_binding_size = if layout_ready {
                    resource_binding_min_size(&layouter, global)
                } else {
                    0
                };
                Some(ReflectedResourceBinding {
                    group: binding.group,
                    binding: binding.binding,
                    kind,
                    min_binding_size,
                    statically_used: self
                        .module
                        .entry_points
                        .iter()
                        .enumerate()
                        .any(|(index, _)| !self.info.get_entry_point(index)[handle].is_empty()),
                })
            })
            .collect()
    }

    /// Returns resource bindings for entry reflected by the validated shader module.
    pub(crate) fn resource_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<ReflectedResourceBinding>, String> {
        let Some((entry_index, _)) = self
            .module
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.name == entry_point)
        else {
            return Err("shader entry point was not found for resource reflection".to_owned());
        };

        let mut layouter = naga::proc::Layouter::default();
        let layout_ready = layouter.update(self.module.to_ctx()).is_ok();
        Ok(self
            .module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                let binding = global.binding?;
                if self.info.get_entry_point(entry_index)[handle].is_empty() {
                    return None;
                }
                let kind = resource_binding_kind(&self.module, global, handle, Some(entry_index))?;
                let min_binding_size = if layout_ready {
                    resource_binding_min_size(&layouter, global)
                } else {
                    0
                };
                Some(ReflectedResourceBinding {
                    group: binding.group,
                    binding: binding.binding,
                    kind,
                    min_binding_size,
                    statically_used: true,
                })
            })
            .collect())
    }

    /// Returns storage buffer bindings that populate MSL `_mslBufferSizes`.
    pub(crate) fn msl_buffer_size_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<MslBufferSizeBinding>, String> {
        msl_buffer_size_bindings_for_entry(&self.module, entry_point)
    }

    /// Returns fragment builtins reflected by the validated shader module.
    pub(crate) fn fragment_builtins(&self) -> Vec<ReflectedFragmentBuiltins> {
        self.module
            .entry_points
            .iter()
            .filter(|entry| entry.stage == naga::ShaderStage::Fragment)
            .map(|entry| {
                let mut builtins = ReflectedFragmentBuiltins {
                    entry_point: entry.name.clone(),
                    frag_depth: false,
                    sample_mask: false,
                };
                collect_output_builtins(&self.module, &entry.function, &mut builtins);
                builtins
            })
            .collect()
    }

    /// Returns overrides reflected by the validated shader module.
    pub(crate) fn overrides(&self) -> Vec<ReflectedOverride> {
        self.module
            .overrides
            .iter()
            .filter_map(|(_, override_)| {
                Some(ReflectedOverride {
                    name: override_.name.clone(),
                    id: override_.id,
                    ty: type_class(&self.module, override_.ty)?,
                    has_default: override_.init.is_some(),
                    default_value: override_
                        .init
                        .and_then(|init| override_default_value(&self.module, init)),
                })
            })
            .collect()
    }

    fn workgroup_storage_size_for_entry(&self, entry_index: usize) -> Result<u64, String> {
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(self.module.to_ctx())
            .map_err(|error| error.to_string())?;
        let mut size = 0u64;
        for (handle, global) in self
            .module
            .global_variables
            .iter()
            .filter(|(handle, global)| {
                global.space == naga::AddressSpace::WorkGroup
                    && !self.info.get_entry_point(entry_index)[*handle].is_empty()
            })
        {
            let global_size = u64::from(layouter[global.ty].size);
            size = size.checked_add(global_size).ok_or_else(|| {
                format!("compute workgroup storage size overflows at global {handle:?}")
            })?;
        }
        Ok(size)
    }
}

fn resolved_workgroup_size(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_point: &str,
) -> Result<ReflectedWorkgroupSize, String> {
    let Some((entry_index, entry)) =
        module.entry_points.iter().enumerate().find(|(_, entry)| {
            entry.name == entry_point && entry.stage == naga::ShaderStage::Compute
        })
    else {
        return Err("compute entry point workgroup size reflection failed".to_owned());
    };

    let mut layouter = naga::proc::Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| error.to_string())?;
    let mut storage_size = 0u64;
    for (handle, global) in module.global_variables.iter().filter(|(handle, global)| {
        global.space == naga::AddressSpace::WorkGroup
            && !info.get_entry_point(entry_index)[*handle].is_empty()
    }) {
        let global_size = u64::from(layouter[global.ty].size);
        storage_size = storage_size.checked_add(global_size).ok_or_else(|| {
            format!("compute workgroup storage size overflows at global {handle:?}")
        })?;
    }

    Ok(ReflectedWorkgroupSize {
        entry_point: entry.name.clone(),
        literal_size: entry.workgroup_size,
        override_keys: [None, None, None],
        workgroup_storage_size: storage_size,
    })
}

/// Collects the per-threadgroup-argument allocation sizes for a compute entry point.
///
/// naga's MSL backend emits each `var<workgroup>` global used by the entry point as
/// a kernel argument `[[threadgroup(N)]]`, in the iteration order of the module's
/// global-variable arena.  Metal requires the compute encoder to call
/// `setThreadgroupMemoryLength:atIndex:` for each such slot before dispatch, with a
/// length rounded up to a multiple of 16 bytes (the Metal alignment requirement).
/// This function mirrors the logic in wgpu-hal/src/metal/device.rs `load_shader`
/// (lines 344-352), using `module.types[var.ty].inner.size(module.to_ctx())` for the
/// raw byte size and then `next_multiple_of(16)` for the aligned allocation size.
fn collect_workgroup_memory_sizes(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    entry_name: &str,
) -> Result<Vec<u32>, String> {
    let Some(entry_index) = module
        .entry_points
        .iter()
        .position(|entry| entry.name == entry_name && entry.stage == naga::ShaderStage::Compute)
    else {
        return Err("compute entry point was not found for workgroup memory reflection".to_owned());
    };
    let ep_info = info.get_entry_point(entry_index);
    let mut sizes = Vec::new();
    for (var_handle, var) in module.global_variables.iter() {
        if var.space == naga::AddressSpace::WorkGroup && !ep_info[var_handle].is_empty() {
            // `TypeInner::size` returns the byte count of the type as laid out
            // in memory, matching naga's MSL emission for the threadgroup arg.
            let raw_size = module.types[var.ty].inner.size(module.to_ctx());
            // Metal requires the threadgroup slot length to be a multiple of 16.
            let aligned = raw_size.next_multiple_of(16);
            sizes.push(aligned);
        }
    }
    Ok(sizes)
}

fn msl_resources(binding_map: &MslBindingMap) -> Result<naga::back::msl::BindingMap, String> {
    binding_map
        .resources
        .iter()
        .map(|binding| {
            let slot = u8::try_from(binding.metal_index)
                .map_err(|_| "MSL resource index exceeds the supported slot range".to_owned())?;
            Ok((
                naga::ResourceBinding {
                    group: binding.group,
                    binding: binding.binding,
                },
                match binding.kind {
                    MslResourceBindingKind::Buffer => naga::back::msl::BindTarget {
                        buffer: Some(slot),
                        ..Default::default()
                    },
                    MslResourceBindingKind::Texture => naga::back::msl::BindTarget {
                        texture: Some(slot),
                        ..Default::default()
                    },
                    MslResourceBindingKind::Sampler => naga::back::msl::BindTarget {
                        sampler: Some(naga::back::msl::BindSamplerTarget::Resource(slot)),
                        ..Default::default()
                    },
                    MslResourceBindingKind::ExternalTexture => {
                        let plane0 = slot;
                        let plane1 = slot.checked_add(1).ok_or_else(|| {
                            "MSL external texture plane index exceeds the supported slot range"
                                .to_owned()
                        })?;
                        let plane2 = slot.checked_add(2).ok_or_else(|| {
                            "MSL external texture plane index exceeds the supported slot range"
                                .to_owned()
                        })?;
                        let params = slot.checked_add(3).ok_or_else(|| {
                            "MSL external texture params index exceeds the supported slot range"
                                .to_owned()
                        })?;
                        naga::back::msl::BindTarget {
                            external_texture: Some(naga::back::msl::BindExternalTextureTarget {
                                planes: [plane0, plane1, plane2],
                                params,
                            }),
                            ..Default::default()
                        }
                    }
                },
            ))
        })
        .collect()
}

/// Returns true when any image type in the module is an external texture
/// (`ImageClass::External`). naga's SPIR-V backend does not implement external
/// textures, so the Vulkan code path uses this to reject such pipelines with a
/// clean error instead of panicking.
fn module_has_external_texture(module: &naga::Module) -> bool {
    module.types.iter().any(|(_, ty)| {
        matches!(
            ty.inner,
            naga::TypeInner::Image {
                class: naga::ImageClass::External,
                ..
            }
        )
    })
}

fn msl_buffer_sizes_slot(
    binding_map: &MslBindingMap,
    buffer_size_bindings: &[MslBufferSizeBinding],
    extra_buffer_indices: &[u32],
) -> Result<Option<naga::back::msl::Slot>, String> {
    if buffer_size_bindings.is_empty() {
        return Ok(None);
    }
    msl_next_buffer_slot(binding_map, extra_buffer_indices).map(Some)
}

fn msl_next_buffer_slot(
    binding_map: &MslBindingMap,
    extra_buffer_indices: &[u32],
) -> Result<naga::back::msl::Slot, String> {
    let resource_max = binding_map
        .resources
        .iter()
        .filter_map(|binding| match binding.kind {
            MslResourceBindingKind::Buffer => Some(binding.metal_index),
            MslResourceBindingKind::ExternalTexture => Some(binding.metal_index.saturating_add(3)),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let next_slot = extra_buffer_indices
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(resource_max)
        .saturating_add(1);
    u8::try_from(next_slot)
        .map_err(|_| "MSL generated buffer slot exceeds the supported slot range".to_owned())
}

fn msl_buffer_size_bindings_for_entry(
    module: &naga::Module,
    entry_point: &str,
) -> Result<Vec<MslBufferSizeBinding>, String> {
    if !module
        .entry_points
        .iter()
        .any(|entry| entry.name == entry_point)
    {
        return Err("shader entry point was not found for MSL buffer-size reflection".to_owned());
    }
    Ok(module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            let binding = global.binding?;
            if !matches!(global.space, naga::AddressSpace::Storage { .. })
                || !msl_needs_array_length(global.ty, &module.types)
            {
                return None;
            }
            Some(MslBufferSizeBinding {
                group: binding.group,
                binding: binding.binding,
            })
        })
        .collect())
}

fn msl_bounds_check_policies() -> naga::proc::BoundsCheckPolicies {
    let bounds_check_policy = naga::proc::BoundsCheckPolicy::Restrict;
    naga::proc::BoundsCheckPolicies {
        index: bounds_check_policy,
        buffer: bounds_check_policy,
        image_load: bounds_check_policy,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    }
}

fn msl_needs_array_length(
    ty: naga::Handle<naga::Type>,
    arena: &naga::UniqueArena<naga::Type>,
) -> bool {
    match arena[ty].inner {
        naga::TypeInner::Struct { ref members, .. } => members.last().is_some_and(|member| {
            matches!(
                arena[member.ty].inner,
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                }
            )
        }),
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => true,
        _ => false,
    }
}

fn fragment_entry_writes_frag_depth(module: &naga::Module, entry_point: &str) -> bool {
    module
        .entry_points
        .iter()
        .find(|entry| entry.stage == naga::ShaderStage::Fragment && entry.name == entry_point)
        .is_some_and(|entry| {
            let mut builtins = ReflectedFragmentBuiltins {
                entry_point: entry.name.clone(),
                frag_depth: false,
                sample_mask: false,
            };
            collect_output_builtins(module, &entry.function, &mut builtins);
            builtins.frag_depth
        })
}

fn msl_vertex_buffer_mappings(
    vertex_buffers: &[MslVertexBufferBinding],
) -> Result<Vec<naga::back::msl::VertexBufferMapping>, String> {
    vertex_buffers
        .iter()
        .map(|buffer| {
            Ok(naga::back::msl::VertexBufferMapping {
                id: buffer.metal_index,
                stride: u32::try_from(buffer.array_stride)
                    .map_err(|_| "MSL vertex stride exceeds the supported range".to_owned())?,
                step_mode: match buffer.step_mode {
                    MslVertexStepMode::Vertex => naga::back::msl::VertexBufferStepMode::ByVertex,
                    MslVertexStepMode::Instance => {
                        naga::back::msl::VertexBufferStepMode::ByInstance
                    }
                },
                attributes: buffer
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(naga::back::msl::AttributeMapping {
                            shader_location: attribute.shader_location,
                            offset: u32::try_from(attribute.offset).map_err(|_| {
                                "MSL vertex attribute offset exceeds the supported range".to_owned()
                            })?,
                            format: msl_vertex_format(attribute.format),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            })
        })
        .collect()
}

fn msl_vertex_format(format: MslVertexFormat) -> naga::back::msl::VertexFormat {
    match format {
        MslVertexFormat::Uint8 => naga::back::msl::VertexFormat::Uint8,
        MslVertexFormat::Uint8x2 => naga::back::msl::VertexFormat::Uint8x2,
        MslVertexFormat::Uint8x4 => naga::back::msl::VertexFormat::Uint8x4,
        MslVertexFormat::Sint8 => naga::back::msl::VertexFormat::Sint8,
        MslVertexFormat::Sint8x2 => naga::back::msl::VertexFormat::Sint8x2,
        MslVertexFormat::Sint8x4 => naga::back::msl::VertexFormat::Sint8x4,
        MslVertexFormat::Unorm8 => naga::back::msl::VertexFormat::Unorm8,
        MslVertexFormat::Unorm8x2 => naga::back::msl::VertexFormat::Unorm8x2,
        MslVertexFormat::Unorm8x4 => naga::back::msl::VertexFormat::Unorm8x4,
        MslVertexFormat::Snorm8 => naga::back::msl::VertexFormat::Snorm8,
        MslVertexFormat::Snorm8x2 => naga::back::msl::VertexFormat::Snorm8x2,
        MslVertexFormat::Snorm8x4 => naga::back::msl::VertexFormat::Snorm8x4,
        MslVertexFormat::Uint16 => naga::back::msl::VertexFormat::Uint16,
        MslVertexFormat::Uint16x2 => naga::back::msl::VertexFormat::Uint16x2,
        MslVertexFormat::Uint16x4 => naga::back::msl::VertexFormat::Uint16x4,
        MslVertexFormat::Sint16 => naga::back::msl::VertexFormat::Sint16,
        MslVertexFormat::Sint16x2 => naga::back::msl::VertexFormat::Sint16x2,
        MslVertexFormat::Sint16x4 => naga::back::msl::VertexFormat::Sint16x4,
        MslVertexFormat::Unorm16 => naga::back::msl::VertexFormat::Unorm16,
        MslVertexFormat::Unorm16x2 => naga::back::msl::VertexFormat::Unorm16x2,
        MslVertexFormat::Unorm16x4 => naga::back::msl::VertexFormat::Unorm16x4,
        MslVertexFormat::Snorm16 => naga::back::msl::VertexFormat::Snorm16,
        MslVertexFormat::Snorm16x2 => naga::back::msl::VertexFormat::Snorm16x2,
        MslVertexFormat::Snorm16x4 => naga::back::msl::VertexFormat::Snorm16x4,
        MslVertexFormat::Float16 => naga::back::msl::VertexFormat::Float16,
        MslVertexFormat::Float16x2 => naga::back::msl::VertexFormat::Float16x2,
        MslVertexFormat::Float16x4 => naga::back::msl::VertexFormat::Float16x4,
        MslVertexFormat::Float32 => naga::back::msl::VertexFormat::Float32,
        MslVertexFormat::Float32x2 => naga::back::msl::VertexFormat::Float32x2,
        MslVertexFormat::Float32x3 => naga::back::msl::VertexFormat::Float32x3,
        MslVertexFormat::Float32x4 => naga::back::msl::VertexFormat::Float32x4,
        MslVertexFormat::Uint32 => naga::back::msl::VertexFormat::Uint32,
        MslVertexFormat::Uint32x2 => naga::back::msl::VertexFormat::Uint32x2,
        MslVertexFormat::Uint32x3 => naga::back::msl::VertexFormat::Uint32x3,
        MslVertexFormat::Uint32x4 => naga::back::msl::VertexFormat::Uint32x4,
        MslVertexFormat::Sint32 => naga::back::msl::VertexFormat::Sint32,
        MslVertexFormat::Sint32x2 => naga::back::msl::VertexFormat::Sint32x2,
        MslVertexFormat::Sint32x3 => naga::back::msl::VertexFormat::Sint32x3,
        MslVertexFormat::Sint32x4 => naga::back::msl::VertexFormat::Sint32x4,
        MslVertexFormat::Unorm10_10_10_2 => naga::back::msl::VertexFormat::Unorm10_10_10_2,
        MslVertexFormat::Unorm8x4Bgra => naga::back::msl::VertexFormat::Unorm8x4Bgra,
    }
}

fn emitted_entry_point_name(
    module: &naga::Module,
    info: &naga::back::msl::TranslationInfo,
    stage: naga::ShaderStage,
    entry_name: &str,
) -> Result<String, String> {
    let entry_index = module
        .entry_points
        .iter()
        .position(|entry| entry.name == entry_name && entry.stage == stage)
        .ok_or_else(|| "MSL entry point was not found".to_owned())?;
    info.entry_point_names
        .get(entry_index)
        .ok_or_else(|| "MSL entry point name was not emitted".to_owned())?
        .as_ref()
        .map_err(|error| error.to_string())
        .cloned()
}

fn map_shader_stage(stage: naga::ShaderStage) -> Option<ReflectedShaderStage> {
    match stage {
        naga::ShaderStage::Vertex => Some(ReflectedShaderStage::Vertex),
        naga::ShaderStage::Fragment => Some(ReflectedShaderStage::Fragment),
        naga::ShaderStage::Compute => Some(ReflectedShaderStage::Compute),
        _ => None,
    }
}

fn override_key_from_expression(
    module: &naga::Module,
    expression: naga::Handle<naga::Expression>,
) -> Option<ReflectedOverrideKey> {
    match module.global_expressions.try_get(expression).ok()? {
        naga::Expression::Override(handle) => {
            let override_ = module.overrides.try_get(*handle).ok()?;
            Some(ReflectedOverrideKey {
                name: override_.name.clone(),
                id: override_.id,
            })
        }
        _ => None,
    }
}

fn collect_function_inputs(
    module: &naga::Module,
    function: &naga::Function,
    stage: ReflectedShaderStage,
) -> Vec<ReflectedIoLocation> {
    if stage == ReflectedShaderStage::Compute {
        return Vec::new();
    }

    let mut locations = Vec::new();
    for argument in &function.arguments {
        collect_binding_locations(
            module,
            argument.ty,
            argument.binding.as_ref(),
            &mut locations,
        );
    }
    locations
}

/// Returns true for the stage-input `@builtin`s that consume a
/// `maxInterStageShaderVariables` slot (per WebGPU): `front_facing`,
/// `sample_index`, `sample_mask`, `primitive_index`, `subgroup_invocation_id`,
/// `subgroup_size`. `position` and every other built-in are excluded.
fn is_inter_stage_counting_builtin(builtin: naga::BuiltIn) -> bool {
    matches!(
        builtin,
        naga::BuiltIn::FrontFacing
            | naga::BuiltIn::SampleIndex
            | naga::BuiltIn::SampleMask
            | naga::BuiltIn::PrimitiveIndex
            | naga::BuiltIn::SubgroupInvocationId
            | naga::BuiltIn::SubgroupSize
    )
}

/// Counts the inter-stage-slot-consuming stage-input `@builtin`s on a fragment
/// (or vertex) entry point. They may appear directly on an argument or as the
/// members of a struct argument.
fn count_inter_stage_input_builtins(
    module: &naga::Module,
    function: &naga::Function,
    stage: ReflectedShaderStage,
) -> u32 {
    if stage == ReflectedShaderStage::Compute {
        return 0;
    }
    let mut count = 0;
    for argument in &function.arguments {
        match argument.binding.as_ref() {
            Some(naga::Binding::BuiltIn(builtin)) => {
                if is_inter_stage_counting_builtin(*builtin) {
                    count += 1;
                }
            }
            _ => {
                if let naga::TypeInner::Struct { members, .. } = &module.types[argument.ty].inner {
                    for member in members {
                        if let Some(naga::Binding::BuiltIn(builtin)) = member.binding.as_ref() {
                            if is_inter_stage_counting_builtin(*builtin) {
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

fn collect_function_outputs(
    module: &naga::Module,
    function: &naga::Function,
    stage: ReflectedShaderStage,
) -> Vec<ReflectedIoLocation> {
    if stage == ReflectedShaderStage::Compute {
        return Vec::new();
    }

    let mut locations = Vec::new();
    if let Some(result) = &function.result {
        collect_binding_locations(module, result.ty, result.binding.as_ref(), &mut locations);
    }
    locations
}

fn collect_binding_locations(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
    locations: &mut Vec<ReflectedIoLocation>,
) {
    if let Some(naga::Binding::Location {
        location,
        interpolation,
        sampling,
        ..
    }) = binding
    {
        if let Some(ty) = type_class(module, ty) {
            locations.push(ReflectedIoLocation {
                location: *location,
                ty,
                interpolation: interpolation.map(map_interpolation),
                sampling: sampling.map(map_sampling),
            });
        }
        return;
    }

    let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        return;
    };
    for member in members {
        if let Some(naga::Binding::Location {
            location,
            interpolation,
            sampling,
            ..
        }) = member.binding.as_ref()
        {
            if let Some(ty) = type_class(module, member.ty) {
                locations.push(ReflectedIoLocation {
                    location: *location,
                    ty,
                    interpolation: interpolation.map(map_interpolation),
                    sampling: sampling.map(map_sampling),
                });
            }
        }
    }
}

fn map_interpolation(value: naga::Interpolation) -> ReflectedInterpolation {
    match value {
        naga::Interpolation::Perspective => ReflectedInterpolation::Perspective,
        naga::Interpolation::Linear => ReflectedInterpolation::Linear,
        naga::Interpolation::Flat => ReflectedInterpolation::Flat,
        naga::Interpolation::PerVertex => ReflectedInterpolation::PerVertex,
    }
}

fn map_sampling(value: naga::Sampling) -> ReflectedSampling {
    match value {
        naga::Sampling::Center => ReflectedSampling::Center,
        naga::Sampling::Centroid => ReflectedSampling::Centroid,
        naga::Sampling::Sample => ReflectedSampling::Sample,
        naga::Sampling::First => ReflectedSampling::First,
        naga::Sampling::Either => ReflectedSampling::Either,
    }
}

fn collect_output_builtins(
    module: &naga::Module,
    function: &naga::Function,
    builtins: &mut ReflectedFragmentBuiltins,
) {
    let Some(result) = &function.result else {
        return;
    };
    collect_binding_builtins(module, result.ty, result.binding.as_ref(), builtins);
}

fn collect_binding_builtins(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
    builtins: &mut ReflectedFragmentBuiltins,
) {
    if let Some(naga::Binding::BuiltIn(builtin)) = binding {
        mark_fragment_builtin(*builtin, builtins);
        return;
    }

    let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        return;
    };
    for member in members {
        if let Some(naga::Binding::BuiltIn(builtin)) = member.binding.as_ref() {
            mark_fragment_builtin(*builtin, builtins);
        }
    }
}

fn mark_fragment_builtin(builtin: naga::BuiltIn, builtins: &mut ReflectedFragmentBuiltins) {
    match builtin {
        naga::BuiltIn::FragDepth => builtins.frag_depth = true,
        naga::BuiltIn::SampleMask => builtins.sample_mask = true,
        _ => {}
    }
}

fn type_class(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Option<ReflectedTypeClass> {
    match &module.types.get_handle(ty).ok()?.inner {
        naga::TypeInner::Scalar(scalar) => scalar_class(*scalar).map(|scalar| ReflectedTypeClass {
            scalar: scalar.0,
            components: 1,
            width: scalar.1,
        }),
        naga::TypeInner::Vector { size, scalar } => {
            scalar_class(*scalar).map(|scalar| ReflectedTypeClass {
                scalar: scalar.0,
                components: vector_components(*size),
                width: scalar.1,
            })
        }
        _ => None,
    }
}

fn scalar_class(scalar: naga::Scalar) -> Option<(ReflectedTypeScalarClass, u8)> {
    match scalar.kind {
        naga::ScalarKind::Float => Some((ReflectedTypeScalarClass::Float, scalar.width)),
        naga::ScalarKind::Sint => Some((ReflectedTypeScalarClass::Sint, scalar.width)),
        naga::ScalarKind::Uint => Some((ReflectedTypeScalarClass::Uint, scalar.width)),
        naga::ScalarKind::Bool => Some((ReflectedTypeScalarClass::Bool, scalar.width)),
        _ => None,
    }
}

fn vector_components(size: naga::VectorSize) -> u8 {
    match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    }
}

fn resource_binding_kind(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    handle: naga::Handle<naga::GlobalVariable>,
    entry_index: Option<usize>,
) -> Option<ReflectedResourceBindingKind> {
    match global.space {
        naga::AddressSpace::Uniform => Some(ReflectedResourceBindingKind::Buffer(
            ReflectedBufferType::Uniform,
        )),
        naga::AddressSpace::Storage { access } => {
            let ty = if access.contains(naga::StorageAccess::STORE) {
                ReflectedBufferType::Storage
            } else {
                ReflectedBufferType::ReadOnlyStorage
            };
            Some(ReflectedResourceBindingKind::Buffer(ty))
        }
        naga::AddressSpace::Handle => match &module.types.get_handle(global.ty).ok()?.inner {
            naga::TypeInner::Sampler { comparison } => {
                Some(ReflectedResourceBindingKind::Sampler {
                    comparison: *comparison,
                })
            }
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => match class {
                naga::ImageClass::Sampled { kind, multi } => {
                    Some(ReflectedResourceBindingKind::Texture {
                        sampled: true,
                        sample_kind: scalar_kind_class(*kind),
                        sample_usage: sampled_texture_usage(module, handle, entry_index),
                        view_dimension: reflected_texture_view_dimension(*dim, *arrayed),
                        multisampled: *multi,
                    })
                }
                naga::ImageClass::Depth { multi } => Some(ReflectedResourceBindingKind::Texture {
                    sampled: false,
                    sample_kind: None,
                    sample_usage: sampled_texture_usage(module, handle, entry_index),
                    view_dimension: reflected_texture_view_dimension(*dim, *arrayed),
                    multisampled: *multi,
                }),
                #[cfg(feature = "tiled")]
                naga::ImageClass::Subpass {
                    aspect: naga::SubpassAspect::Color { kind },
                    multi,
                } => Some(ReflectedResourceBindingKind::InputAttachment {
                    sample_kind: scalar_kind_class(*kind)?,
                    multisampled: *multi,
                }),
                naga::ImageClass::Storage { format, access } => {
                    Some(ReflectedResourceBindingKind::StorageTexture {
                        format: format!("{format:?}"),
                        access: ReflectedStorageTextureAccess {
                            read: access.contains(naga::StorageAccess::LOAD),
                            write: access.contains(naga::StorageAccess::STORE),
                        },
                        view_dimension: reflected_texture_view_dimension(*dim, *arrayed),
                    })
                }
                naga::ImageClass::External => Some(ReflectedResourceBindingKind::ExternalTexture),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn reflected_texture_view_dimension(
    dim: naga::ImageDimension,
    arrayed: bool,
) -> ReflectedTextureViewDimension {
    match (dim, arrayed) {
        (naga::ImageDimension::D1, _) => ReflectedTextureViewDimension::D1,
        (naga::ImageDimension::D2, false) => ReflectedTextureViewDimension::D2,
        (naga::ImageDimension::D2, true) => ReflectedTextureViewDimension::D2Array,
        (naga::ImageDimension::Cube, false) => ReflectedTextureViewDimension::Cube,
        (naga::ImageDimension::Cube, true) => ReflectedTextureViewDimension::CubeArray,
        (naga::ImageDimension::D3, _) => ReflectedTextureViewDimension::D3,
    }
}

fn scalar_kind_class(kind: naga::ScalarKind) -> Option<ReflectedTypeScalarClass> {
    match kind {
        naga::ScalarKind::Float => Some(ReflectedTypeScalarClass::Float),
        naga::ScalarKind::Sint => Some(ReflectedTypeScalarClass::Sint),
        naga::ScalarKind::Uint => Some(ReflectedTypeScalarClass::Uint),
        naga::ScalarKind::Bool => Some(ReflectedTypeScalarClass::Bool),
        _ => None,
    }
}

fn resource_binding_min_size(
    layouter: &naga::proc::Layouter,
    global: &naga::GlobalVariable,
) -> u64 {
    match global.space {
        naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
            u64::from(layouter[global.ty].size)
        }
        _ => 0,
    }
}

fn override_default_value(
    module: &naga::Module,
    expression: naga::Handle<naga::Expression>,
) -> Option<ReflectedOverrideValue> {
    match module.global_expressions.try_get(expression).ok()? {
        naga::Expression::Literal(literal) => literal_value(*literal),
        naga::Expression::Constant(handle) => {
            let constant = module.constants.try_get(*handle).ok()?;
            override_default_value(module, constant.init)
        }
        _ => None,
    }
}

fn literal_value(literal: naga::Literal) -> Option<ReflectedOverrideValue> {
    match literal {
        naga::Literal::F64(value) => Some(ReflectedOverrideValue::Number(value)),
        naga::Literal::F32(value) => Some(ReflectedOverrideValue::Number(f64::from(value))),
        naga::Literal::F16(value) => {
            Some(ReflectedOverrideValue::Number(f64::from(value.to_f32())))
        }
        naga::Literal::I64(value) => Some(ReflectedOverrideValue::Number(value as f64)),
        naga::Literal::I32(value) => Some(ReflectedOverrideValue::Number(f64::from(value))),
        naga::Literal::U64(value) => Some(ReflectedOverrideValue::Number(value as f64)),
        naga::Literal::U32(value) => Some(ReflectedOverrideValue::Number(f64::from(value))),
        naga::Literal::Bool(value) => Some(ReflectedOverrideValue::Bool(value)),
        _ => None,
    }
}

fn sampled_texture_usage(
    module: &naga::Module,
    handle: naga::Handle<naga::GlobalVariable>,
    entry_index: Option<usize>,
) -> ReflectedTextureSampleUsage {
    if let Some(entry_index) = entry_index {
        entry_texture_sample_usage(module, entry_index, handle)
    } else {
        module
            .entry_points
            .iter()
            .enumerate()
            .map(|(index, _)| entry_texture_sample_usage(module, index, handle))
            .max_by_key(|usage| match usage {
                ReflectedTextureSampleUsage::Load => 0,
                ReflectedTextureSampleUsage::Sample => 1,
                ReflectedTextureSampleUsage::Gather => 2,
            })
            .unwrap_or(ReflectedTextureSampleUsage::Load)
    }
}

fn entry_texture_sample_usage(
    module: &naga::Module,
    entry_index: usize,
    handle: naga::Handle<naga::GlobalVariable>,
) -> ReflectedTextureSampleUsage {
    let Some(entry) = module.entry_points.get(entry_index) else {
        return ReflectedTextureSampleUsage::Load;
    };
    let mut usage = function_texture_sample_usage(&entry.function, handle);
    if usage == ReflectedTextureSampleUsage::Gather {
        return usage;
    }

    let mut reachable = std::collections::BTreeSet::new();
    collect_function_calls_from_block(&entry.function.body, &mut reachable);
    while let Some(function) = reachable.iter().copied().find(|function| {
        module
            .functions
            .try_get(*function)
            .is_ok_and(|function| !function_calls_collected(function, &reachable))
    }) {
        let Ok(function_ref) = module.functions.try_get(function) else {
            reachable.remove(&function);
            continue;
        };
        let before = reachable.len();
        collect_function_calls_from_block(&function_ref.body, &mut reachable);
        if before == reachable.len() {
            break;
        }
    }

    for function in reachable {
        let Ok(function) = module.functions.try_get(function) else {
            continue;
        };
        usage = max_texture_sample_usage(usage, function_texture_sample_usage(function, handle));
        if usage == ReflectedTextureSampleUsage::Gather {
            break;
        }
    }

    usage
}

fn max_texture_sample_usage(
    lhs: ReflectedTextureSampleUsage,
    rhs: ReflectedTextureSampleUsage,
) -> ReflectedTextureSampleUsage {
    match (lhs, rhs) {
        (ReflectedTextureSampleUsage::Gather, _) | (_, ReflectedTextureSampleUsage::Gather) => {
            ReflectedTextureSampleUsage::Gather
        }
        (ReflectedTextureSampleUsage::Sample, _) | (_, ReflectedTextureSampleUsage::Sample) => {
            ReflectedTextureSampleUsage::Sample
        }
        _ => ReflectedTextureSampleUsage::Load,
    }
}

fn function_calls_collected(
    function: &naga::Function,
    reachable: &std::collections::BTreeSet<naga::Handle<naga::Function>>,
) -> bool {
    let mut calls = std::collections::BTreeSet::new();
    collect_function_calls_from_block(&function.body, &mut calls);
    calls.into_iter().all(|call| reachable.contains(&call))
}

fn collect_function_calls_from_block(
    block: &naga::Block,
    calls: &mut std::collections::BTreeSet<naga::Handle<naga::Function>>,
) {
    for statement in block {
        match statement {
            naga::Statement::Block(block) => collect_function_calls_from_block(block, calls),
            naga::Statement::If { accept, reject, .. } => {
                collect_function_calls_from_block(accept, calls);
                collect_function_calls_from_block(reject, calls);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_function_calls_from_block(&case.body, calls);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_function_calls_from_block(body, calls);
                collect_function_calls_from_block(continuing, calls);
            }
            naga::Statement::Call { function, .. } => {
                calls.insert(*function);
            }
            _ => {}
        }
    }
}

fn function_texture_sample_usage(
    function: &naga::Function,
    handle: naga::Handle<naga::GlobalVariable>,
) -> ReflectedTextureSampleUsage {
    function
        .expressions
        .iter()
        .filter_map(|(_, expression)| {
            let naga::Expression::ImageSample { image, gather, .. } = expression else {
                return None;
            };
            (expression_global(function, *image) == Some(handle)).then_some(if gather.is_some() {
                ReflectedTextureSampleUsage::Gather
            } else {
                ReflectedTextureSampleUsage::Sample
            })
        })
        .fold(ReflectedTextureSampleUsage::Load, max_texture_sample_usage)
}

fn expression_global(
    function: &naga::Function,
    expression: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    match function.expressions.try_get(expression).ok()? {
        naga::Expression::GlobalVariable(handle) => Some(*handle),
        naga::Expression::Access { base, .. }
        | naga::Expression::AccessIndex { base, .. }
        | naga::Expression::Load { pointer: base } => expression_global(function, *base),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_and_validate_wgsl, MslBindingMap, MslResourceBinding, MslResourceBindingKind,
        MslVertexAttribute, MslVertexBufferBinding, MslVertexFormat, MslVertexStepMode,
        ReflectedBufferType, ReflectedResourceBindingKind, ReflectedShaderStage,
        ReflectedTextureSampleUsage, ReflectedTextureViewDimension, ReflectedTypeScalarClass,
    };
    use naga::ShaderStage;

    #[test]
    fn parses_and_validates_trivial_wgsl() {
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        assert!(parse_and_validate_wgsl(source).is_ok());
    }

    #[test]
    fn counts_inter_stage_input_builtins() {
        // F-063: front_facing + sample_index + sample_mask each consume an
        // inter-stage variable slot; position does not.
        let module = parse_and_validate_wgsl(
            r"
@fragment fn fs(
    @builtin(position) pos: vec4f,
    @builtin(front_facing) ff: bool,
    @builtin(sample_index) si: u32,
    @builtin(sample_mask) sm: u32,
    @location(0) uv: vec2f,
) -> @location(0) vec4f {
    return vec4f(uv, 0.0, 1.0);
}
",
        )
        .expect("fragment with builtin inputs should validate");
        let io = module
            .entry_point_io()
            .into_iter()
            .find(|io| io.entry_point == "fs")
            .expect("fs entry point");
        assert_eq!(io.input_inter_stage_builtins, 3);
        assert_eq!(io.inputs.len(), 1);
    }

    #[test]
    fn validates_cube_array_and_multisampled_sampled_textures() {
        // F-057: a float cube-array sampled texture is a core WebGPU type; naga
        // gates the `Cube` + arrayed image type behind `CUBE_ARRAY_TEXTURES`, so
        // dropping that capability turned every cube-array shader into an error
        // module (masked for sint/uint, whose filtering-sampler cases the CTS
        // expects to fail anyway, but surfaced for the float case it expects to
        // succeed). Both capabilities are WebGPU baseline.
        for source in [
            r"
@group(0) @binding(0) var t: texture_cube_array<f32>;
@group(0) @binding(1) var s: sampler;
@compute @workgroup_size(1) fn cs() {
    _ = textureGather(0, t, s, vec3f(0), 0);
}
",
            r"
@group(0) @binding(0) var t: texture_cube_array<u32>;
@group(0) @binding(1) var s: sampler;
@compute @workgroup_size(1) fn cs() {
    _ = textureGather(0, t, s, vec3f(0), 0);
}
",
            r"
@group(0) @binding(0) var t: texture_multisampled_2d<f32>;
@compute @workgroup_size(1) fn cs() {
    _ = textureLoad(t, vec2i(0), 0);
}
",
        ] {
            assert!(
                parse_and_validate_wgsl(source).is_ok(),
                "expected shader to validate: {source}"
            );
        }
    }

    #[test]
    fn generate_msl_carries_buffer_sizes_for_runtime_storage_array() {
        let module = parse_and_validate_wgsl(
            r"
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>;

@compute @workgroup_size(1)
fn cs() {
    let value = textureLoad(tex, vec2<i32>(0, 0));
    outputBuffer[0] = u32(value.r * 255.0);
}
",
        )
        .expect("runtime-array storage texture shader should validate");
        let generated = module
            .generate_msl(
                "cs",
                &MslBindingMap {
                    resources: vec![
                        MslResourceBinding {
                            group: 0,
                            binding: 0,
                            metal_index: 0,
                            kind: MslResourceBindingKind::Texture,
                        },
                        MslResourceBinding {
                            group: 0,
                            binding: 1,
                            metal_index: 1,
                            kind: MslResourceBindingKind::Buffer,
                        },
                    ],
                },
                &naga::back::PipelineConstants::default(),
            )
            .expect("MSL generation should provide a sizes buffer slot");

        assert_eq!(generated.buffer_sizes_slot, Some(2));
        assert_eq!(
            generated.buffer_size_bindings,
            [super::MslBufferSizeBinding {
                group: 0,
                binding: 1
            }]
        );
        assert!(generated.source.contains("_mslBufferSizes"));
    }

    #[test]
    fn msl_buffer_sizes_cover_all_runtime_arrays_in_module_order() {
        let module = parse_and_validate_wgsl(
            r"
@group(0) @binding(0) var<storage, read_write> unusedFirst: array<u32>;
@group(0) @binding(1) var<storage, read_write> usedSecond: array<u32>;

@compute @workgroup_size(1)
fn uses_second() {
    usedSecond[0] = 1u;
}

@compute @workgroup_size(1)
fn uses_first() {
    unusedFirst[0] = 1u;
}
",
        )
        .expect("multi-entry runtime-array shader should validate");

        let bindings = module
            .msl_buffer_size_bindings_for_entry("uses_second")
            .expect("entry should reflect MSL buffer sizes");

        assert_eq!(
            bindings,
            [
                super::MslBufferSizeBinding {
                    group: 0,
                    binding: 0
                },
                super::MslBufferSizeBinding {
                    group: 0,
                    binding: 1
                },
            ]
        );
    }

    #[test]
    fn render_msl_buffer_sizes_slot_avoids_vertex_buffer_slots() {
        let module = parse_and_validate_wgsl(
            r"
@group(0) @binding(0) var<storage, read> data: array<vec4<f32>>;

@vertex
fn vs(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
    return data[0] + pos;
}
",
        )
        .expect("vertex runtime-array shader should validate");
        let generated = module
            .generate_render_vertex_msl(
                "vs",
                &MslBindingMap {
                    resources: vec![MslResourceBinding {
                        group: 0,
                        binding: 0,
                        metal_index: 0,
                        kind: MslResourceBindingKind::Buffer,
                    }],
                },
                &[MslVertexBufferBinding {
                    slot: 0,
                    metal_index: 2,
                    array_stride: 16,
                    step_mode: MslVertexStepMode::Vertex,
                    attributes: vec![MslVertexAttribute {
                        shader_location: 0,
                        offset: 0,
                        format: MslVertexFormat::Float32x4,
                    }],
                }],
                false,
                &naga::back::PipelineConstants::default(),
            )
            .expect("MSL vertex generation should provide a non-colliding sizes slot");

        assert_eq!(generated.buffer_sizes_slot, Some(3));
    }

    #[test]
    fn rejects_invalid_wgsl() {
        assert!(parse_and_validate_wgsl("not wgsl @@@").is_err());
    }

    #[test]
    fn reflects_entry_points() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() {}
             @compute @workgroup_size(1) fn cs() {}",
        )
        .unwrap();

        let entry_points = module.entry_points();
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "vs" && entry.stage == ReflectedShaderStage::Vertex));
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "fs" && entry.stage == ReflectedShaderStage::Fragment));
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "cs" && entry.stage == ReflectedShaderStage::Compute));
    }

    #[test]
    fn reflects_compute_workgroup_size_and_storage() {
        let module = parse_and_validate_wgsl(
            "var<workgroup> scratch: array<u32, 4>;
             @compute @workgroup_size(8, 4, 1) fn cs() {
                 scratch[0] = 1u;
             }",
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [8, 4, 1]);
        assert_eq!(reflected.override_keys, [None, None, None]);
        assert_eq!(reflected.workgroup_storage_size, 16);
    }

    #[test]
    fn reflects_override_driven_workgroup_size() {
        let module = parse_and_validate_wgsl(
            "override wg_x: u32 = 8u;
             @compute @workgroup_size(wg_x, 1, 1) fn cs() {}",
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [1, 1, 1]);
        assert_eq!(
            reflected.override_keys[0].as_ref().unwrap().name.as_deref(),
            Some("wg_x")
        );
    }

    #[test]
    fn reflects_vertex_fragment_io() {
        let module = parse_and_validate_wgsl(
            "struct VsOut {
                 @builtin(position) pos: vec4<f32>,
                 @location(1) color: vec4<f32>,
             }
             @vertex fn vs(@location(0) a: vec3<f32>) -> VsOut {
                 return VsOut(vec4<f32>(0.0), vec4<f32>(a, 1.0));
             }
             @fragment fn fs(@location(1) color: vec4<f32>) -> @location(0) vec4<f32> {
                 return color;
             }",
        )
        .unwrap();

        let io = module.entry_point_io();
        let vs = io.iter().find(|entry| entry.entry_point == "vs").unwrap();
        assert_eq!(vs.inputs[0].location, 0);
        assert_eq!(vs.inputs[0].ty.scalar, ReflectedTypeScalarClass::Float);
        assert_eq!(vs.inputs[0].ty.components, 3);
        assert_eq!(vs.outputs[0].location, 1);

        let fs = io.iter().find(|entry| entry.entry_point == "fs").unwrap();
        assert_eq!(fs.inputs[0].location, 1);
        assert_eq!(fs.outputs[0].location, 0);
    }

    #[test]
    fn reflects_resource_bindings_and_static_use() {
        let module = parse_and_validate_wgsl(
            "struct U { value: vec4<f32> }
             @group(0) @binding(0) var<uniform> u: U;
             @group(0) @binding(1) var samp: sampler;
             @group(0) @binding(2) var tex: texture_2d<f32>;
             @group(0) @binding(3) var unused_tex: texture_2d<f32>;
             @fragment fn fs() -> @location(0) vec4<f32> {
                 return textureSample(tex, samp, vec2<f32>(0.5)) + u.value;
             }",
        )
        .unwrap();

        let bindings = module.resource_bindings();
        let uniform = bindings
            .iter()
            .find(|binding| binding.binding == 0)
            .unwrap();
        assert_eq!(
            uniform.kind,
            ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Uniform)
        );
        assert!(uniform.statically_used);

        let texture = bindings
            .iter()
            .find(|binding| binding.binding == 2)
            .unwrap();
        assert_eq!(
            texture.kind,
            ReflectedResourceBindingKind::Texture {
                sampled: true,
                sample_kind: Some(ReflectedTypeScalarClass::Float),
                sample_usage: ReflectedTextureSampleUsage::Sample,
                view_dimension: ReflectedTextureViewDimension::D2,
                multisampled: false
            }
        );
        assert!(texture.statically_used);

        let unused = bindings
            .iter()
            .find(|binding| binding.binding == 3)
            .unwrap();
        assert!(!unused.statically_used);
    }

    #[test]
    fn external_texture_validates_reflects_and_generates_backend_sources() {
        let module = parse_and_validate_wgsl(
            "@group(0) @binding(0) var tex: texture_external;
             @fragment fn fs() -> @location(0) vec4<f32> {
                 return textureLoad(tex, vec2<i32>(0, 0));
             }",
        )
        .expect("external texture WGSL should validate");

        let bindings = module.resource_bindings_for_entry("fs").unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].kind,
            ReflectedResourceBindingKind::ExternalTexture
        );
        assert!(bindings[0].statically_used);

        let msl = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: vec![MslResourceBinding {
                        group: 0,
                        binding: 0,
                        metal_index: 0,
                        kind: MslResourceBindingKind::ExternalTexture,
                    }],
                },
                &[],
                &naga::back::PipelineConstants::default(),
                u32::MAX,
            )
            .expect("external texture fragment MSL should generate");
        assert!(msl.source.contains("_plane0"));
        assert!(msl.source.contains("_params"));

        // SPIR-V (Vulkan): external textures are unsupported. Must return a clean
        // error (never panic, never silently emit a wrong shader), matching wgpu
        // which also leaves Vulkan external textures unimplemented.
        let spirv_err = module
            .generate_spirv(
                "fs",
                ShaderStage::Fragment,
                &naga::back::PipelineConstants::default(),
            )
            .expect_err("external texture SPIR-V must be cleanly rejected, not generated");
        assert!(spirv_err.contains("external textures are not supported on the Vulkan backend"));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn reflects_subpass_input_binding_kind() {
        let module = parse_and_validate_wgsl(
            "@group(0) @binding(0) var s: subpass_input<i32>;
             @fragment fn fs() -> @location(0) vec4<i32> {
                 return subpassLoad(s);
             }",
        )
        .unwrap();

        let bindings = module.resource_bindings_for_entry("fs").unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].kind,
            ReflectedResourceBindingKind::InputAttachment {
                sample_kind: ReflectedTypeScalarClass::Sint,
                multisampled: false
            }
        );
    }

    #[test]
    fn reflects_fragment_builtin_outputs() {
        let module = parse_and_validate_wgsl(
            "struct Out {
                 @builtin(frag_depth) depth: f32,
                 @builtin(sample_mask) mask: u32,
             }
             @fragment fn fs() -> Out {
                 return Out(0.5, 1u);
             }",
        )
        .unwrap();

        let builtins = module.fragment_builtins();
        assert!(builtins[0].frag_depth);
        assert!(builtins[0].sample_mask);
    }

    #[test]
    fn reflects_overrides_and_accepts_f16_override() {
        let module = parse_and_validate_wgsl(
            "enable f16;
             override half_value: f16;
             @id(7) override int_value: i32 = 3;
             @compute @workgroup_size(1) fn cs() {}",
        )
        .unwrap();

        let overrides = module.overrides();
        let half = overrides
            .iter()
            .find(|override_| override_.name.as_deref() == Some("half_value"))
            .unwrap();
        assert_eq!(half.ty.scalar, ReflectedTypeScalarClass::Float);
        assert!(!half.has_default);

        let int = overrides
            .iter()
            .find(|override_| override_.id == Some(7))
            .unwrap();
        assert_eq!(int.ty.scalar, ReflectedTypeScalarClass::Sint);
        assert!(int.has_default);
    }

    #[test]
    fn generate_render_msl_accepts_vertex_only_entry() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let generated = module
            .generate_render_msl(
                "vs",
                None,
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &[],
                false,
            )
            .expect("vertex-only render MSL should generate");

        assert!(generated.source.contains("vertex"));
        assert!(!generated.vertex_entry_point.is_empty());
        assert_eq!(generated.fragment_entry_point, None);
    }

    #[test]
    fn generate_render_vertex_msl_emits_single_vertex_stage() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let generated = module
            .generate_render_vertex_msl(
                "vs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                false,
                &naga::back::PipelineConstants::default(),
            )
            .expect("render vertex MSL should generate");

        assert!(generated.source.contains("vertex"));
        assert!(!generated.entry_point.is_empty());
    }

    #[test]
    fn generate_render_msl_forces_point_size_only_when_requested() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();
        let binding_map = MslBindingMap {
            resources: Vec::new(),
        };

        let point = module
            .generate_render_msl("vs", None, &binding_map, &[], &[], true)
            .expect("point render MSL should generate");
        let triangle = module
            .generate_render_msl("vs", None, &binding_map, &[], &[], false)
            .expect("triangle render MSL should generate");

        assert!(point.source.contains("point_size"));
        assert!(!triangle.source.contains("point_size"));
    }

    #[test]
    fn generate_render_fragment_msl_emits_single_fragment_stage() {
        let module = parse_and_validate_wgsl(
            "@fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(0.0, 1.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let generated = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &naga::back::PipelineConstants::default(),
                u32::MAX,
            )
            .expect("render fragment MSL should generate");

        assert!(generated.source.contains("fragment"));
        assert!(!generated.entry_point.is_empty());
        assert_eq!(generated.frag_depth_clamp_slot, None);
        assert!(!generated.source.contains("metal::clamp"));
    }

    #[test]
    fn generate_render_fragment_msl_clamps_frag_depth_output() {
        let module = parse_and_validate_wgsl(
            "struct Out {
                 @location(0) color: vec4<f32>,
                 @builtin(frag_depth) depth: f32,
             }

             @fragment fn fs() -> Out {
                 return Out(vec4<f32>(0.0, 1.0, 0.0, 1.0), 2.0);
             }",
        )
        .unwrap();

        let generated = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &naga::back::PipelineConstants::default(),
                u32::MAX,
            )
            .expect("frag-depth fragment MSL should generate");

        assert!(generated.source.contains("metal::clamp"));
        assert!(generated.source.contains("naga_frag_depth_clamp"));
        assert!(generated.frag_depth_clamp_slot.is_some());
    }

    #[test]
    fn generate_render_fragment_msl_applies_override_default_and_pipeline_value() {
        let module = override_fragment_module();
        let default = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &naga::back::PipelineConstants::default(),
                u32::MAX,
            )
            .expect("default override should generate MSL");
        let provided = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &override_constants("R", 0.6),
                u32::MAX,
            )
            .expect("provided override should generate MSL");

        assert!(default.source.contains("1.0"));
        assert!(provided.source.contains("0.6"));
        assert_ne!(default.source, provided.source);
        assert!(!provided.source.contains("override"));
    }

    #[test]
    fn generate_render_fragment_msl_applies_sample_mask() {
        let module = parse_and_validate_wgsl(
            "struct Out {
                 @location(0) color: vec4<f32>,
                 @builtin(sample_mask) mask: u32,
             }

             @fragment fn fs() -> Out {
                 return Out(vec4<f32>(0.0, 1.0, 0.0, 1.0), 0xffffffffu);
             }",
        )
        .unwrap();
        let default = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &naga::back::PipelineConstants::default(),
                u32::MAX,
            )
            .expect("default sample mask fragment MSL should generate");
        let masked = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &naga::back::PipelineConstants::default(),
                0b0101,
            )
            .expect("masked fragment MSL should generate");

        assert!(default.source.contains("sample_mask"));
        assert!(masked.source.contains("sample_mask"));
        assert_ne!(default.source, masked.source);
    }

    #[test]
    fn generate_spirv_applies_override_default_and_pipeline_value() {
        let module = override_fragment_module();
        let default = module
            .generate_spirv(
                "fs",
                ShaderStage::Fragment,
                &naga::back::PipelineConstants::default(),
            )
            .expect("default override should generate SPIR-V");
        let provided = module
            .generate_spirv("fs", ShaderStage::Fragment, &override_constants("R", 0.6))
            .expect("provided override should generate SPIR-V");

        assert!(!default.is_empty());
        assert!(!provided.is_empty());
        assert_ne!(default, provided);
    }

    fn override_fragment_module() -> super::ReflectedModule {
        parse_and_validate_wgsl(
            "
override R: f32 = 1.0;

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(R, 0.25, 0.5, 0.75);
}
",
        )
        .expect("override fragment shader should validate")
    }

    fn override_constants(key: &str, value: f64) -> naga::back::PipelineConstants {
        [(key.to_owned(), value)].into_iter().collect()
    }

    #[test]
    fn generate_spirv_accepts_vertex_only_entry() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let spirv = module
            .generate_spirv(
                "vs",
                ShaderStage::Vertex,
                &naga::back::PipelineConstants::default(),
            )
            .expect("vertex-only SPIR-V should generate");

        assert!(!spirv.is_empty());
    }

    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_emits_vertex_main() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let generated = module
            .generate_glsl(
                "vs",
                ShaderStage::Vertex,
                &naga::back::PipelineConstants::default(),
            )
            .expect("vertex GLSL generation should succeed");

        assert!(generated.source.contains("#version 310 es"));
        assert!(generated.source.contains("void main()"));
    }

    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_emits_fragment_main() {
        let module = parse_and_validate_wgsl(
            "@fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }",
        )
        .unwrap();

        let generated = module
            .generate_glsl(
                "fs",
                ShaderStage::Fragment,
                &naga::back::PipelineConstants::default(),
            )
            .expect("fragment GLSL generation should succeed");

        assert!(generated.source.contains("#version 310 es"));
        assert!(generated.source.contains("void main()"));
    }

    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_applies_override_default_and_pipeline_value() {
        let module = override_fragment_module();
        let default = module
            .generate_glsl(
                "fs",
                ShaderStage::Fragment,
                &naga::back::PipelineConstants::default(),
            )
            .expect("default override should generate GLSL");
        let provided = module
            .generate_glsl("fs", ShaderStage::Fragment, &override_constants("R", 0.6))
            .expect("provided override should generate GLSL");

        assert!(default.source.contains("1.0"));
        assert!(provided.source.contains("0.6"));
        assert_ne!(default.source, provided.source);
        assert!(!provided.source.contains("override"));
    }

    // --- F-069: workgroup memory sizes ---

    fn empty_binding_map() -> MslBindingMap {
        MslBindingMap {
            resources: Vec::new(),
        }
    }

    /// A compute shader with two var<workgroup> globals:
    ///
    /// - `a: array<u32, 7>` → raw size 28 bytes → aligned to 32
    /// - `b: f32`           → raw size  4 bytes → aligned to 16
    ///
    /// Globals are collected in module declaration order (naga arena order),
    /// which matches the `[[threadgroup(N)]]` assignment order in the MSL emitter.
    #[test]
    fn generate_msl_returns_workgroup_memory_sizes_for_workgroup_vars() {
        let module = parse_and_validate_wgsl(
            r"
var<workgroup> a: array<u32, 7>;
var<workgroup> b: f32;

@compute @workgroup_size(1)
fn cs() {
    a[0] = 1u;
    b = 2.0;
}
",
        )
        .expect("workgroup shader should validate");

        let generated = module
            .generate_msl("cs", &empty_binding_map(), &naga::back::PipelineConstants::default())
            .expect("MSL generation should succeed");

        // array<u32, 7> = 7 * 4 = 28 bytes → next_multiple_of(16) = 32
        // f32 = 4 bytes → next_multiple_of(16) = 16
        assert_eq!(
            generated.workgroup_memory_sizes,
            vec![32, 16],
            "workgroup memory sizes must be rounded up to multiples of 16"
        );
    }

    /// A compute shader with no workgroup vars returns an empty vec.
    #[test]
    fn generate_msl_returns_empty_workgroup_memory_sizes_when_no_workgroup_vars() {
        let module = parse_and_validate_wgsl(
            r"
@compute @workgroup_size(1)
fn cs() {}
",
        )
        .expect("trivial compute shader should validate");

        let generated = module
            .generate_msl("cs", &empty_binding_map(), &naga::back::PipelineConstants::default())
            .expect("MSL generation should succeed");

        assert!(
            generated.workgroup_memory_sizes.is_empty(),
            "shader without workgroup vars must return empty workgroup_memory_sizes"
        );
    }

    /// Render (vertex) MSL generation does not produce workgroup_memory_sizes —
    /// the field is always empty for render stages.
    #[test]
    fn generate_render_msl_returns_empty_workgroup_memory_sizes() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }",
        )
        .expect("vertex shader should validate");

        let generated = module
            .generate_render_vertex_msl(
                "vs",
                &empty_binding_map(),
                &[],
                false,
                &naga::back::PipelineConstants::default(),
            )
            .expect("render vertex MSL should succeed");

        assert!(
            generated.workgroup_memory_sizes.is_empty(),
            "render stages must always return empty workgroup_memory_sizes"
        );
    }
}
