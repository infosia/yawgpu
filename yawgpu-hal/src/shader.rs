/// Enumerates HAL shader source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalShaderSource {
    /// Msl variant.
    Msl(String),
    /// MSL source with immediate-block and buffer-size metadata.
    MslWithBufferSizes {
        /// MSL source.
        source: String,
        /// Bindings whose byte lengths populate the buffer sizes in the
        /// immediate block.
        buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Per-argument threadgroup memory allocation sizes (bytes, already
        /// rounded up to a multiple of 16) for compute shaders that use
        /// `var<workgroup>` globals.  The Metal HAL calls
        /// `setThreadgroupMemoryLength:atIndex:` for each entry before dispatch.
        /// Empty when the compute shader has no workgroup variables.
        workgroup_memory_sizes: Vec<u32>,
        /// Compute-stage immediate block delivery metadata, when the entry
        /// point needs the block (`var<immediate>` user data or buffer
        /// sizes). `None` when it needs neither.
        immediates: Option<HalMslImmediates>,
    },
    /// Per-stage MSL render sources.
    MslStages {
        /// Vertex stage MSL source.
        vertex: String,
        /// Optional fragment stage MSL source.
        fragment: Option<String>,
    },
    /// Per-stage MSL render sources with immediate-block and buffer-size
    /// metadata.
    MslStagesWithBufferSizes {
        /// Vertex stage MSL source.
        vertex: String,
        /// Optional fragment stage MSL source.
        fragment: Option<String>,
        /// Vertex-stage bindings whose byte lengths populate the buffer sizes
        /// in the vertex immediate block.
        vertex_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Fragment-stage bindings whose byte lengths populate the buffer
        /// sizes in the fragment immediate block.
        fragment_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Vertex-stage immediate block delivery metadata, when the vertex
        /// entry point needs the block (`var<immediate>` user data, storage
        /// sizes, or vertex-pulling sizes). `None` when it needs none.
        vertex_immediates: Option<HalMslImmediates>,
        /// Fragment-stage immediate block delivery metadata. Also carries
        /// the frag-depth clamp range offset when this pipeline clamps
        /// frag_depth. `None` when the fragment entry point needs no block.
        fragment_immediates: Option<HalMslImmediates>,
        /// Metal buffer indices for vertex buffers, in the same order as
        /// `vertex_buffer_mappings` passed to Tint's MSL codegen. Their sizes
        /// follow the storage sizes in the vertex immediate block; the HAL
        /// encoder writes the effective byte size (buffer.size − bind_offset)
        /// of each bound vertex buffer so Tint's vertex-pulling OOB guards
        /// compare against real data.
        vertex_buffer_metal_indices: Vec<u32>,
    },
    /// Spir v variant.
    SpirV(Vec<u32>),
    /// Spir vstages variant.
    SpirVStages {
        /// Vertex variant.
        vertex: Vec<u32>,
        /// Optional fragment variant.
        fragment: Option<Vec<u32>>,
    },
    /// GLSL render stages.
    GlslStages {
        /// Vertex stage GLSL ES source.
        vertex: String,
        /// Optional fragment stage GLSL ES source.
        fragment: Option<String>,
        /// Combined texture/sampler uniforms emitted by Tint's GLSL writer.
        combined_samplers: Vec<HalCombinedSampler>,
        /// Texture metadata UBO slots emitted by Tint's GLSL writer.
        texture_metadata_slots: Vec<HalTextureMetadataSlot>,
        /// GLES flat resource binding remaps emitted alongside the GLSL.
        binding_remaps: Vec<HalGlesBindingRemap>,
        /// Uniform-buffer binding carrying GLSL texture metadata, if emitted.
        texture_metadata_ubo_binding: Option<u32>,
    },
    /// Glsl variant.
    Glsl {
        /// Source variant.
        source: String,
        /// Stage variant.
        stage: HalShaderStage,
        /// Combined texture/sampler uniforms emitted by Tint's GLSL writer.
        combined_samplers: Vec<HalCombinedSampler>,
        /// Texture metadata UBO slots emitted by Tint's GLSL writer.
        texture_metadata_slots: Vec<HalTextureMetadataSlot>,
        /// GLES flat resource binding remaps emitted alongside the GLSL.
        binding_remaps: Vec<HalGlesBindingRemap>,
        /// Uniform-buffer binding carrying GLSL texture metadata, if emitted.
        texture_metadata_ubo_binding: Option<u32>,
    },
}

/// A GLSL combined texture/sampler uniform and the WGSL bindings it represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HalCombinedSampler {
    /// GLSL sampler uniform name.
    pub glsl_uniform_name: String,
    /// WGSL bind group of the texture.
    pub texture_group: u32,
    /// WGSL binding number of the texture.
    pub texture_binding: u32,
    /// WGSL bind group of the sampler, or Tint's placeholder sampler sentinel.
    pub sampler_group: u32,
    /// WGSL binding number of the sampler, or Tint's placeholder sampler sentinel.
    pub sampler_binding: u32,
    /// Whether this record uses Tint's placeholder sampler for a texture-only use.
    pub uses_placeholder_sampler: bool,
}

/// A texture metadata UBO slot and the WGSL texture binding that populates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalTextureMetadataSlot {
    /// Slot index into Tint's packed u32 metadata UBO.
    pub offset: u32,
    /// WGSL bind group of the queried texture.
    pub texture_group: u32,
    /// WGSL binding number of the queried texture.
    pub texture_binding: u32,
}

/// GLES resource classes whose GLSL layout binding is consumed by the HAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HalGlesBindingClass {
    /// Uniform-buffer binding.
    UniformBuffer,
    /// Storage-buffer binding.
    StorageBuffer,
    /// Storage-texture image binding.
    StorageTexture,
}

/// Maps one WGSL binding point to its flat GLES binding number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct HalGlesBindingRemap {
    /// WGSL bind group.
    pub group: u32,
    /// WGSL binding number.
    pub binding: u32,
    /// Resource class.
    pub class: HalGlesBindingClass,
    /// Flat GLES binding number used in GLSL and GL bind calls.
    pub flat_binding: u32,
}

impl HalGlesBindingRemap {
    /// Creates a new GLES binding remap entry.
    #[must_use]
    pub fn new(group: u32, binding: u32, class: HalGlesBindingClass, flat_binding: u32) -> Self {
        Self {
            group,
            binding,
            class,
            flat_binding,
        }
    }
}

/// Stores one MSL buffer-size entry binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct HalMslBufferSizeBinding {
    /// Bind group index.
    pub group: u32,
    /// Binding index.
    pub binding: u32,
}

impl HalMslBufferSizeBinding {
    /// Creates a new MSL buffer-size binding entry.
    #[must_use]
    pub fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }
}

/// Metal immediates delivery descriptor for one shader stage: where to bind
/// the combined immediate block and how it is laid out. Mirrors Dawn's
/// `ImmediatesLayoutMTL.h`: user immediate bytes at `[0, user_size)`, then
/// the frag-depth clamp range when present, a non-constant-zero word that is
/// always 0, and the buffer sizes when present. The offsets come from
/// `yawgpu-tint`, which computes the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct HalMslImmediates {
    /// Metal buffer slot the combined block is delivered to via
    /// `set{Vertex,Fragment}Bytes` / `setBytes`.
    pub slot: u32,
    /// Total block size in bytes delivered to this stage.
    pub block_size: u32,
    /// Bytes of user immediates at the start of the block: the pipeline
    /// layout's reserved user-immediate budget.
    pub user_size: u32,
    /// Byte offset of the 8-byte frag-depth clamp range
    /// (`[min_depth, max_depth]`, both `f32`) within the block, when this
    /// stage clamps frag_depth. Always `None` for vertex and compute
    /// stages.
    pub frag_depth_clamp_offset: Option<u32>,
    /// Byte offset of the `array<u32, N>` buffer sizes within the block,
    /// when this stage has any (`N > 0`).
    pub buffer_sizes_offset: Option<u32>,
}

impl HalMslImmediates {
    /// Creates a new Metal immediates delivery descriptor.
    #[must_use]
    pub fn new(
        slot: u32,
        block_size: u32,
        user_size: u32,
        frag_depth_clamp_offset: Option<u32>,
        buffer_sizes_offset: Option<u32>,
    ) -> Self {
        Self {
            slot,
            block_size,
            user_size,
            frag_depth_clamp_offset,
            buffer_sizes_offset,
        }
    }
}

/// Enumerates shader stages for stage-specific source formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalShaderStage {
    /// Vertex variant.
    Vertex,
    /// Fragment variant.
    Fragment,
    /// Compute variant.
    Compute,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_shader_source_msl_stages_round_trips_sources() {
        let source = HalShaderSource::MslStages {
            vertex: "vertex".to_owned(),
            fragment: Some("fragment".to_owned()),
        };

        assert!(matches!(
            source,
            HalShaderSource::MslStages { vertex, fragment }
                if vertex == "vertex" && fragment.as_deref() == Some("fragment")
        ));
    }

    #[test]
    fn hal_shader_source_msl_buffer_size_metadata_round_trips() {
        let source = HalShaderSource::MslWithBufferSizes {
            source: "kernel void main0() {}".to_owned(),
            buffer_size_bindings: vec![HalMslBufferSizeBinding::new(1, 2)],
            workgroup_memory_sizes: vec![32, 16],
            immediates: Some(HalMslImmediates::new(4, 16, 0, None, Some(4))),
        };

        assert!(matches!(
            source,
            HalShaderSource::MslWithBufferSizes {
                source,
                buffer_size_bindings,
                workgroup_memory_sizes,
                immediates: Some(HalMslImmediates {
                    slot: 4,
                    block_size: 16,
                    user_size: 0,
                    frag_depth_clamp_offset: None,
                    buffer_sizes_offset: Some(4),
                }),
            } if source == "kernel void main0() {}"
                && buffer_size_bindings == [HalMslBufferSizeBinding::new(1, 2)]
                && workgroup_memory_sizes == [32, 16]
        ));
    }

    #[test]
    fn hal_shader_source_msl_stages_with_buffer_sizes_includes_vertex_buffer_metal_indices() {
        let source = HalShaderSource::MslStagesWithBufferSizes {
            vertex: "vertex_src".to_owned(),
            fragment: Some("fragment_src".to_owned()),
            vertex_buffer_size_bindings: vec![HalMslBufferSizeBinding::new(0, 1)],
            fragment_buffer_size_bindings: Vec::new(),
            vertex_immediates: None,
            fragment_immediates: Some(HalMslImmediates::new(30, 80, 64, Some(64), None)),
            vertex_buffer_metal_indices: vec![3, 5],
        };

        assert!(matches!(
            source,
            HalShaderSource::MslStagesWithBufferSizes {
                vertex,
                fragment,
                vertex_buffer_size_bindings,
                fragment_buffer_size_bindings,
                vertex_immediates: None,
                fragment_immediates: Some(HalMslImmediates {
                    slot: 30,
                    block_size: 80,
                    user_size: 64,
                    frag_depth_clamp_offset: Some(64),
                    buffer_sizes_offset: None,
                }),
                vertex_buffer_metal_indices,
            } if vertex == "vertex_src"
                && fragment.as_deref() == Some("fragment_src")
                && vertex_buffer_size_bindings == [HalMslBufferSizeBinding::new(0, 1)]
                && fragment_buffer_size_bindings.is_empty()
                && vertex_buffer_metal_indices == [3, 5]
        ));
    }

    #[test]
    fn hal_msl_immediates_new_round_trips_fields() {
        let immediates = HalMslImmediates::new(30, 96, 64, Some(64), Some(76));

        assert_eq!(
            immediates,
            HalMslImmediates {
                slot: 30,
                block_size: 96,
                user_size: 64,
                frag_depth_clamp_offset: Some(64),
                buffer_sizes_offset: Some(76),
            }
        );
    }
}
