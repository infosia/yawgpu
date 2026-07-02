/// Enumerates HAL shader source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalShaderSource {
    /// Msl variant.
    Msl(String),
    /// MSL source with `_mslBufferSizes` binding metadata.
    MslWithBufferSizes {
        /// MSL source.
        source: String,
        /// Reserved buffer slot for `_mslBufferSizes`.
        buffer_sizes_slot: Option<u32>,
        /// Bindings whose byte lengths populate `_mslBufferSizes`.
        buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Per-argument threadgroup memory allocation sizes (bytes, already
        /// rounded up to a multiple of 16) for compute shaders that use
        /// `var<workgroup>` globals.  The Metal HAL calls
        /// `setThreadgroupMemoryLength:atIndex:` for each entry before dispatch.
        /// Empty when the compute shader has no workgroup variables.
        workgroup_memory_sizes: Vec<u32>,
        /// Compute-stage immediates delivery metadata (Block 94 S2), when
        /// the compute entry point uses `var<immediate>` user data. `None`
        /// when the entry point declares no immediates.
        immediates: Option<HalMslImmediates>,
    },
    /// Per-stage MSL render sources.
    MslStages {
        /// Vertex stage MSL source.
        vertex: String,
        /// Optional fragment stage MSL source.
        fragment: Option<String>,
    },
    /// Per-stage MSL render sources with `_mslBufferSizes` binding metadata.
    MslStagesWithBufferSizes {
        /// Vertex stage MSL source.
        vertex: String,
        /// Optional fragment stage MSL source.
        fragment: Option<String>,
        /// Reserved vertex-stage buffer slot for `_mslBufferSizes`.
        vertex_buffer_sizes_slot: Option<u32>,
        /// Vertex-stage bindings whose byte lengths populate `_mslBufferSizes`.
        vertex_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Reserved fragment-stage buffer slot for `_mslBufferSizes`.
        fragment_buffer_sizes_slot: Option<u32>,
        /// Fragment-stage bindings whose byte lengths populate `_mslBufferSizes`.
        fragment_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
        /// Vertex-stage immediates delivery metadata (Block 94 S2), when the
        /// vertex entry point uses `var<immediate>` user data. `None` when
        /// the vertex entry point declares no immediates.
        vertex_immediates: Option<HalMslImmediates>,
        /// Fragment-stage immediates delivery metadata (Block 94 S2). Also
        /// carries the frag-depth clamp range offset within the block when
        /// this pipeline clamps frag_depth -- absorbs the old
        /// `fragment_frag_depth_clamp_slot` (a clamp-only pipeline still
        /// gets `Some` here, with `frag_depth_clamp_offset` set and no user
        /// immediates). `None` when the fragment entry point uses no
        /// immediates and does not clamp frag_depth.
        fragment_immediates: Option<HalMslImmediates>,
        /// Metal buffer indices for vertex buffers, in the same order as
        /// `vertex_buffer_mappings` passed to Tint's MSL codegen. These correspond to
        /// the `buffer_sizeN` fields appended after the storage-array size fields
        /// inside `_mslBufferSizes`; the HAL encoder must write the effective byte
        /// size (buffer.size − bind_offset) of each bound vertex buffer into those
        /// fields so Tint's vertex-pulling OOB guards compare against real data.
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
    },
    /// Glsl variant.
    Glsl {
        /// Source variant.
        source: String,
        /// Stage variant.
        stage: HalShaderStage,
    },
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

/// Metal immediates delivery descriptor for one shader stage (Block 94 S2):
/// where to bind the combined immediates block and how large it is. Mirrors
/// Dawn's `ImmediatesLayout.h` layout -- user immediate bytes first
/// (`[0, block_size)`, or `[0, frag_depth_clamp_offset)` when the clamp
/// range is also present), with pipeline-internal constants (currently only
/// the fragment frag-depth clamp range) appended directly after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct HalMslImmediates {
    /// Metal buffer slot the combined block is delivered to via
    /// `set{Vertex,Fragment}Bytes` / `setBytes`.
    pub slot: u32,
    /// Total block size in bytes delivered to this stage: the pipeline
    /// layout's reserved user-immediate budget, plus 8 bytes when this
    /// stage also carries the frag-depth clamp range.
    pub block_size: u32,
    /// Byte offset of the 8-byte frag-depth clamp range
    /// (`[min_depth, max_depth]`, both `f32`) within the block, when this
    /// stage clamps frag_depth. Always `None` for vertex and compute
    /// stages.
    pub frag_depth_clamp_offset: Option<u32>,
}

impl HalMslImmediates {
    /// Creates a new Metal immediates delivery descriptor.
    #[must_use]
    pub fn new(slot: u32, block_size: u32, frag_depth_clamp_offset: Option<u32>) -> Self {
        Self {
            slot,
            block_size,
            frag_depth_clamp_offset,
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
            buffer_sizes_slot: Some(3),
            buffer_size_bindings: vec![HalMslBufferSizeBinding::new(1, 2)],
            workgroup_memory_sizes: vec![32, 16],
            immediates: Some(HalMslImmediates::new(4, 16, None)),
        };

        assert!(matches!(
            source,
            HalShaderSource::MslWithBufferSizes {
                source,
                buffer_sizes_slot: Some(3),
                buffer_size_bindings,
                workgroup_memory_sizes,
                immediates: Some(HalMslImmediates {
                    slot: 4,
                    block_size: 16,
                    frag_depth_clamp_offset: None,
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
            vertex_buffer_sizes_slot: Some(7),
            vertex_buffer_size_bindings: vec![HalMslBufferSizeBinding::new(0, 1)],
            fragment_buffer_sizes_slot: None,
            fragment_buffer_size_bindings: Vec::new(),
            vertex_immediates: None,
            fragment_immediates: Some(HalMslImmediates::new(30, 72, Some(64))),
            vertex_buffer_metal_indices: vec![3, 5],
        };

        assert!(matches!(
            source,
            HalShaderSource::MslStagesWithBufferSizes {
                vertex,
                fragment,
                vertex_buffer_sizes_slot: Some(7),
                vertex_buffer_size_bindings,
                fragment_buffer_sizes_slot: None,
                fragment_buffer_size_bindings,
                vertex_immediates: None,
                fragment_immediates: Some(HalMslImmediates {
                    slot: 30,
                    block_size: 72,
                    frag_depth_clamp_offset: Some(64),
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
        let immediates = HalMslImmediates::new(30, 72, Some(64));

        assert_eq!(
            immediates,
            HalMslImmediates {
                slot: 30,
                block_size: 72,
                frag_depth_clamp_offset: Some(64),
            }
        );
    }
}
