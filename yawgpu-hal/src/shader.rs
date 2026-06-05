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
        };

        assert!(matches!(
            source,
            HalShaderSource::MslWithBufferSizes {
                source,
                buffer_sizes_slot: Some(3),
                buffer_size_bindings,
            } if source == "kernel void main0() {}"
                && buffer_size_bindings == [HalMslBufferSizeBinding::new(1, 2)]
        ));
    }
}
