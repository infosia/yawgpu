/// Enumerates HAL shader source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalShaderSource {
    /// Msl variant.
    Msl(String),
    /// Spir v variant.
    SpirV(Vec<u32>),
    /// Spir vstages variant.
    SpirVStages {
        /// Vertex variant.
        vertex: Vec<u32>,
        /// Fragment variant.
        fragment: Vec<u32>,
    },
    /// Glsl variant.
    Glsl {
        /// Source variant.
        source: String,
        /// Stage variant.
        stage: HalShaderStage,
    },
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
