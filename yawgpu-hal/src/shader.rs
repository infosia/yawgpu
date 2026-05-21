/// Enumerates HAL shader source values.
#[derive(Debug, Clone)]
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
}
