/// Enumerates HAL vertex format values.
#[derive(Debug, Clone, Copy)]
pub enum HalVertexFormat {
    /// Float32 variant.
    Float32,
    /// Float32x2 variant.
    Float32x2,
    /// Float32x3 variant.
    Float32x3,
    /// Float32x4 variant.
    Float32x4,
    /// Unsupported variant.
    Unsupported,
}

/// Enumerates HAL vertex step mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalVertexStepMode {
    /// Vertex variant.
    Vertex,
    /// Instance variant.
    Instance,
}

/// Enumerates HAL primitive topology values.
#[derive(Debug, Clone, Copy)]
pub enum HalPrimitiveTopology {
    /// Point list variant.
    PointList,
    /// Line list variant.
    LineList,
    /// Line strip variant.
    LineStrip,
    /// Triangle list variant.
    TriangleList,
    /// Triangle strip variant.
    TriangleStrip,
}

/// Enumerates HAL texture format values.
#[derive(Debug, Clone, Copy)]
pub enum HalTextureFormat {
    /// R8 unorm variant.
    R8Unorm,
    /// Rgba8 unorm variant.
    Rgba8Unorm,
    /// Bgra8 unorm variant.
    Bgra8Unorm,
    /// Unsupported variant.
    Unsupported,
}

/// Enumerates HAL texture usage.
#[derive(Debug, Clone, Copy)]
pub struct HalTextureUsage {
    /// Copy src.
    pub copy_src: bool,
    /// Copy dst.
    pub copy_dst: bool,
    /// Texture binding.
    pub texture_binding: bool,
    /// Storage binding.
    pub storage_binding: bool,
    /// Render attachment.
    pub render_attachment: bool,
}

/// Enumerates HAL address mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalAddressMode {
    /// Clamp to edge variant.
    ClampToEdge,
    /// Repeat variant.
    Repeat,
    /// Mirror repeat variant.
    MirrorRepeat,
}

/// Enumerates HAL filter mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalFilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates HAL mipmap filter mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalMipmapFilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates HAL compare function values.
#[derive(Debug, Clone, Copy)]
pub enum HalCompareFunction {
    /// Never variant.
    Never,
    /// Less variant.
    Less,
    /// Equal variant.
    Equal,
    /// Less equal variant.
    LessEqual,
    /// Greater variant.
    Greater,
    /// Not equal variant.
    NotEqual,
    /// Greater equal variant.
    GreaterEqual,
    /// Always variant.
    Always,
}
