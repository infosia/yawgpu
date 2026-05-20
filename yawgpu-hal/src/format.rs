#[derive(Debug, Clone, Copy)]
pub enum HalVertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Unsupported,
}

#[derive(Debug, Clone, Copy)]
pub enum HalVertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy)]
pub enum HalPrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy)]
pub enum HalTextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Unsupported,
}

#[derive(Debug, Clone, Copy)]
pub struct HalTextureUsage {
    pub copy_src: bool,
    pub copy_dst: bool,
    pub texture_binding: bool,
    pub storage_binding: bool,
    pub render_attachment: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum HalAddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone, Copy)]
pub enum HalFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum HalMipmapFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum HalCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}
