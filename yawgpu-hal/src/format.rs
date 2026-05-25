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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HalTextureFormat {
    /// R8 unorm variant.
    R8Unorm,
    /// Rgba8 unorm variant.
    Rgba8Unorm,
    /// Bgra8 unorm variant.
    Bgra8Unorm,
    /// Rgba16 float variant.
    Rgba16Float,
    /// Stencil8 variant.
    Stencil8,
    /// Depth16 unorm variant.
    Depth16Unorm,
    /// Depth24 plus variant.
    Depth24Plus,
    /// Depth24 plus stencil8 variant.
    Depth24PlusStencil8,
    /// Depth32 float variant.
    Depth32Float,
    /// Depth32 float stencil8 variant.
    Depth32FloatStencil8,
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

/// Enumerates HAL buffer usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalBufferUsage {
    /// MAP_READ.
    pub map_read: bool,
    /// MAP_WRITE.
    pub map_write: bool,
    /// Copy src.
    pub copy_src: bool,
    /// Copy dst.
    pub copy_dst: bool,
    /// Index buffer.
    pub index: bool,
    /// Vertex buffer.
    pub vertex: bool,
    /// Uniform buffer.
    pub uniform: bool,
    /// Storage buffer.
    pub storage: bool,
    /// Indirect buffer.
    pub indirect: bool,
    /// Query resolve destination.
    pub query_resolve: bool,
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

/// Enumerates HAL stencil operation values.
#[derive(Debug, Clone, Copy)]
pub enum HalStencilOperation {
    /// Keep variant.
    Keep,
    /// Zero variant.
    Zero,
    /// Replace variant.
    Replace,
    /// Invert variant.
    Invert,
    /// Increment clamp variant.
    IncrementClamp,
    /// Decrement clamp variant.
    DecrementClamp,
    /// Increment wrap variant.
    IncrementWrap,
    /// Decrement wrap variant.
    DecrementWrap,
}

#[allow(dead_code)]
pub(crate) fn format_has_depth_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Depth16Unorm
            | HalTextureFormat::Depth24Plus
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32Float
            | HalTextureFormat::Depth32FloatStencil8
    )
}

#[allow(dead_code)]
pub(crate) fn format_has_stencil_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Stencil8
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32FloatStencil8
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_buffer_usage_default_is_all_false() {
        let usage = HalBufferUsage::default();

        assert!(!usage.map_read);
        assert!(!usage.map_write);
        assert!(!usage.copy_src);
        assert!(!usage.copy_dst);
        assert!(!usage.index);
        assert!(!usage.vertex);
        assert!(!usage.uniform);
        assert!(!usage.storage);
        assert!(!usage.indirect);
        assert!(!usage.query_resolve);
    }

    #[test]
    fn hal_stencil_operation_variants_are_constructible() {
        let operations = [
            HalStencilOperation::Keep,
            HalStencilOperation::Zero,
            HalStencilOperation::Replace,
            HalStencilOperation::Invert,
            HalStencilOperation::IncrementClamp,
            HalStencilOperation::DecrementClamp,
            HalStencilOperation::IncrementWrap,
            HalStencilOperation::DecrementWrap,
        ];

        assert!(matches!(operations[0], HalStencilOperation::Keep));
        assert!(matches!(operations[1], HalStencilOperation::Zero));
        assert!(matches!(operations[2], HalStencilOperation::Replace));
        assert!(matches!(operations[3], HalStencilOperation::Invert));
        assert!(matches!(operations[4], HalStencilOperation::IncrementClamp));
        assert!(matches!(operations[5], HalStencilOperation::DecrementClamp));
        assert!(matches!(operations[6], HalStencilOperation::IncrementWrap));
        assert!(matches!(operations[7], HalStencilOperation::DecrementWrap));
    }

    #[test]
    fn format_has_depth_aspect_covers_relevant_formats() {
        assert!(format_has_depth_aspect(HalTextureFormat::Depth16Unorm));
        assert!(format_has_depth_aspect(HalTextureFormat::Depth24Plus));
        assert!(format_has_depth_aspect(
            HalTextureFormat::Depth24PlusStencil8
        ));
        assert!(format_has_depth_aspect(HalTextureFormat::Depth32Float));
        assert!(format_has_depth_aspect(
            HalTextureFormat::Depth32FloatStencil8
        ));
        assert!(!format_has_depth_aspect(HalTextureFormat::Stencil8));
        assert!(!format_has_depth_aspect(HalTextureFormat::Rgba16Float));
    }

    #[test]
    fn format_has_stencil_aspect_covers_relevant_formats() {
        assert!(format_has_stencil_aspect(HalTextureFormat::Stencil8));
        assert!(format_has_stencil_aspect(
            HalTextureFormat::Depth24PlusStencil8
        ));
        assert!(format_has_stencil_aspect(
            HalTextureFormat::Depth32FloatStencil8
        ));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth16Unorm));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth24Plus));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth32Float));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Rgba16Float));
    }
}
