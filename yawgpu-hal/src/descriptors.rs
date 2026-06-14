use crate::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology,
    HalStencilOperation, HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

/// Describes HAL render pipeline descriptor.
#[derive(Debug, Clone)]
pub struct HalRenderPipelineDescriptor {
    /// Multisample count.
    pub sample_count: u32,
    /// Multisample coverage mask.
    pub sample_mask: u32,
    /// Enables alpha-to-coverage.
    pub alpha_to_coverage_enabled: bool,
    /// Color target states in attachment slot order; `None` is an empty slot.
    pub color_targets: Vec<Option<HalColorTargetState>>,
    /// Depth stencil state.
    pub depth_stencil: Option<HalDepthStencilState>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalVertexBufferLayout>,
    /// Primitive topology.
    pub primitive_topology: HalPrimitiveTopology,
    /// Front-facing winding.
    pub front_face: HalFrontFace,
    /// Face culling mode.
    pub cull_mode: HalCullMode,
    /// Enables unclipped depth.
    pub unclipped_depth: bool,
}

/// Enumerates HAL front-facing winding values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalFrontFace {
    /// Counter-clockwise winding.
    Ccw,
    /// Clockwise winding.
    Cw,
}

/// Enumerates HAL primitive culling values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalCullMode {
    /// Disable face culling.
    None,
    /// Cull front-facing primitives.
    Front,
    /// Cull back-facing primitives.
    Back,
}

/// Describes one HAL color target state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalColorTargetState {
    /// Texture format.
    pub format: HalTextureFormat,
    /// Optional blend state.
    pub blend: Option<HalBlendState>,
    /// RGBA channel write mask bits.
    pub write_mask: u32,
}

/// Describes HAL color and alpha blend state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalBlendState {
    /// Color blend component.
    pub color: HalBlendComponent,
    /// Alpha blend component.
    pub alpha: HalBlendComponent,
}

/// Describes one HAL blend component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalBlendComponent {
    /// Blend operation.
    pub operation: HalBlendOperation,
    /// Source blend factor.
    pub src_factor: HalBlendFactor,
    /// Destination blend factor.
    pub dst_factor: HalBlendFactor,
}

/// Enumerates HAL blend operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalBlendOperation {
    /// Add source and destination terms.
    Add,
    /// Subtract destination from source.
    Subtract,
    /// Subtract source from destination.
    ReverseSubtract,
    /// Take the component-wise minimum.
    Min,
    /// Take the component-wise maximum.
    Max,
}

/// Enumerates HAL blend factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalBlendFactor {
    /// Zero factor.
    Zero,
    /// One factor.
    One,
    /// Source color factor.
    Src,
    /// One minus source color factor.
    OneMinusSrc,
    /// Source alpha factor.
    SrcAlpha,
    /// One minus source alpha factor.
    OneMinusSrcAlpha,
    /// Destination color factor.
    Dst,
    /// One minus destination color factor.
    OneMinusDst,
    /// Destination alpha factor.
    DstAlpha,
    /// One minus destination alpha factor.
    OneMinusDstAlpha,
    /// Saturated source alpha factor.
    SrcAlphaSaturated,
    /// Blend constant factor.
    Constant,
    /// One minus blend constant factor.
    OneMinusConstant,
    /// Source one color factor.
    Src1,
    /// One minus source one color factor.
    OneMinusSrc1,
    /// Source one alpha factor.
    Src1Alpha,
    /// One minus source one alpha factor.
    OneMinusSrc1Alpha,
}

/// Describes HAL depth stencil state.
#[derive(Debug, Clone, Copy)]
pub struct HalDepthStencilState {
    /// Format.
    pub format: HalTextureFormat,
    /// Depth write enabled.
    pub depth_write_enabled: bool,
    /// Depth compare.
    pub depth_compare: HalCompareFunction,
    /// Stencil front state.
    pub stencil_front: HalStencilFaceState,
    /// Stencil back state.
    pub stencil_back: HalStencilFaceState,
    /// Stencil read mask.
    pub stencil_read_mask: u32,
    /// Stencil write mask.
    pub stencil_write_mask: u32,
    /// Depth bias.
    pub depth_bias: i32,
    /// Depth bias slope scale.
    pub depth_bias_slope_scale: f32,
    /// Depth bias clamp.
    pub depth_bias_clamp: f32,
}

/// Describes HAL stencil face state.
#[derive(Debug, Clone, Copy)]
pub struct HalStencilFaceState {
    /// Compare function.
    pub compare: HalCompareFunction,
    /// Stencil fail operation.
    pub fail_op: HalStencilOperation,
    /// Depth fail operation.
    pub depth_fail_op: HalStencilOperation,
    /// Depth stencil pass operation.
    pub pass_op: HalStencilOperation,
}

/// Stores layout metadata.
#[derive(Debug, Clone)]
pub struct HalVertexBufferLayout {
    /// Original WebGPU vertex buffer slot.
    pub slot: u32,
    /// Array stride.
    pub array_stride: u64,
    /// Step mode.
    pub step_mode: HalVertexStepMode,
    /// Attributes.
    pub attributes: Vec<HalVertexAttribute>,
}

/// Stores attribute metadata.
#[derive(Debug, Clone)]
pub struct HalVertexAttribute {
    /// Format.
    pub format: HalVertexFormat,
    /// Offset.
    pub offset: u64,
    /// Shader location.
    pub shader_location: u32,
    /// Metal buffer index.
    pub metal_buffer_index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stencil_face_state() -> HalStencilFaceState {
        HalStencilFaceState {
            compare: HalCompareFunction::Always,
            fail_op: HalStencilOperation::Keep,
            depth_fail_op: HalStencilOperation::Replace,
            pass_op: HalStencilOperation::IncrementClamp,
        }
    }

    #[test]
    fn hal_depth_stencil_state_constructs_and_formats_debug() {
        let state = HalDepthStencilState {
            format: HalTextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: HalCompareFunction::LessEqual,
            stencil_front: stencil_face_state(),
            stencil_back: stencil_face_state(),
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            depth_bias: 42,
            depth_bias_slope_scale: 1.5,
            depth_bias_clamp: 0.25,
        };

        assert!(format!("{state:?}").contains("HalDepthStencilState"));
    }

    #[test]
    fn hal_stencil_face_state_constructs_and_formats_debug() {
        let state = stencil_face_state();

        assert!(format!("{state:?}").contains("HalStencilFaceState"));
    }

    #[test]
    fn hal_color_target_state_carries_blend_and_write_mask() {
        let component = HalBlendComponent {
            operation: HalBlendOperation::ReverseSubtract,
            src_factor: HalBlendFactor::SrcAlpha,
            dst_factor: HalBlendFactor::OneMinusConstant,
        };
        let state = HalColorTargetState {
            format: HalTextureFormat::Rgba8Unorm,
            blend: Some(HalBlendState {
                color: component,
                alpha: component,
            }),
            write_mask: 0b0101,
        };

        assert_eq!(state.write_mask, 0b0101);
        assert_eq!(state.blend.expect("blend state").color, component);
        assert!(format!("{state:?}").contains("HalColorTargetState"));
    }
}

/// Stores origin metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalOrigin3d {
    /// X.
    pub x: u32,
    /// Y.
    pub y: u32,
    /// Z.
    pub z: u32,
}

/// Stores extent metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalExtent3d {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Depth or array layers.
    pub depth_or_array_layers: u32,
}

/// Enumerates HAL texture dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalTextureDimension {
    /// One-dimensional texture.
    D1,
    /// Two-dimensional texture or two-dimensional texture array.
    D2,
    /// Three-dimensional texture.
    D3,
}

/// Describes HAL texture descriptor.
#[derive(Debug, Clone, Copy)]
pub struct HalTextureDescriptor {
    /// Dimension.
    pub dimension: HalTextureDimension,
    /// Format.
    pub format: HalTextureFormat,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Depth or array layers.
    pub depth_or_array_layers: u32,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Usage.
    pub usage: HalTextureUsage,
}

/// Describes HAL transient attachment descriptor.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy)]
pub struct HalTransientAttachmentDescriptor {
    /// Format.
    pub format: HalTextureFormat,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Sample count.
    pub sample_count: u32,
}

/// Describes HAL sampler descriptor.
#[derive(Debug, Clone, Copy)]
pub struct HalSamplerDescriptor {
    /// Address mode u.
    pub address_mode_u: HalAddressMode,
    /// Address mode v.
    pub address_mode_v: HalAddressMode,
    /// Address mode w.
    pub address_mode_w: HalAddressMode,
    /// Mag filter.
    pub mag_filter: HalFilterMode,
    /// Min filter.
    pub min_filter: HalFilterMode,
    /// Mipmap filter.
    pub mipmap_filter: HalMipmapFilterMode,
    /// Lod min clamp.
    pub lod_min_clamp: f32,
    /// Lod max clamp.
    pub lod_max_clamp: f32,
    /// Compare.
    pub compare: Option<HalCompareFunction>,
    /// Max anisotropy.
    pub max_anisotropy: u16,
}
