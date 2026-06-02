use crate::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology,
    HalStencilOperation, HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

/// Describes HAL render pipeline descriptor.
#[derive(Debug, Clone)]
pub struct HalRenderPipelineDescriptor {
    /// Color formats.
    pub color_formats: Vec<HalTextureFormat>,
    /// Depth stencil state.
    pub depth_stencil: Option<HalDepthStencilState>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalVertexBufferLayout>,
    /// Primitive topology.
    pub primitive_topology: HalPrimitiveTopology,
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
