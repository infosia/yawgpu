use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::buffer::{Buffer, BufferDescriptor, BufferUsage};
use crate::device::Device;
use crate::error::{DeviceError, ErrorKind};
use crate::extent::Extent3d;
use crate::format::FormatOutputClass;
use crate::queue::QueueBufferWrite;
use crate::texture::TextureUsage;
use crate::texture_view::{TextureAspect, TextureView, TextureViewDimension};

/// Origin in two dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin2d {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
}

/// Enumerates external texture formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExternalTextureFormat {
    /// Single-plane RGBA passthrough.
    Rgba,
    /// Two-plane NV12.
    Nv12,
}

/// Enumerates external texture rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExternalTextureRotation {
    /// No rotation.
    Rotate0,
    /// Rotate 90 degrees.
    Rotate90,
    /// Rotate 180 degrees.
    Rotate180,
    /// Rotate 270 degrees.
    Rotate270,
}

/// Describes an external texture.
#[derive(Debug, Clone)]
pub struct ExternalTextureDescriptor {
    /// First texture plane.
    pub plane0: Arc<TextureView>,
    /// Optional second texture plane.
    pub plane1: Option<Arc<TextureView>>,
    /// External texture format.
    pub format: ExternalTextureFormat,
    /// Crop origin in plane0 texels.
    pub crop_origin: Origin2d,
    /// Crop size in plane0 texels.
    pub crop_size: Extent3d,
    /// Shader-visible size.
    pub apparent_size: Extent3d,
    /// Whether to stop after YUV-to-RGB conversion.
    pub do_yuv_to_rgb_conversion_only: bool,
    /// Optional column-major mat3x4 conversion matrix.
    pub yuv_to_rgb_conversion_matrix: Option<[f32; 12]>,
    /// Source transfer-function parameters.
    pub src_transfer_function_parameters: [f32; 7],
    /// Destination transfer-function parameters.
    pub dst_transfer_function_parameters: [f32; 7],
    /// Column-major mat3x3 gamut conversion matrix.
    pub gamut_conversion_matrix: [f32; 9],
    /// Whether sampling is mirrored horizontally.
    pub mirrored: bool,
    /// Sampling rotation.
    pub rotation: ExternalTextureRotation,
}

/// External texture resource.
#[derive(Debug, Clone)]
pub struct ExternalTexture {
    inner: Arc<ExternalTextureInner>,
}

/// Shared external texture state.
#[derive(Debug)]
pub struct ExternalTextureInner {
    /// Plane texture views.
    pub planes: ArrayVec<Arc<TextureView>, 2>,
    /// Tint parameter uniform buffer.
    pub params: Arc<Buffer>,
    /// External texture format.
    pub format: ExternalTextureFormat,
    /// Shader-visible width.
    pub width: u32,
    /// Shader-visible height.
    pub height: u32,
    /// Owning device.
    pub device: Arc<Device>,
}

/// Packed bytes for Tint's external-texture params UBO.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalTextureParams {
    bytes: [u8; Self::SIZE],
}

impl ExternalTexture {
    /// Returns the shared inner state.
    #[must_use]
    pub fn inner(&self) -> &Arc<ExternalTextureInner> {
        &self.inner
    }

    /// Returns true when this object is an error external texture.
    #[must_use]
    pub fn is_error(&self) -> bool {
        false
    }
}

impl Device {
    /// Creates an external texture and its Tint params buffer.
    pub fn create_external_texture(
        &self,
        descriptor: ExternalTextureDescriptor,
    ) -> Result<ExternalTexture, DeviceError> {
        validate_external_texture_descriptor(&descriptor)?;

        let params = ExternalTextureParams::from_descriptor(&descriptor)?;
        let buffer = self.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            size: ExternalTextureParams::SIZE as u64,
            mapped_at_creation: false,
        });
        if buffer.is_error() {
            return Err(DeviceError::new(
                ErrorKind::Internal,
                "failed to create external texture params buffer",
            ));
        }
        if let Some(error) = self.queue().write_buffer(QueueBufferWrite {
            device: self.hal(),
            buffer: &buffer,
            offset: 0,
            data: params.as_bytes(),
        }) {
            return Err(error);
        }

        let mut planes = ArrayVec::new();
        planes.push(Arc::clone(&descriptor.plane0));
        if let Some(plane1) = &descriptor.plane1 {
            planes.push(Arc::clone(plane1));
        }

        Ok(ExternalTexture {
            inner: Arc::new(ExternalTextureInner {
                planes,
                params: Arc::new(buffer),
                format: descriptor.format,
                width: descriptor.apparent_size.width,
                height: descriptor.apparent_size.height,
                device: Arc::new(self.clone()),
            }),
        })
    }
}

impl ExternalTextureParams {
    /// Size in bytes of Tint's `tint_ExternalTextureParams` UBO.
    pub const SIZE: usize = 296;

    const NUM_PLANES: usize = 0;
    const DO_YUV_TO_RGB_CONVERSION_ONLY: usize = 4;
    const YUV_TO_RGB_CONVERSION_MATRIX: usize = 8;
    const SRC_TRANSFER_FUNCTION: usize = 56;
    const DST_TRANSFER_FUNCTION: usize = 88;
    const GAMUT_CONVERSION_MATRIX: usize = 120;
    const SAMPLE_TRANSFORM: usize = 168;
    const LOAD_TRANSFORM: usize = 200;
    const SAMPLE_PLANE0_RECT_MIN: usize = 232;
    const SAMPLE_PLANE0_RECT_MAX: usize = 240;
    const SAMPLE_PLANE1_RECT_MIN: usize = 248;
    const SAMPLE_PLANE1_RECT_MAX: usize = 256;
    const APPARENT_SIZE: usize = 264;
    const PLANE1_COORD_FACTOR: usize = 272;
    const OOTF_PARAM: usize = 280;

    /// Builds packed Tint params from an external-texture descriptor.
    pub fn from_descriptor(descriptor: &ExternalTextureDescriptor) -> Result<Self, DeviceError> {
        validate_external_texture_descriptor(descriptor)?;
        let plane0_extent = view_extent(&descriptor.plane0);
        let plane1_extent = descriptor.plane1.as_ref().map_or(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            |view| view_extent(view),
        );

        let mut params = Self {
            bytes: [0; Self::SIZE],
        };
        let num_planes = if descriptor.plane1.is_some() { 2 } else { 1 };
        params.write_u32(Self::NUM_PLANES, num_planes);
        params.write_u32(
            Self::DO_YUV_TO_RGB_CONVERSION_ONLY,
            u32::from(descriptor.do_yuv_to_rgb_conversion_only),
        );

        let yuv = descriptor.yuv_to_rgb_conversion_matrix.unwrap_or([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]);
        params.write_f32s(Self::YUV_TO_RGB_CONVERSION_MATRIX, &yuv);
        params.write_transfer_function(
            Self::SRC_TRANSFER_FUNCTION,
            descriptor.src_transfer_function_parameters,
        );
        params.write_transfer_function(
            Self::DST_TRANSFER_FUNCTION,
            descriptor.dst_transfer_function_parameters,
        );
        params.write_mat3x3(
            Self::GAMUT_CONVERSION_MATRIX,
            descriptor.gamut_conversion_matrix,
        );

        let transform = compute_transforms(descriptor, plane0_extent);
        params.write_mat3x2(Self::SAMPLE_TRANSFORM, transform.sample);
        params.write_mat3x2(Self::LOAD_TRANSFORM, transform.load);
        params.write_vec2(Self::SAMPLE_PLANE0_RECT_MIN, transform.plane0_rect_min);
        params.write_vec2(Self::SAMPLE_PLANE0_RECT_MAX, transform.plane0_rect_max);
        params.write_vec2(Self::SAMPLE_PLANE1_RECT_MIN, transform.plane1_rect_min);
        params.write_vec2(Self::SAMPLE_PLANE1_RECT_MAX, transform.plane1_rect_max);
        params.write_u32(Self::APPARENT_SIZE, transform.apparent_size[0]);
        params.write_u32(Self::APPARENT_SIZE + 4, transform.apparent_size[1]);
        params.write_vec2(
            Self::PLANE1_COORD_FACTOR,
            [
                plane1_extent.width as f32 / plane0_extent.width as f32,
                plane1_extent.height as f32 / plane0_extent.height as f32,
            ],
        );
        params.write_f32s(Self::OOTF_PARAM, &[0.0, 0.0, 0.0, 0.0]);
        Ok(params)
    }

    /// Returns the packed parameter bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; Self::SIZE] {
        &self.bytes
    }

    fn write_u32(&mut self, offset: usize, value: u32) {
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }

    fn write_f32s(&mut self, offset: usize, values: &[f32]) {
        for (index, value) in values.iter().enumerate() {
            self.bytes[offset + index * 4..offset + index * 4 + 4]
                .copy_from_slice(&value.to_ne_bytes());
        }
    }

    fn write_vec2(&mut self, offset: usize, value: [f32; 2]) {
        self.write_f32s(offset, &value);
    }

    fn write_mat3x2(&mut self, offset: usize, value: [[f32; 2]; 3]) {
        self.write_f32s(offset, &value[0]);
        self.write_f32s(offset + 8, &value[1]);
        self.write_f32s(offset + 16, &value[2]);
    }

    fn write_mat3x3(&mut self, offset: usize, value: [f32; 9]) {
        self.write_f32s(offset, &value[0..3]);
        self.write_f32s(offset + 16, &value[3..6]);
        self.write_f32s(offset + 32, &value[6..9]);
    }

    fn write_transfer_function(&mut self, offset: usize, value: [f32; 7]) {
        let mode = if value[0] < 0.0 {
            (-value[0]) as u32
        } else {
            0
        };
        self.write_u32(offset, mode);
        self.write_f32s(
            offset + 4,
            &[
                value[1], value[2], value[3], value[4], value[5], value[6], value[0],
            ],
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Transforms {
    sample: [[f32; 2]; 3],
    load: [[f32; 2]; 3],
    plane0_rect_min: [f32; 2],
    plane0_rect_max: [f32; 2],
    plane1_rect_min: [f32; 2],
    plane1_rect_max: [f32; 2],
    apparent_size: [u32; 2],
}

fn validate_external_texture_descriptor(
    descriptor: &ExternalTextureDescriptor,
) -> Result<(), DeviceError> {
    match descriptor.format {
        ExternalTextureFormat::Rgba if descriptor.plane1.is_some() => {
            return Err(DeviceError::validation(
                "RGBA external texture must not specify plane1",
            ));
        }
        ExternalTextureFormat::Nv12 if descriptor.plane1.is_none() => {
            return Err(DeviceError::validation(
                "NV12 external texture requires plane1",
            ));
        }
        _ => {}
    }
    validate_external_texture_plane(&descriptor.plane0)?;
    if let Some(plane1) = &descriptor.plane1 {
        validate_external_texture_plane(plane1)?;
    }
    let plane0_extent = view_extent(&descriptor.plane0);
    if descriptor.crop_size.width == 0 || descriptor.crop_size.height == 0 {
        return Err(DeviceError::validation(
            "external texture cropSize must be non-zero",
        ));
    }
    if descriptor.apparent_size.width == 0 || descriptor.apparent_size.height == 0 {
        return Err(DeviceError::validation(
            "external texture apparentSize must be non-zero",
        ));
    }
    let Some(crop_x_end) = descriptor
        .crop_origin
        .x
        .checked_add(descriptor.crop_size.width)
    else {
        return Err(DeviceError::validation(
            "external texture crop x range overflows",
        ));
    };
    let Some(crop_y_end) = descriptor
        .crop_origin
        .y
        .checked_add(descriptor.crop_size.height)
    else {
        return Err(DeviceError::validation(
            "external texture crop y range overflows",
        ));
    };
    if crop_x_end > plane0_extent.width || crop_y_end > plane0_extent.height {
        return Err(DeviceError::validation(
            "external texture crop rectangle exceeds plane0 size",
        ));
    }
    Ok(())
}

fn validate_external_texture_plane(plane: &TextureView) -> Result<(), DeviceError> {
    if plane.is_error() {
        return Err(DeviceError::validation(
            "external texture plane must not be an error texture view",
        ));
    }
    if !plane.usage().contains(TextureUsage::TEXTURE_BINDING) {
        return Err(DeviceError::validation(
            "external texture plane usage must include TEXTURE_BINDING",
        ));
    }
    if plane.dimension() != TextureViewDimension::D2 {
        return Err(DeviceError::validation(
            "external texture plane dimension must be 2D",
        ));
    }
    if plane.mip_level_count() != 1 {
        return Err(DeviceError::validation(
            "external texture plane mip level count must be 1",
        ));
    }
    let texture = plane.texture();
    if texture.sample_count() != 1 {
        return Err(DeviceError::validation(
            "external texture plane sample count must be 1",
        ));
    }
    let Some(caps) = texture.view_format_caps(plane.format()) else {
        return Err(DeviceError::validation(
            "external texture plane format must be supported",
        ));
    };
    if !caps.aspects.color
        || !caps.filterable
        || caps.output_class != Some(FormatOutputClass::Float)
        || plane.aspect() != TextureAspect::All
    {
        return Err(DeviceError::validation(
            "external texture plane must expose filterable-float color samples",
        ));
    }
    Ok(())
}

fn view_extent(view: &TextureView) -> Extent3d {
    view.texture().subresource_size(view.base_mip_level())
}

fn compute_transforms(
    descriptor: &ExternalTextureDescriptor,
    plane0_extent: Extent3d,
) -> Transforms {
    let plane0_size = [plane0_extent.width as f32, plane0_extent.height as f32];
    let crop_origin = [
        descriptor.crop_origin.x as f32,
        descriptor.crop_origin.y as f32,
    ];
    let crop_size = [
        descriptor.crop_size.width as f32,
        descriptor.crop_size.height as f32,
    ];

    let mut sample = translation(-0.5, -0.5);
    if descriptor.mirrored {
        sample = mul3(scale(-1.0, 1.0), sample);
    }

    let mut load_bounds = [
        descriptor.apparent_size.width.saturating_sub(1),
        descriptor.apparent_size.height.saturating_sub(1),
    ];
    match descriptor.rotation {
        ExternalTextureRotation::Rotate0 => {}
        ExternalTextureRotation::Rotate90 => {
            load_bounds.swap(0, 1);
            sample = mul3([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], sample);
        }
        ExternalTextureRotation::Rotate180 => {
            sample = mul3(scale(-1.0, -1.0), sample);
        }
        ExternalTextureRotation::Rotate270 => {
            load_bounds.swap(0, 1);
            sample = mul3([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], sample);
        }
    }
    sample = mul3(translation(0.5, 0.5), sample);

    let rect_scale = [crop_size[0] / plane0_size[0], crop_size[1] / plane0_size[1]];
    let rect_offset = [
        crop_origin[0] / plane0_size[0],
        crop_origin[1] / plane0_size[1],
    ];
    sample = mul3(scale(rect_scale[0], rect_scale[1]), sample);
    sample = mul3(translation(rect_offset[0], rect_offset[1]), sample);

    let to_texels = scale(plane0_size[0] - 1.0, plane0_size[1] - 1.0);
    let to_normalized = scale(
        1.0 / load_bounds[0].max(1) as f32,
        1.0 / load_bounds[1].max(1) as f32,
    );
    let load = mul3(to_texels, mul3(sample, to_normalized));

    let plane1_extent = descriptor.plane1.as_ref().map_or(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        |view| view_extent(view),
    );
    let half0 = [0.5 / plane0_size[0], 0.5 / plane0_size[1]];
    let half1 = [
        0.5 / plane1_extent.width as f32,
        0.5 / plane1_extent.height as f32,
    ];

    Transforms {
        sample: crop_mat3x2(sample),
        load: crop_mat3x2(load),
        plane0_rect_min: [rect_offset[0] + half0[0], rect_offset[1] + half0[1]],
        plane0_rect_max: [
            rect_offset[0] + rect_scale[0] - half0[0],
            rect_offset[1] + rect_scale[1] - half0[1],
        ],
        plane1_rect_min: [rect_offset[0] + half1[0], rect_offset[1] + half1[1]],
        plane1_rect_max: [
            rect_offset[0] + rect_scale[0] - half1[0],
            rect_offset[1] + rect_scale[1] - half1[1],
        ],
        apparent_size: load_bounds,
    }
}

fn translation(x: f32, y: f32) -> [[f32; 3]; 3] {
    [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]]
}

fn scale(x: f32, y: f32) -> [[f32; 3]; 3] {
    [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, 1.0]]
}

fn mul3(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

fn crop_mat3x2(value: [[f32; 3]; 3]) -> [[f32; 2]; 3] {
    [
        [value[0][0], value[1][0]],
        [value[0][1], value[1][1]],
        [value[0][2], value[1][2]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::noop_device;
    use crate::{TextureDescriptor, TextureDimension, TextureFormat, TextureViewDescriptor};

    fn texture_view(
        device: &Device,
        format: TextureFormat,
        usage: TextureUsage,
        sample_count: u32,
        dimension: Option<TextureViewDimension>,
    ) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format,
            mip_level_count: 1,
            sample_count,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: Some(usage),
            swizzle: None,
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn rgba_descriptor(device: &Device) -> ExternalTextureDescriptor {
        ExternalTextureDescriptor {
            plane0: texture_view(
                device,
                TextureFormat::from_raw(TextureFormat::RGBA8_UNORM),
                TextureUsage::TEXTURE_BINDING,
                1,
                Some(TextureViewDimension::D2),
            ),
            plane1: None,
            format: ExternalTextureFormat::Rgba,
            crop_origin: Origin2d { x: 0, y: 0 },
            crop_size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            apparent_size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            do_yuv_to_rgb_conversion_only: true,
            yuv_to_rgb_conversion_matrix: None,
            src_transfer_function_parameters: [0.0; 7],
            dst_transfer_function_parameters: [0.0; 7],
            gamut_conversion_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            mirrored: false,
            rotation: ExternalTextureRotation::Rotate0,
        }
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("slice len"))
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("slice len"))
    }

    #[test]
    fn external_texture_params_offsets_match_spec_table() {
        assert_eq!(ExternalTextureParams::SIZE, 296);
        assert_eq!(ExternalTextureParams::NUM_PLANES, 0);
        assert_eq!(ExternalTextureParams::DO_YUV_TO_RGB_CONVERSION_ONLY, 4);
        assert_eq!(ExternalTextureParams::YUV_TO_RGB_CONVERSION_MATRIX, 8);
        assert_eq!(ExternalTextureParams::SRC_TRANSFER_FUNCTION, 56);
        assert_eq!(ExternalTextureParams::DST_TRANSFER_FUNCTION, 88);
        assert_eq!(ExternalTextureParams::GAMUT_CONVERSION_MATRIX, 120);
        assert_eq!(ExternalTextureParams::SAMPLE_TRANSFORM, 168);
        assert_eq!(ExternalTextureParams::LOAD_TRANSFORM, 200);
        assert_eq!(ExternalTextureParams::SAMPLE_PLANE0_RECT_MIN, 232);
        assert_eq!(ExternalTextureParams::SAMPLE_PLANE0_RECT_MAX, 240);
        assert_eq!(ExternalTextureParams::SAMPLE_PLANE1_RECT_MIN, 248);
        assert_eq!(ExternalTextureParams::SAMPLE_PLANE1_RECT_MAX, 256);
        assert_eq!(ExternalTextureParams::APPARENT_SIZE, 264);
        assert_eq!(ExternalTextureParams::PLANE1_COORD_FACTOR, 272);
        assert_eq!(ExternalTextureParams::OOTF_PARAM, 280);
    }

    #[test]
    fn rgba_identity_params_pack_expected_values() {
        let device = noop_device();
        let params =
            ExternalTextureParams::from_descriptor(&rgba_descriptor(&device)).expect("params");
        let bytes = params.as_bytes();
        assert_eq!(read_u32(bytes, 0), 1);
        assert_eq!(read_u32(bytes, 4), 1);
        assert_eq!(read_f32(bytes, 8), 1.0);
        assert_eq!(read_f32(bytes, 28), 1.0);
        assert_eq!(read_f32(bytes, 48), 1.0);
        assert_eq!(read_f32(bytes, 168), 1.0);
        assert_eq!(read_f32(bytes, 180), 1.0);
        assert_eq!(read_f32(bytes, 184), 0.0);
        assert_eq!(read_u32(bytes, 264), 3);
        assert_eq!(read_u32(bytes, 268), 3);
        assert_eq!(read_f32(bytes, 272), 0.25);
        assert_eq!(read_f32(bytes, 276), 0.25);
    }

    #[test]
    fn nv12_plane_count_validation_rejects_one_plane() {
        let device = noop_device();
        let mut desc = rgba_descriptor(&device);
        desc.format = ExternalTextureFormat::Nv12;
        let error = device
            .create_external_texture(desc)
            .expect_err("one-plane NV12 must fail");
        assert_eq!(error.kind, ErrorKind::Validation);
    }

    #[test]
    fn plane_validation_rejects_missing_texture_binding_usage() {
        let device = noop_device();
        let mut desc = rgba_descriptor(&device);
        desc.plane0 = texture_view(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA8_UNORM),
            TextureUsage::COPY_SRC,
            1,
            Some(TextureViewDimension::D2),
        );
        let error = ExternalTextureParams::from_descriptor(&desc).expect_err("bad usage");
        assert_eq!(error.kind, ErrorKind::Validation);
    }

    #[test]
    fn plane_validation_rejects_non_2d_dimension() {
        let device = noop_device();
        let mut desc = rgba_descriptor(&device);
        desc.plane0 = texture_view(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA8_UNORM),
            TextureUsage::TEXTURE_BINDING,
            1,
            Some(TextureViewDimension::D2Array),
        );
        let error = ExternalTextureParams::from_descriptor(&desc).expect_err("bad dimension");
        assert_eq!(error.kind, ErrorKind::Validation);
    }

    #[test]
    fn plane_validation_rejects_multisampled_view() {
        let device = crate::test_helpers::noop_adapter()
            .create_device(None, &[crate::Feature::TextureFormatsTier1], "", "")
            .expect("Noop device with texture-formats-tier1");
        let mut desc = rgba_descriptor(&device);
        desc.plane0 = texture_view(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA8_SNORM),
            TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            4,
            Some(TextureViewDimension::D2),
        );
        let error = ExternalTextureParams::from_descriptor(&desc).expect_err("multisampled");
        assert_eq!(error.kind, ErrorKind::Validation);
    }

    #[test]
    fn plane_validation_rejects_non_filterable_float_samples() {
        let device = noop_device();
        let mut desc = rgba_descriptor(&device);
        desc.plane0 = texture_view(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA8_UINT),
            TextureUsage::TEXTURE_BINDING,
            1,
            Some(TextureViewDimension::D2),
        );
        let error = ExternalTextureParams::from_descriptor(&desc).expect_err("uint samples");
        assert_eq!(error.kind, ErrorKind::Validation);
    }

    #[test]
    fn device_create_external_texture_returns_resource() {
        let device = noop_device();
        let external = device
            .create_external_texture(rgba_descriptor(&device))
            .expect("external texture");
        assert!(!external.is_error());
        assert_eq!(external.inner().width, 4);
        assert_eq!(external.inner().height, 4);
        assert_eq!(external.inner().planes.len(), 1);
        assert_eq!(
            external.inner().params.size(),
            ExternalTextureParams::SIZE as u64
        );
    }
}
