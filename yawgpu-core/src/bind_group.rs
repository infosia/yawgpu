use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::device::*;
use crate::external_texture::*;
use crate::format::*;
use crate::limits::*;
use crate::sampler::*;
use crate::texture::*;
use crate::texture_view::*;

/// Stores entry metadata.
#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    /// Binding.
    pub binding: u32,
    /// Resource.
    pub resource: BindGroupResource,
}

/// Enumerates bind group resource values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BindGroupResource {
    /// Buffer variant.
    Buffer {
        /// Buffer variant.
        buffer: Arc<Buffer>,
        /// Device variant.
        device: Arc<Device>,
        /// Offset variant.
        offset: u64,
        /// Size variant.
        size: u64,
    },
    /// Sampler variant.
    Sampler {
        /// Sampler variant.
        sampler: Arc<Sampler>,
        /// Device variant.
        device: Arc<Device>,
    },
    /// Texture view variant.
    TextureView {
        /// Texture view variant.
        texture_view: Arc<TextureView>,
        /// Device variant.
        device: Arc<Device>,
    },
    /// External texture variant.
    ExternalTexture {
        /// External texture variant.
        external_texture: Arc<ExternalTexture>,
        /// Device variant.
        device: Arc<Device>,
    },
    /// Invalid variant.
    Invalid(String),
}

/// Stores bind group data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct BindGroup {
    pub(crate) inner: Arc<BindGroupInner>,
}

/// Holds shared state for the bind group handle.
#[derive(Debug)]
pub(crate) struct BindGroupInner {
    pub(crate) _layout: Arc<BindGroupLayout>,
    pub(crate) _entries: Vec<BindGroupEntry>,
    pub(crate) is_error: bool,
}

impl BindGroup {
    /// Creates a new instance.
    pub(crate) fn new(
        layout: Arc<BindGroupLayout>,
        entries: Vec<BindGroupEntry>,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(BindGroupInner {
                _layout: layout,
                _entries: entries,
                is_error,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the layout.
    #[must_use]
    pub(crate) fn layout(&self) -> &Arc<BindGroupLayout> {
        &self.inner._layout
    }

    /// Returns the entries.
    #[must_use]
    pub fn entries(&self) -> &[BindGroupEntry] {
        &self.inner._entries
    }
}

/// Validates bind group descriptor and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_descriptor(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry],
    limits: Limits,
) -> Option<String> {
    if layout.is_error() {
        return Some("cannot create bind group from an error bind group layout".to_owned());
    }
    let required_entry_count = layout.entries().len();
    if entries.len() != required_entry_count {
        return Some("bind group entry count must match bind group layout".to_owned());
    }

    let layout_entries = layout
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();

    for entry in entries {
        if !seen.insert(entry.binding) {
            return Some("bind group binding must not be set more than once".to_owned());
        }
        let Some(layout_entry) = layout_entries.get(&entry.binding).copied() else {
            return Some("bind group entry binding is not present in the layout".to_owned());
        };
        let Some(kind) = layout_entry.kind else {
            return Some("cannot create bind group from an invalid bind group layout".to_owned());
        };
        if let Some(message) = validate_bind_group_entry(device, entry, kind, limits) {
            return Some(message);
        }
    }

    for layout_entry in layout.entries() {
        if !seen.contains(&layout_entry.binding) {
            return Some("bind group is missing a layout binding".to_owned());
        }
    }

    None
}

/// Validates bind group entry and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_entry(
    device: &Device,
    entry: &BindGroupEntry,
    kind: BindingLayoutKind,
    limits: Limits,
) -> Option<String> {
    match (&entry.resource, kind) {
        (
            BindGroupResource::Buffer {
                buffer,
                device: resource_device,
                offset,
                size,
            },
            BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            },
        ) => validate_bind_group_buffer(
            device,
            resource_device,
            BindGroupBufferValidation {
                buffer,
                offset: *offset,
                size: *size,
                ty,
                min_binding_size,
                limits,
            },
        ),
        (
            BindGroupResource::Sampler {
                sampler,
                device: resource_device,
            },
            BindingLayoutKind::Sampler { ty },
        ) => {
            if !device.same(resource_device) {
                Some("bind group sampler must belong to the same device".to_owned())
            } else if sampler.is_error() {
                Some("bind group sampler must not be an error sampler".to_owned())
            } else if ty == SamplerBindingType::Comparison && sampler.descriptor().compare.is_none()
            {
                Some("comparison sampler bindings require a compare function".to_owned())
            } else if ty != SamplerBindingType::Comparison && sampler.descriptor().compare.is_some()
            {
                Some("non-comparison sampler bindings must not use a compare function".to_owned())
            } else if ty == SamplerBindingType::NonFiltering && sampler.descriptor().is_filtering()
            {
                Some(
                    "filtering sampler is incompatible with non-filtering sampler binding"
                        .to_owned(),
                )
            } else {
                None
            }
        }
        (
            BindGroupResource::TextureView {
                texture_view,
                device: resource_device,
            },
            BindingLayoutKind::Texture {
                sample_type,
                view_dimension,
                multisampled,
            },
        ) => validate_bind_group_texture(
            device,
            resource_device,
            texture_view,
            sample_type,
            view_dimension,
            multisampled,
        ),
        (
            BindGroupResource::TextureView {
                texture_view,
                device: resource_device,
            },
            BindingLayoutKind::StorageTexture {
                format,
                view_dimension,
                ..
            },
        ) => validate_bind_group_storage_texture(
            device,
            resource_device,
            texture_view,
            format,
            view_dimension,
        ),
        (
            BindGroupResource::ExternalTexture {
                external_texture,
                device: resource_device,
            },
            BindingLayoutKind::ExternalTexture,
        ) => validate_bind_group_external_texture(device, resource_device, external_texture),
        (BindGroupResource::ExternalTexture { .. }, _) => {
            Some("external texture bind group entry requires an external texture layout".to_owned())
        }
        (_, BindingLayoutKind::ExternalTexture) => {
            Some("external texture layout requires an external texture resource".to_owned())
        }
        (BindGroupResource::Invalid(message), _) => Some(message.clone()),
        _ => Some("bind group entry resource kind must match the layout".to_owned()),
    }
}

/// Validates bind group external texture and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_external_texture(
    device: &Device,
    resource_device: &Device,
    external_texture: &ExternalTexture,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group external texture must belong to the same device".to_owned());
    }
    if external_texture.is_error() {
        return Some(
            "bind group external texture must not be an error external texture".to_owned(),
        );
    }
    None
}

/// Validates bind group buffer and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_buffer(
    device: &Device,
    resource_device: &Device,
    validation: BindGroupBufferValidation<'_>,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group buffer must belong to the same device".to_owned());
    }
    let BindGroupBufferValidation {
        buffer,
        offset,
        size,
        ty,
        min_binding_size,
        limits,
    } = validation;
    if buffer.is_error() {
        return Some("bind group buffer must not be an error buffer".to_owned());
    }
    // Destroyed-buffer check is intentionally deferred to queue.submit time
    // (WebGPU spec §17.3 "Queue submit validation"). The buffer may be destroyed
    // after bind-group creation; as long as it is valid at create time and not
    // yet destroyed, the bind group itself is not an error. The buffer is tracked
    // in CommandBuffer::referenced_buffers (added in set_bind_group) and
    // queue.submit validates every referenced buffer is not destroyed.

    let (required_usage, alignment, max_binding_size) = match ty {
        BufferBindingType::Uniform => (
            BufferUsage::UNIFORM,
            u64::from(limits.min_uniform_buffer_offset_alignment),
            limits.max_uniform_buffer_binding_size,
        ),
        BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage => (
            BufferUsage::STORAGE,
            u64::from(limits.min_storage_buffer_offset_alignment),
            limits.max_storage_buffer_binding_size,
        ),
    };

    if !buffer.usage().contains(required_usage) {
        return Some("bind group buffer usage does not satisfy the layout".to_owned());
    }
    if alignment != 0 && !offset.is_multiple_of(alignment) {
        return Some("bind group buffer offset is not correctly aligned".to_owned());
    }

    let effective_size = if size == u64::MAX {
        let Some(remaining) = buffer.size().checked_sub(offset) else {
            return Some("bind group buffer offset exceeds buffer size".to_owned());
        };
        remaining
    } else {
        size
    };
    if effective_size == 0 {
        return Some("bind group buffer binding size must be greater than zero".to_owned());
    }
    if offset
        .checked_add(effective_size)
        .is_none_or(|end| end > buffer.size())
    {
        return Some("bind group buffer binding range exceeds buffer size".to_owned());
    }
    if min_binding_size != 0 && effective_size < min_binding_size {
        return Some("bind group buffer binding size is below the layout minimum".to_owned());
    }
    if effective_size > max_binding_size {
        return Some("bind group buffer binding size exceeds the device limit".to_owned());
    }
    if matches!(
        ty,
        BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage
    ) && !effective_size.is_multiple_of(4)
    {
        return Some("storage buffer binding size must be a multiple of 4".to_owned());
    }

    None
}

/// Stores bind group buffer validation data used by validation and backend submission.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BindGroupBufferValidation<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) ty: BufferBindingType,
    pub(crate) min_binding_size: u64,
    pub(crate) limits: Limits,
}

/// Validates bind group texture and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_texture(
    device: &Device,
    resource_device: &Device,
    texture_view: &TextureView,
    sample_type: TextureSampleType,
    view_dimension: TextureViewDimension,
    multisampled: bool,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group texture view must belong to the same device".to_owned());
    }
    if texture_view.is_error() {
        return Some("bind group texture view must not be an error texture view".to_owned());
    }
    let texture = texture_view.texture();
    // Destroyed-texture check is intentionally deferred to queue.submit time
    // (WebGPU spec §17.3 "Queue submit validation"). The texture may be destroyed
    // after bind-group creation; the texture is tracked in
    // CommandBuffer::referenced_textures (added in set_bind_group) and
    // queue.submit validates every referenced texture is not destroyed.
    if !texture_view.usage().contains(TextureUsage::TEXTURE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if (texture.sample_count() > 1) != multisampled {
        return Some("bind group texture multisampling must match the layout".to_owned());
    }
    let Some(caps) = texture_view
        .texture()
        .view_format_caps(texture_view.format())
    else {
        return Some("bind group texture view format must be supported".to_owned());
    };

    // WebGPU sample-type compatibility is keyed on the *view's aspect*, not the
    // whole format: the depth aspect of a depth-stencil format samples as
    // `depth` (and is also bindable to an unfilterable-float `texture_2d<f32>`),
    // the stencil aspect as `uint`. Reducing to the whole format's output class
    // (which is `None` for a depth-stencil format) wrongly rejects both. Mirror
    // wgpu's `device::resource` compatibility matrix.
    let Some(view_sample_type) = texture_view_sample_type(&caps, texture_view.aspect()) else {
        return Some("bind group texture view aspect has no sampleable type".to_owned());
    };
    let compatible = matches!(
        (sample_type, view_sample_type),
        (TextureSampleType::Uint, TextureSampleType::Uint)
            | (TextureSampleType::Sint, TextureSampleType::Sint)
            | (TextureSampleType::Depth, TextureSampleType::Depth)
            // A filterable-float binding requires a filterable-float view.
            | (TextureSampleType::Float, TextureSampleType::Float)
            // An unfilterable-float binding accepts any float view, and also a
            // depth view (the depth aspect of a depth/depth-stencil format).
            | (
                TextureSampleType::UnfilterableFloat,
                TextureSampleType::Float
                    | TextureSampleType::UnfilterableFloat
                    | TextureSampleType::Depth,
            )
    );
    if !compatible {
        return Some(format!(
            "bind group texture view sample type {view_sample_type:?} is not compatible with \
             layout sample type {sample_type:?}"
        ));
    }

    None
}

/// Returns the sample type a texture view exposes for `aspect`, following the
/// WebGPU per-aspect rules: a depth aspect samples as `depth`, a stencil aspect
/// as `uint`, and a colour format from its output class. Returns `None` when the
/// (format, aspect) pair is not sampleable — e.g. the combined `All` aspect of a
/// depth-stencil format, which must select a single aspect before sampling.
fn texture_view_sample_type(caps: &FormatCaps, aspect: TextureAspect) -> Option<TextureSampleType> {
    match aspect {
        TextureAspect::DepthOnly => caps.aspects.depth.then_some(TextureSampleType::Depth),
        TextureAspect::StencilOnly => caps.aspects.stencil.then_some(TextureSampleType::Uint),
        TextureAspect::All => {
            if caps.aspects.depth && caps.aspects.stencil {
                None
            } else if caps.aspects.depth {
                Some(TextureSampleType::Depth)
            } else if caps.aspects.stencil {
                Some(TextureSampleType::Uint)
            } else {
                match caps.output_class {
                    Some(FormatOutputClass::Float) => Some(if caps.filterable {
                        TextureSampleType::Float
                    } else {
                        TextureSampleType::UnfilterableFloat
                    }),
                    Some(FormatOutputClass::Sint) => Some(TextureSampleType::Sint),
                    Some(FormatOutputClass::Uint) => Some(TextureSampleType::Uint),
                    None => None,
                }
            }
        }
    }
}

/// Validates bind group storage texture and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_storage_texture(
    device: &Device,
    resource_device: &Device,
    texture_view: &TextureView,
    format: TextureFormat,
    view_dimension: TextureViewDimension,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group texture view must belong to the same device".to_owned());
    }
    if texture_view.is_error() {
        return Some("bind group texture view must not be an error texture view".to_owned());
    }
    // Destroyed-texture check is intentionally deferred to queue.submit time
    // (WebGPU spec §17.3 "Queue submit validation"). See the same comment in
    // validate_bind_group_texture.
    if !texture_view.usage().contains(TextureUsage::STORAGE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if texture_view.format() != format {
        return Some("storage texture view format must match the layout".to_owned());
    }
    if texture_view.mip_level_count() != 1 {
        return Some("storage texture bindings require a single mip level".to_owned());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

    fn external_texture_view(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn external_texture_descriptor(device: &Device) -> ExternalTextureDescriptor {
        ExternalTextureDescriptor {
            plane0: external_texture_view(device),
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

    #[test]
    fn bind_group_accessors_pin_entries_and_is_error() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: 4,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 4,
                }),
            }],
            error: None,
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM,
            size: 4,
            mapped_at_creation: false,
        }));
        let entry = BindGroupEntry {
            binding: 0,
            resource: BindGroupResource::Buffer {
                buffer,
                device: Arc::new(device.clone()),
                offset: 0,
                size: 4,
            },
        };

        let group = device.create_bind_group(layout.clone(), vec![entry.clone()]);
        assert!(!group.is_error());
        assert_eq!(group.entries().len(), 1);
        assert_eq!(group.entries()[0].binding, entry.binding);

        device.push_error_scope(ErrorFilter::Validation);
        let error_group = device.create_bind_group(layout, Vec::new());
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid bind group should be scoped");
        assert!(error_group.is_error());
        assert!(error_group.entries().is_empty());
        assert_eq!(
            error.message,
            "bind group entry count must match bind group layout"
        );
    }

    #[test]
    fn bind_group_texture_validation_uses_view_usage_override() {
        let device = noop_device();
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::STORAGE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);

        assert_eq!(
            validate_bind_group_texture(
                &device,
                &device,
                &view,
                TextureSampleType::UnfilterableFloat,
                TextureViewDimension::D2,
                false,
            ),
            None
        );
        assert_eq!(
            validate_bind_group_storage_texture(
                &device,
                &device,
                &view,
                rgba8_unorm(),
                TextureViewDimension::D2,
            ),
            Some("bind group texture usage does not satisfy the layout".to_owned())
        );
    }

    #[test]
    fn bind_group_storage_texture_allows_2d_array_layers_but_requires_single_mip() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: 4,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: rgba8_unorm(),
                    view_dimension: TextureViewDimension::D2Array,
                }),
            }],
            error: None,
        }));
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::STORAGE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 2,
            },
            format: rgba8_unorm(),
            mip_level_count: 2,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (array_view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(2),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        let group = device.create_bind_group(
            layout.clone(),
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(array_view),
                    device: Arc::new(device.clone()),
                },
            }],
        );
        assert!(!group.is_error());

        let (multi_mip_view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: Some(2),
            base_array_layer: 0,
            array_layer_count: Some(2),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        device.push_error_scope(ErrorFilter::Validation);
        let error_group = device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(multi_mip_view),
                    device: Arc::new(device.clone()),
                },
            }],
        );
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid bind group should be scoped");
        assert!(error_group.is_error());
        assert_eq!(
            error.message,
            "storage texture bindings require a single mip level"
        );
    }

    #[test]
    fn sampler_binding_type_validates_filtering_compatibility() {
        let device = noop_device();
        let filtering = Arc::new(device.create_sampler(SamplerDescriptor {
            mag_filter: Some(FilterMode::Linear),
            ..SamplerDescriptor::default()
        }));
        let non_filtering = Arc::new(device.create_sampler(SamplerDescriptor::default()));

        let entry = |sampler: Arc<Sampler>| BindGroupEntry {
            binding: 0,
            resource: BindGroupResource::Sampler {
                sampler,
                device: Arc::new(device.clone()),
            },
        };

        assert_eq!(
            validate_bind_group_entry(
                &device,
                &entry(Arc::clone(&filtering)),
                BindingLayoutKind::Sampler {
                    ty: SamplerBindingType::NonFiltering
                },
                device.limits(),
            ),
            Some("filtering sampler is incompatible with non-filtering sampler binding".to_owned())
        );
        assert_eq!(
            validate_bind_group_entry(
                &device,
                &entry(non_filtering),
                BindingLayoutKind::Sampler {
                    ty: SamplerBindingType::NonFiltering
                },
                device.limits(),
            ),
            None
        );
        assert_eq!(
            validate_bind_group_entry(
                &device,
                &entry(filtering),
                BindingLayoutKind::Sampler {
                    ty: SamplerBindingType::Filtering
                },
                device.limits(),
            ),
            None
        );
    }

    #[test]
    fn external_texture_binding_validation_accepts_layout_and_rejects_mismatches() {
        let device = noop_device();
        let external_texture = Arc::new(
            device
                .create_external_texture(external_texture_descriptor(&device))
                .expect("external texture"),
        );
        let external_entry = BindGroupEntry {
            binding: 0,
            resource: BindGroupResource::ExternalTexture {
                external_texture,
                device: Arc::new(device.clone()),
            },
        };

        assert_eq!(
            validate_bind_group_entry(
                &device,
                &external_entry,
                BindingLayoutKind::ExternalTexture,
                device.limits(),
            ),
            None
        );
        assert_eq!(
            validate_bind_group_entry(
                &device,
                &external_entry,
                BindingLayoutKind::Sampler {
                    ty: SamplerBindingType::Filtering
                },
                device.limits(),
            ),
            Some(
                "external texture bind group entry requires an external texture layout".to_owned()
            )
        );

        let sampler = Arc::new(device.create_sampler(SamplerDescriptor::default()));
        let sampler_entry = BindGroupEntry {
            binding: 0,
            resource: BindGroupResource::Sampler {
                sampler,
                device: Arc::new(device.clone()),
            },
        };
        assert_eq!(
            validate_bind_group_entry(
                &device,
                &sampler_entry,
                BindingLayoutKind::ExternalTexture,
                device.limits(),
            ),
            Some("external texture layout requires an external texture resource".to_owned())
        );
    }

    #[test]
    fn texture_view_sample_type_is_aspect_specific() {
        // Depth-stencil format: the aspect selects the sampleable type, and the
        // combined `All` aspect is not sampleable (must select one). This is the
        // F-055 root cause — reducing to the whole format's output class (which
        // is `None` for a depth-stencil format) loses the aspect.
        let ds = FormatCaps::depth_stencil(4);
        assert_eq!(
            texture_view_sample_type(&ds, TextureAspect::DepthOnly),
            Some(TextureSampleType::Depth)
        );
        assert_eq!(
            texture_view_sample_type(&ds, TextureAspect::StencilOnly),
            Some(TextureSampleType::Uint)
        );
        assert_eq!(texture_view_sample_type(&ds, TextureAspect::All), None);

        // Depth-only format: `All` resolves to the single depth aspect.
        let depth = FormatCaps::depth(4);
        assert_eq!(
            texture_view_sample_type(&depth, TextureAspect::All),
            Some(TextureSampleType::Depth)
        );

        // Colour formats resolve from their output class via the `All` aspect.
        assert_eq!(
            texture_view_sample_type(&FormatCaps::uint_color(4, 1), TextureAspect::All),
            Some(TextureSampleType::Uint)
        );
        assert_eq!(
            texture_view_sample_type(&FormatCaps::sint_color(4, 1), TextureAspect::All),
            Some(TextureSampleType::Sint)
        );
        // A non-filterable float colour format samples as unfilterable-float.
        assert_eq!(
            texture_view_sample_type(&FormatCaps::float_color(16, 4), TextureAspect::All),
            Some(TextureSampleType::UnfilterableFloat)
        );
    }
}
