use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::device::*;
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
///
/// `InputAttachment`-kind layout slots may be omitted by the caller — the
/// subpass render pass auto-wires them from the pass layout's input-source
/// map (block 55 "Input attachments are pass-local, auto-wired"). Non-input
/// slots are still required; an explicit `TextureView` supplied for an
/// input-attachment slot is still rejected (the slot owner is the pass, not
/// the caller).
pub(crate) fn validate_bind_group_descriptor(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry],
    limits: Limits,
) -> Option<String> {
    if layout.is_error() {
        return Some("cannot create bind group from an error bind group layout".to_owned());
    }
    let required_entry_count = layout
        .entries()
        .iter()
        .filter(|entry| !bind_group_layout_entry_is_input_attachment(entry))
        .count();
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
        if !seen.contains(&layout_entry.binding)
            && !bind_group_layout_entry_is_input_attachment(layout_entry)
        {
            return Some("bind group is missing a layout binding".to_owned());
        }
    }

    None
}

fn bind_group_layout_entry_is_input_attachment(entry: &BindGroupLayoutEntry) -> bool {
    #[cfg(feature = "tiled")]
    {
        matches!(entry.kind, Some(BindingLayoutKind::InputAttachment { .. }))
    }
    #[cfg(not(feature = "tiled"))]
    {
        let _ = entry;
        false
    }
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
            } else if ty == SamplerBindingType::Comparison
                && sampler.descriptor().compare.is_none()
            {
                Some("comparison sampler bindings require a compare function".to_owned())
            } else if ty != SamplerBindingType::Comparison
                && sampler.descriptor().compare.is_some()
            {
                Some("non-comparison sampler bindings must not use a compare function".to_owned())
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
        #[cfg(feature = "tiled")]
        (_, BindingLayoutKind::InputAttachment { .. }) => Some(
            "input-attachment binding must not be supplied in the bind group; it is auto-wired by the subpass pass"
                .to_owned(),
        ),
        (BindGroupResource::Invalid(message), _) => Some(message.clone()),
        _ => Some("bind group entry resource kind must match the layout".to_owned()),
    }
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
    if buffer.is_destroyed() {
        return Some("bind group buffer must not be destroyed".to_owned());
    }

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
    if texture.is_destroyed() {
        return Some("bind group texture must not be destroyed".to_owned());
    }
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
    match sample_type {
        TextureSampleType::Float => {
            if caps.output_class != Some(FormatOutputClass::Float) || !caps.filterable {
                return Some(
                    "float texture bindings require a filterable float texture format".to_owned(),
                );
            }
        }
        TextureSampleType::UnfilterableFloat => {
            if caps.output_class != Some(FormatOutputClass::Float) {
                return Some(
                    "unfilterable-float texture bindings require a float texture format".to_owned(),
                );
            }
        }
        TextureSampleType::Depth => {
            if !caps.aspects.depth {
                return Some("depth texture bindings require a depth texture format".to_owned());
            }
        }
        TextureSampleType::Sint => {
            if caps.output_class != Some(FormatOutputClass::Sint) {
                return Some("sint texture bindings require a sint texture format".to_owned());
            }
        }
        TextureSampleType::Uint => {
            if caps.output_class != Some(FormatOutputClass::Uint) {
                return Some("uint texture bindings require a uint texture format".to_owned());
            }
        }
    }

    None
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
    let texture = texture_view.texture();
    if texture.is_destroyed() {
        return Some("bind group texture must not be destroyed".to_owned());
    }
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
    if texture_view.array_layer_count() != 1 {
        return Some("storage texture bindings require a single array layer".to_owned());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "tiled")]
    use crate::shader::SHADER_STAGE_FRAGMENT;
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

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

    #[cfg(feature = "tiled")]
    fn input_attachment_layout_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: SHADER_STAGE_FRAGMENT,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::InputAttachment {
                sample_type: TextureSampleType::Float,
                multisampled: false,
            }),
        }
    }

    #[cfg(feature = "tiled")]
    fn uniform_layout_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: SHADER_STAGE_FRAGMENT,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 4,
            }),
        }
    }

    #[cfg(feature = "tiled")]
    fn uniform_bind_group_entry(device: &Device, binding: u32) -> BindGroupEntry {
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM,
            size: 4,
            mapped_at_creation: false,
        }));
        BindGroupEntry {
            binding,
            resource: BindGroupResource::Buffer {
                buffer,
                device: Arc::new(device.clone()),
                offset: 0,
                size: 4,
            },
        }
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_bind_group_descriptor_accepts_mixed_group_with_input_omitted() {
        let device = noop_device();
        let layout = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![input_attachment_layout_entry(0), uniform_layout_entry(1)],
            error: None,
        });
        let entries = vec![uniform_bind_group_entry(&device, 1)];

        assert_eq!(
            validate_bind_group_descriptor(&device, &layout, &entries, device.limits()),
            None
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_bind_group_descriptor_rejects_mixed_group_missing_non_input() {
        let device = noop_device();
        let layout = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![input_attachment_layout_entry(0), uniform_layout_entry(1)],
            error: None,
        });

        assert_eq!(
            validate_bind_group_descriptor(&device, &layout, &[], device.limits()),
            Some("bind group entry count must match bind group layout".to_owned())
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_bind_group_descriptor_rejects_explicit_view_for_input_slot() {
        let device = noop_device();
        let layout = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![uniform_layout_entry(0), input_attachment_layout_entry(1)],
            error: None,
        });
        let view = noop_render_attachment(&device);
        let entries = vec![BindGroupEntry {
            binding: 1,
            resource: BindGroupResource::TextureView {
                texture_view: view,
                device: Arc::new(device.clone()),
            },
        }];

        assert_eq!(
            validate_bind_group_descriptor(&device, &layout, &entries, device.limits()),
            Some(
                "input-attachment binding must not be supplied in the bind group; it is auto-wired by the subpass pass"
                    .to_owned()
            )
        );
    }
}
