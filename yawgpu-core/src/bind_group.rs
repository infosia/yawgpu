use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::device::*;
use crate::limits::*;
use crate::sampler::*;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindGroupResource,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BindGroupResource {
    Buffer {
        buffer: Arc<Buffer>,
        device: Arc<Device>,
        offset: u64,
        size: u64,
    },
    Sampler {
        sampler: Arc<Sampler>,
        device: Arc<Device>,
    },
    TextureView {
        texture_view: Arc<TextureView>,
        device: Arc<Device>,
    },
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct BindGroup {
    pub(crate) inner: Arc<BindGroupInner>,
}

#[derive(Debug)]
pub(crate) struct BindGroupInner {
    pub(crate) _layout: Arc<BindGroupLayout>,
    pub(crate) _entries: Vec<BindGroupEntry>,
    pub(crate) is_error: bool,
}

impl BindGroup {
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

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn layout(&self) -> &Arc<BindGroupLayout> {
        &self.inner._layout
    }

    #[must_use]
    pub fn entries(&self) -> &[BindGroupEntry] {
        &self.inner._entries
    }
}

pub(crate) fn validate_bind_group_descriptor(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry],
    limits: Limits,
) -> Option<String> {
    if layout.is_error() {
        return Some("cannot create bind group from an error bind group layout".to_owned());
    }
    if entries.len() != layout.entries().len() {
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
            BindingLayoutKind::Sampler { .. },
        ) => {
            if !device.same(resource_device) {
                Some("bind group sampler must belong to the same device".to_owned())
            } else if sampler.is_error() {
                Some("bind group sampler must not be an error sampler".to_owned())
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
            BindingLayoutKind::StorageTexture { view_dimension, .. },
        ) => validate_bind_group_storage_texture(
            device,
            resource_device,
            texture_view,
            view_dimension,
        ),
        (BindGroupResource::Invalid(message), _) => Some(message.clone()),
        _ => Some("bind group entry resource kind must match the layout".to_owned()),
    }
}

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

    None
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BindGroupBufferValidation<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) ty: BufferBindingType,
    pub(crate) min_binding_size: u64,
    pub(crate) limits: Limits,
}

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
    if !texture.usage().contains(TextureUsage::TEXTURE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if (texture.sample_count() > 1) != multisampled {
        return Some("bind group texture multisampling must match the layout".to_owned());
    }
    if texture_view.format().caps().is_some_and(|caps| {
        (caps.aspects.depth || caps.aspects.stencil) && sample_type == TextureSampleType::Float
    }) {
        return Some("depth or stencil texture bindings must not use Float sample type".to_owned());
    }

    None
}

pub(crate) fn validate_bind_group_storage_texture(
    device: &Device,
    resource_device: &Device,
    texture_view: &TextureView,
    view_dimension: TextureViewDimension,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group texture view must belong to the same device".to_owned());
    }
    if texture_view.is_error() {
        return Some("bind group texture view must not be an error texture view".to_owned());
    }
    let texture = texture_view.texture();
    if !texture.usage().contains(TextureUsage::STORAGE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if texture_view.array_layer_count() != 1 {
        return Some("storage texture bindings require a single array layer".to_owned());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
