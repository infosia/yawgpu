use std::collections::BTreeSet;
use std::sync::Arc;

use crate::format::*;
use crate::limits::*;
use crate::shader::*;
use crate::texture_view::*;

pub(crate) fn validate_bind_group_layout_descriptor(
    entries: &[BindGroupLayoutEntry],
    limits: Limits,
) -> Option<String> {
    if entries.len() > 1000 {
        return Some("bind group layout entry count exceeds 1000".to_owned());
    }

    let mut bindings = BTreeSet::new();
    let mut dynamic_uniform_buffers = 0_u32;
    let mut dynamic_storage_buffers = 0_u32;
    let mut stage_counts = [StageResourceCounts::default(); 3];

    for entry in entries {
        if entry.binding >= 1000 {
            return Some("bind group layout binding must be less than 1000".to_owned());
        }
        if !bindings.insert(entry.binding) {
            return Some("bind group layout bindings must be unique".to_owned());
        }
        if entry.binding_array_size > 1 {
            return Some(
                "bind group layout bindingArraySize greater than one is not supported".to_owned(),
            );
        }

        let Some(kind) = entry.kind else {
            return Some("bind group layout entry must set exactly one binding layout".to_owned());
        };

        match kind {
            BindingLayoutKind::Buffer {
                ty,
                has_dynamic_offset,
                ..
            } => {
                match ty {
                    BufferBindingType::Uniform if has_dynamic_offset => {
                        dynamic_uniform_buffers += 1;
                    }
                    BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage
                        if has_dynamic_offset =>
                    {
                        dynamic_storage_buffers += 1;
                    }
                    _ => {}
                }
                if dynamic_uniform_buffers > limits.max_dynamic_uniform_buffers_per_pipeline_layout
                {
                    return Some(
                        "too many dynamic uniform buffers in bind group layout".to_owned(),
                    );
                }
                if dynamic_storage_buffers > limits.max_dynamic_storage_buffers_per_pipeline_layout
                {
                    return Some(
                        "too many dynamic storage buffers in bind group layout".to_owned(),
                    );
                }
            }
            BindingLayoutKind::Texture {
                view_dimension,
                multisampled,
                ..
            } => {
                if multisampled && view_dimension != TextureViewDimension::D2 {
                    return Some(
                        "multisampled texture bindings require 2D view dimension".to_owned(),
                    );
                }
            }
            BindingLayoutKind::StorageTexture {
                format,
                view_dimension,
                ..
            } => {
                if view_dimension == TextureViewDimension::D1 {
                    return Some(
                        "storage texture bindings must not use 1D view dimension".to_owned(),
                    );
                }
                let Some(caps) = format.caps() else {
                    return Some("storage texture binding format must not be Undefined".to_owned());
                };
                if !caps.storage_capable {
                    return Some(
                        "storage texture binding format must support storage usage".to_owned(),
                    );
                }
            }
            BindingLayoutKind::Sampler { .. } => {}
        }

        for stage in visible_stages(entry.visibility) {
            stage_counts[stage].add(kind);
            if stage_counts[stage].sampled_textures > limits.max_sampled_textures_per_shader_stage {
                return Some("too many sampled textures for one shader stage".to_owned());
            }
            if stage_counts[stage].samplers > limits.max_samplers_per_shader_stage {
                return Some("too many samplers for one shader stage".to_owned());
            }
            if stage_counts[stage].storage_buffers > limits.max_storage_buffers_per_shader_stage {
                return Some("too many storage buffers for one shader stage".to_owned());
            }
            if stage_counts[stage].storage_textures > limits.max_storage_textures_per_shader_stage {
                return Some("too many storage textures for one shader stage".to_owned());
            }
            if stage_counts[stage].uniform_buffers > limits.max_uniform_buffers_per_shader_stage {
                return Some("too many uniform buffers for one shader stage".to_owned());
            }
        }
    }

    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupLayoutDescriptor {
    pub entries: Vec<BindGroupLayoutEntry>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: u64,
    pub binding_array_size: u32,
    pub kind: Option<BindingLayoutKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindingLayoutKind {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: u64,
    },
    Sampler {
        ty: SamplerBindingType,
    },
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferBindingType {
    Uniform,
    Storage,
    ReadOnlyStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureSampleType {
    Float,
    UnfilterableFloat,
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    pub(crate) inner: Arc<BindGroupLayoutInner>,
}

#[derive(Debug)]
pub(crate) struct BindGroupLayoutInner {
    pub(crate) entries: Vec<BindGroupLayoutEntry>,
    pub(crate) is_error: bool,
    pub(crate) is_default: bool,
}

impl BindGroupLayout {
    pub(crate) fn new(
        entries: Vec<BindGroupLayoutEntry>,
        is_error: bool,
        is_default: bool,
    ) -> Self {
        Self {
            inner: Arc::new(BindGroupLayoutInner {
                entries,
                is_error,
                is_default,
            }),
        }
    }

    #[must_use]
    pub fn error() -> Self {
        Self::new(Vec::new(), true, false)
    }

    #[must_use]
    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.inner.entries
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn is_default(&self) -> bool {
        self.inner.is_default
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    #[test]
    fn bind_group_layout_accessors_pin_entries_error_and_same() {
        let device = noop_device();
        let entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: 4,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 4,
            }),
        };

        let layout = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![entry],
            error: None,
        });
        let clone = layout.clone();
        let other = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: None,
        });
        let static_error = BindGroupLayout::error();

        assert!(!layout.is_error());
        assert_eq!(layout.entries(), &[entry]);
        assert!(layout.same(&clone));
        assert!(!layout.same(&other));
        assert!(static_error.is_error());
        assert!(static_error.entries().is_empty());

        device.push_error_scope(ErrorFilter::Validation);
        let invalid = device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: 4,
                binding_array_size: 2,
                kind: Some(BindingLayoutKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 4,
                }),
            }],
            error: None,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid bind group layout should be scoped");
        assert!(invalid.is_error());
        assert_eq!(
            error.message,
            "bind group layout bindingArraySize greater than one is not supported"
        );
    }
}
