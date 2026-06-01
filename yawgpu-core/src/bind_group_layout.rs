use std::collections::BTreeSet;
use std::sync::Arc;

use crate::device::FeatureSet;
use crate::format::*;
use crate::limits::*;
use crate::shader::*;
use crate::texture_view::*;

/// Validates bind group layout descriptor and returns a descriptive error on failure.
pub(crate) fn validate_bind_group_layout_descriptor(
    entries: &[BindGroupLayoutEntry],
    limits: Limits,
    features: &FeatureSet,
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
                if ty == BufferBindingType::Storage && entry.visibility & SHADER_STAGE_VERTEX != 0 {
                    return Some(
                        "writable storage buffer bindings must not be visible to vertex shaders"
                            .to_owned(),
                    );
                }
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
                sample_type,
                view_dimension,
                multisampled,
            } => {
                if multisampled && view_dimension != TextureViewDimension::D2 {
                    return Some(
                        "multisampled texture bindings require 2D view dimension".to_owned(),
                    );
                }
                if multisampled && sample_type == TextureSampleType::Float {
                    return Some(
                        "multisampled texture bindings must not use Float sample type".to_owned(),
                    );
                }
            }
            #[cfg(feature = "tiled")]
            BindingLayoutKind::InputAttachment { .. } => {
                if entry.visibility & !SHADER_STAGE_FRAGMENT != 0 {
                    return Some(
                        "input attachment bindings must only be visible to the fragment stage"
                            .to_owned(),
                    );
                }
            }
            BindingLayoutKind::StorageTexture {
                access,
                format,
                view_dimension,
            } => {
                if matches!(
                    view_dimension,
                    TextureViewDimension::Cube | TextureViewDimension::CubeArray
                ) {
                    return Some(
                        "storage texture bindings must not use cube view dimensions".to_owned(),
                    );
                }
                if access != StorageTextureAccess::ReadOnly
                    && entry.visibility & SHADER_STAGE_VERTEX != 0
                {
                    return Some(
                        "writable storage texture bindings must not be visible to vertex shaders"
                            .to_owned(),
                    );
                }
                let Some(caps) = format.caps(features) else {
                    return Some("storage texture binding format must not be Undefined".to_owned());
                };
                if !caps.storage_capable {
                    return Some(
                        "storage texture binding format must support storage usage".to_owned(),
                    );
                }
                if access == StorageTextureAccess::ReadWrite && !caps.read_write_storage_capable {
                    return Some(
                        "storage texture binding format must support read-write storage access"
                            .to_owned(),
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

/// Describes bind group layout descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupLayoutDescriptor {
    /// Entries.
    pub entries: Vec<BindGroupLayoutEntry>,
    /// Error.
    pub error: Option<String>,
}

/// Stores layout metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    /// Binding.
    pub binding: u32,
    /// Visibility.
    pub visibility: u64,
    /// Binding array size.
    pub binding_array_size: u32,
    /// Kind.
    pub kind: Option<BindingLayoutKind>,
}

/// Enumerates binding layout kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindingLayoutKind {
    /// Buffer variant.
    Buffer {
        /// Ty variant.
        ty: BufferBindingType,
        /// Has dynamic offset variant.
        has_dynamic_offset: bool,
        /// Min binding size variant.
        min_binding_size: u64,
    },
    /// Sampler variant.
    Sampler {
        /// Ty variant.
        ty: SamplerBindingType,
    },
    /// Texture variant.
    Texture {
        /// Sample type variant.
        sample_type: TextureSampleType,
        /// View dimension variant.
        view_dimension: TextureViewDimension,
        /// Multisampled variant.
        multisampled: bool,
    },
    /// Input attachment variant.
    #[cfg(feature = "tiled")]
    InputAttachment {
        /// Sample type variant.
        sample_type: TextureSampleType,
        /// Multisampled variant.
        multisampled: bool,
    },
    /// Storage texture variant.
    StorageTexture {
        /// Access variant.
        access: StorageTextureAccess,
        /// Format variant.
        format: TextureFormat,
        /// View dimension variant.
        view_dimension: TextureViewDimension,
    },
}

/// Enumerates buffer binding type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferBindingType {
    /// Uniform variant.
    Uniform,
    /// Storage variant.
    Storage,
    /// Read only storage variant.
    ReadOnlyStorage,
}

/// Enumerates sampler binding type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplerBindingType {
    /// Filtering variant.
    Filtering,
    /// Non filtering variant.
    NonFiltering,
    /// Comparison variant.
    Comparison,
}

/// Enumerates texture sample type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureSampleType {
    /// Float variant.
    Float,
    /// Unfilterable float variant.
    UnfilterableFloat,
    /// Depth variant.
    Depth,
    /// Sint variant.
    Sint,
    /// Uint variant.
    Uint,
}

/// Enumerates storage texture access values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageTextureAccess {
    /// Write only variant.
    WriteOnly,
    /// Read only variant.
    ReadOnly,
    /// Read write variant.
    ReadWrite,
}

/// Stores layout metadata.
#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    pub(crate) inner: Arc<BindGroupLayoutInner>,
}

/// Stores layout metadata.
#[derive(Debug)]
pub(crate) struct BindGroupLayoutInner {
    pub(crate) entries: Vec<BindGroupLayoutEntry>,
    pub(crate) is_error: bool,
    pub(crate) is_default: bool,
}

impl BindGroupLayout {
    /// Creates a new instance.
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

    /// Creates an error bind group layout sentinel, returned after a failed creation.
    #[must_use]
    pub fn error() -> Self {
        Self::new(Vec::new(), true, false)
    }

    /// Creates an empty default bind group layout returned for unused pipeline layout slots.
    #[must_use]
    pub fn empty_default() -> Self {
        Self::new(Vec::new(), false, true)
    }

    /// Returns the entries.
    #[must_use]
    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.inner.entries
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns true when this object is default.
    #[must_use]
    pub(crate) fn is_default(&self) -> bool {
        self.inner.is_default
    }

    /// Returns true when both handles share the same backing object.
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
