use std::sync::Arc;

use crate::adapter::{tiled_features_supported, TiledCapabilities};
use crate::device::Device;
use crate::error::ErrorKind;
use crate::format::TextureFormat;

/// Sentinel source attachment index for the depth-stencil attachment.
pub const DEPTH_STENCIL_ATTACHMENT_INDEX: u32 = u32::MAX;

/// Describes one subpass attachment slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttachmentLayout {
    /// Format.
    pub format: TextureFormat,
    /// Sample count.
    pub sample_count: u32,
}

/// Describes one input attachment source mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubpassInputAttachment {
    /// Bind group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Source subpass.
    pub source_subpass: u32,
    /// Source attachment index, or `DEPTH_STENCIL_ATTACHMENT_INDEX`.
    pub source_attachment: u32,
}

/// Enumerates subpass dependency kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SubpassDependencyType {
    /// Color to input variant.
    ColorToInput,
    /// Depth to input variant.
    DepthToInput,
    /// Color and depth to input variant.
    ColorDepthToInput,
}

/// Describes one subpass dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubpassDependency {
    /// Source subpass.
    pub src_subpass: u32,
    /// Destination subpass.
    pub dst_subpass: u32,
    /// Dependency kind.
    pub dependency_type: SubpassDependencyType,
    /// Whether dependency is region-local.
    pub by_region: bool,
}

/// Describes one subpass in a pass layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubpassLayoutDesc {
    /// Color attachment slot indices used by this subpass.
    pub color_attachment_indices: Vec<u32>,
    /// Whether this subpass uses the depth-stencil slot.
    pub uses_depth_stencil: bool,
    /// Input attachment mappings for this subpass.
    pub input_attachments: Vec<SubpassInputAttachment>,
}

/// Describes a subpass pass layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubpassPassLayoutDescriptor {
    /// Color attachment slot layouts.
    pub color_attachments: Vec<AttachmentLayout>,
    /// Optional depth-stencil attachment layout.
    pub depth_stencil_attachment: Option<AttachmentLayout>,
    /// Subpasses.
    pub subpasses: Vec<SubpassLayoutDesc>,
    /// Dependencies.
    pub dependencies: Vec<SubpassDependency>,
    /// Descriptor error from FFI conversion.
    pub error: Option<String>,
}

/// Maps each input attachment of `subpass_index` to its Metal `[[color(N)]]`
/// slot: key = the input's `(group, binding)`, value = the source color
/// attachment index it reads (`SubpassInputAttachment::source_attachment`).
#[allow(dead_code)]
pub(crate) fn compute_subpass_color_slots(
    layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
) -> Vec<((u32, u32), u32)> {
    layout
        .subpasses
        .get(subpass_index as usize)
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| ((input.group, input.binding), input.source_attachment))
                .collect()
        })
        .unwrap_or_default()
}

/// Stores a reusable subpass pass layout.
#[derive(Debug, Clone)]
pub struct SubpassPassLayout {
    inner: Arc<SubpassPassLayoutInner>,
}

/// Holds shared subpass pass layout state.
#[derive(Debug)]
pub(crate) struct SubpassPassLayoutInner {
    pub(crate) descriptor: SubpassPassLayoutDescriptor,
    pub(crate) is_error: bool,
}

impl SubpassPassLayout {
    /// Creates a new layout.
    #[must_use]
    pub(crate) fn new(descriptor: SubpassPassLayoutDescriptor, is_error: bool) -> Self {
        Self {
            inner: Arc::new(SubpassPassLayoutInner {
                descriptor,
                is_error,
            }),
        }
    }

    /// Returns the descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &SubpassPassLayoutDescriptor {
        &self.inner.descriptor
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns true when both handles share the same backing object.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Validates a subpass pass layout descriptor.
pub(crate) fn validate_subpass_pass_layout_descriptor(
    device: &Device,
    descriptor: &SubpassPassLayoutDescriptor,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    let caps = tiled_capabilities_for_device(device);
    validate_subpass_pass_layout_descriptor_with_caps(descriptor, caps)
}

fn tiled_capabilities_for_device(device: &Device) -> TiledCapabilities {
    if !tiled_features_supported(device.hal().backend()) {
        return TiledCapabilities {
            max_subpasses: 0,
            max_subpass_color_attachments: 0,
            max_input_attachments: 0,
            estimated_tile_memory_bytes: 0,
        };
    }
    let limits = device.limits();
    TiledCapabilities {
        max_subpasses: 4,
        max_subpass_color_attachments: limits.max_color_attachments,
        max_input_attachments: limits.max_color_attachments,
        estimated_tile_memory_bytes: 256 * 1024,
    }
}

fn validate_subpass_pass_layout_descriptor_with_caps(
    descriptor: &SubpassPassLayoutDescriptor,
    caps: TiledCapabilities,
) -> Option<String> {
    let enforce_caps = caps.max_subpasses != 0
        || caps.max_subpass_color_attachments != 0
        || caps.max_input_attachments != 0;
    if descriptor.subpasses.is_empty() {
        return Some("subpass pass layout requires at least one subpass".to_owned());
    }
    // Noop advertises zero tiled capabilities but still accepts subpass objects
    // so validation and lifecycle tests remain GPU-independent.
    if enforce_caps && descriptor.subpasses.len() > caps.max_subpasses as usize {
        return Some("subpass count exceeds tiled capabilities".to_owned());
    }
    for (subpass_index, subpass) in descriptor.subpasses.iter().enumerate() {
        if enforce_caps
            && subpass.color_attachment_indices.len() > caps.max_subpass_color_attachments as usize
        {
            return Some("subpass color attachment count exceeds tiled capabilities".to_owned());
        }
        if enforce_caps && subpass.input_attachments.len() > caps.max_input_attachments as usize {
            return Some("subpass input attachment count exceeds tiled capabilities".to_owned());
        }
        for &color_index in &subpass.color_attachment_indices {
            if color_index as usize >= descriptor.color_attachments.len() {
                return Some("subpass color attachment index is out of range".to_owned());
            }
        }
        for input in &subpass.input_attachments {
            if input.source_subpass >= subpass_index as u32 {
                return Some(
                    "subpass input sourceSubpass must refer to a prior subpass".to_owned(),
                );
            }
            if input.source_attachment == DEPTH_STENCIL_ATTACHMENT_INDEX {
                if descriptor.depth_stencil_attachment.is_none() {
                    return Some(
                        "subpass input depth source requires a depth-stencil attachment".to_owned(),
                    );
                }
            } else if input.source_attachment as usize >= descriptor.color_attachments.len() {
                return Some("subpass input sourceAttachment is out of range".to_owned());
            }
        }
    }
    for dependency in &descriptor.dependencies {
        if dependency.src_subpass as usize >= descriptor.subpasses.len()
            || dependency.dst_subpass as usize >= descriptor.subpasses.len()
        {
            return Some("subpass dependency index is out of range".to_owned());
        }
    }
    None
}

impl Device {
    /// Creates a subpass pass layout.
    #[must_use]
    pub fn create_subpass_pass_layout(
        &self,
        descriptor: SubpassPassLayoutDescriptor,
    ) -> SubpassPassLayout {
        if self.is_lost() {
            return SubpassPassLayout::new(descriptor, true);
        }
        let error = validate_subpass_pass_layout_descriptor(self, &descriptor);
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        SubpassPassLayout::new(descriptor, is_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{noop_device, rgba8_unorm};

    fn attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: rgba8_unorm(),
            sample_count: 1,
        }
    }

    fn depth_attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
            sample_count: 1,
        }
    }

    fn caps() -> TiledCapabilities {
        TiledCapabilities {
            max_subpasses: 4,
            max_subpass_color_attachments: 4,
            max_input_attachments: 4,
            estimated_tile_memory_bytes: 256 * 1024,
        }
    }

    fn zero_caps() -> TiledCapabilities {
        TiledCapabilities {
            max_subpasses: 0,
            max_subpass_color_attachments: 0,
            max_input_attachments: 0,
            estimated_tile_memory_bytes: 0,
        }
    }

    fn valid_two_subpass_deferred_layout() -> SubpassPassLayoutDescriptor {
        SubpassPassLayoutDescriptor {
            color_attachments: vec![attachment_layout(), attachment_layout()],
            depth_stencil_attachment: None,
            subpasses: vec![
                SubpassLayoutDesc {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: false,
                    input_attachments: Vec::new(),
                },
                SubpassLayoutDesc {
                    color_attachment_indices: vec![1],
                    uses_depth_stencil: false,
                    input_attachments: vec![SubpassInputAttachment {
                        group: 0,
                        binding: 0,
                        source_subpass: 0,
                        source_attachment: 0,
                    }],
                },
            ],
            dependencies: vec![SubpassDependency {
                src_subpass: 0,
                dst_subpass: 1,
                dependency_type: SubpassDependencyType::ColorToInput,
                by_region: true,
            }],
            error: None,
        }
    }

    #[test]
    fn subpass_pass_layout_rejects_empty_subpasses() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses.clear();

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass pass layout requires at least one subpass".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_color_attachment_index_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[0].color_attachment_indices = vec![2];

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass color attachment index is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_input_source_attachment_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment = 2;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input sourceAttachment is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_input_source_subpass_that_is_not_prior() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_subpass = 1;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input sourceSubpass must refer to a prior subpass".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_depth_input_without_depth_stencil_attachment() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment =
            DEPTH_STENCIL_ATTACHMENT_INDEX;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input depth source requires a depth-stencil attachment".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_accepts_depth_input_with_depth_stencil_attachment() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.depth_stencil_attachment = Some(depth_attachment_layout());
        descriptor.subpasses[0].uses_depth_stencil = true;
        descriptor.subpasses[1].input_attachments[0].source_attachment =
            DEPTH_STENCIL_ATTACHMENT_INDEX;
        descriptor.dependencies[0].dependency_type = SubpassDependencyType::DepthToInput;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            None
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_dependency_index_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.dependencies[0].dst_subpass = 2;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass dependency index is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_accepts_valid_two_subpass_deferred_layout() {
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &valid_two_subpass_deferred_layout(),
                caps(),
            ),
            None
        );
    }

    #[test]
    fn subpass_pass_layout_zero_caps_still_validate_well_formed_layout() {
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &valid_two_subpass_deferred_layout(),
                zero_caps(),
            ),
            None
        );

        let device = noop_device();
        assert_eq!(
            validate_subpass_pass_layout_descriptor(&device, &valid_two_subpass_deferred_layout()),
            None
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_capability_overflow_when_caps_are_enforced() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses.push(SubpassLayoutDesc {
            color_attachment_indices: Vec::new(),
            uses_depth_stencil: false,
            input_attachments: Vec::new(),
        });

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &descriptor,
                TiledCapabilities {
                    max_subpasses: 2,
                    max_subpass_color_attachments: 4,
                    max_input_attachments: 4,
                    estimated_tile_memory_bytes: 0,
                },
            ),
            Some("subpass count exceeds tiled capabilities".to_owned())
        );
    }

    #[test]
    fn compute_subpass_color_slots_maps_input_binding_to_source_attachment_one() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment = 1;

        assert_eq!(
            compute_subpass_color_slots(&descriptor, 1),
            vec![((0, 0), 1)]
        );
    }

    #[test]
    fn compute_subpass_color_slots_maps_input_binding_to_source_attachment_two() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.color_attachments.push(attachment_layout());
        descriptor.subpasses[1].input_attachments[0].source_attachment = 2;

        assert_eq!(
            compute_subpass_color_slots(&descriptor, 1),
            vec![((0, 0), 2)]
        );
    }

    #[test]
    fn compute_subpass_color_slots_returns_empty_for_out_of_range_subpass() {
        let descriptor = valid_two_subpass_deferred_layout();

        assert_eq!(compute_subpass_color_slots(&descriptor, 2), Vec::new());
    }
}
