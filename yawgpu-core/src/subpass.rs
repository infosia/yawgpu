use std::sync::Arc;

use parking_lot::Mutex;

use yawgpu_hal::{HalDevice, HalSubpassRenderPass};

use crate::adapter::tiled_features_supported;
use crate::command_encoder::*;
use crate::copy::*;
use crate::device::Device;
use crate::error::ErrorKind;
use crate::extent::*;
use crate::format::*;
use crate::texture_view::*;
use crate::transient_attachment::*;

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

/// Enumerates subpass attachment resources.
#[derive(Debug, Clone)]
pub enum SubpassAttachmentResource {
    /// Persistent texture view.
    Persistent {
        /// View.
        view: Arc<TextureView>,
        /// Resolve target.
        resolve_target: Option<Arc<TextureView>>,
    },
    /// Transient attachment.
    Transient(Arc<TransientAttachment>),
}

/// Describes one color attachment binding.
#[derive(Debug, Clone)]
pub struct SubpassColorAttachmentBinding {
    /// Resource.
    pub resource: SubpassAttachmentResource,
    /// Load op.
    pub load_op: LoadOp,
    /// Store op.
    pub store_op: StoreOp,
    /// Clear value.
    pub clear_value: Color,
}

/// Describes one depth-stencil attachment binding.
#[derive(Debug, Clone)]
pub struct SubpassDepthStencilAttachmentBinding {
    /// Resource.
    pub resource: SubpassAttachmentResource,
    /// Depth load op.
    pub depth_load_op: LoadOp,
    /// Depth store op.
    pub depth_store_op: StoreOp,
    /// Depth clear value.
    pub depth_clear_value: f32,
    /// Stencil load op.
    pub stencil_load_op: LoadOp,
    /// Stencil store op.
    pub stencil_store_op: StoreOp,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
}

/// Describes a subpass render pass begin operation.
#[derive(Debug, Clone)]
pub struct SubpassRenderPassDescriptor {
    /// Pass layout.
    pub pass_layout: Arc<SubpassPassLayout>,
    /// Pass extent.
    pub extent: Extent3d,
    /// Color attachments by slot.
    pub color_attachments: Vec<SubpassColorAttachmentBinding>,
    /// Depth-stencil attachment.
    pub depth_stencil_attachment: Option<SubpassDepthStencilAttachmentBinding>,
    /// Descriptor error from FFI conversion.
    pub error: Option<String>,
}

/// Stores subpass render pass state.
#[derive(Debug, Clone)]
pub struct SubpassRenderPass {
    inner: Arc<SubpassRenderPassInner>,
}

/// Holds subpass render pass shared state.
#[derive(Debug)]
pub(crate) struct SubpassRenderPassInner {
    encoder: CommandEncoder,
    token: PassToken,
    layout: Arc<SubpassPassLayout>,
    color_attachments: Vec<SubpassColorAttachmentBinding>,
    depth_stencil_attachment: Option<SubpassDepthStencilAttachmentBinding>,
    hal: Mutex<Option<HalSubpassRenderPass>>,
    active_subpass: Mutex<u32>,
    ended: Mutex<bool>,
    is_error: bool,
}

impl SubpassRenderPass {
    /// Creates a new subpass render pass.
    pub(crate) fn new(
        encoder: CommandEncoder,
        token: PassToken,
        descriptor: SubpassRenderPassDescriptor,
        hal: Option<HalSubpassRenderPass>,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(SubpassRenderPassInner {
                encoder,
                token,
                layout: Arc::clone(&descriptor.pass_layout),
                color_attachments: descriptor.color_attachments,
                depth_stencil_attachment: descriptor.depth_stencil_attachment,
                hal: Mutex::new(hal),
                active_subpass: Mutex::new(0),
                ended: Mutex::new(false),
                is_error,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the active subpass index.
    #[must_use]
    pub fn active_subpass(&self) -> u32 {
        *self.inner.active_subpass.lock()
    }

    /// Returns retained color attachment count.
    #[must_use]
    pub fn color_attachment_count(&self) -> usize {
        self.inner.color_attachments.len()
    }

    /// Returns true when a depth-stencil attachment is retained.
    #[must_use]
    pub fn has_depth_stencil_attachment(&self) -> bool {
        self.inner.depth_stencil_attachment.is_some()
    }

    /// Advances to the next subpass.
    pub fn next_subpass(&self) -> Option<String> {
        if *self.inner.ended.lock() {
            return Some("subpass render pass has already ended".to_owned());
        }
        if self.is_error() {
            return None;
        }
        let subpass_count = self.inner.layout.descriptor().subpasses.len() as u32;
        let mut active = self.inner.active_subpass.lock();
        if active.saturating_add(1) >= subpass_count {
            self.inner
                .encoder
                .record_first_error("subpass render pass cannot advance past the last subpass");
            return None;
        }
        if let Some(pass) = self.inner.hal.lock().as_mut() {
            if let Err(error) = pass.next_subpass() {
                let message = error.to_string();
                self.inner.encoder.record_first_error(message.clone());
                return Some(message);
            }
        }
        *active += 1;
        None
    }

    /// Ends the subpass render pass.
    pub fn end(&self) -> Option<String> {
        let mut ended = self.inner.ended.lock();
        if *ended {
            return None;
        }
        *ended = true;
        if let Some(pass) = self.inner.hal.lock().take() {
            if let Err(error) = pass.end() {
                let message = error.to_string();
                self.inner.encoder.record_first_error(message.clone());
                self.inner.encoder.end_pass(self.inner.token);
                return Some(message);
            }
        }
        self.inner.encoder.end_pass(self.inner.token);
        None
    }
}

impl Drop for SubpassRenderPassInner {
    fn drop(&mut self) {
        if !*self.ended.lock() {
            if let Some(pass) = self.hal.lock().take() {
                let _ = pass.end();
            }
            self.encoder.end_pass(self.token);
        }
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
    let caps = crate::TiledCapabilities {
        max_subpasses: 4,
        max_subpass_color_attachments: device.limits().max_color_attachments,
        max_input_attachments: device.limits().max_sampled_textures_per_shader_stage,
        estimated_tile_memory_bytes: 0,
    };
    if descriptor.subpasses.is_empty() {
        return Some("subpass pass layout requires at least one subpass".to_owned());
    }
    if descriptor.subpasses.len() > caps.max_subpasses as usize {
        return Some("subpass count exceeds tiled capabilities".to_owned());
    }
    for (subpass_index, subpass) in descriptor.subpasses.iter().enumerate() {
        if subpass.color_attachment_indices.len() > caps.max_subpass_color_attachments as usize {
            return Some("subpass color attachment count exceeds tiled capabilities".to_owned());
        }
        if subpass.input_attachments.len() > caps.max_input_attachments as usize {
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

/// Validates a subpass render pass descriptor.
pub(crate) fn validate_subpass_render_pass_descriptor(
    descriptor: &SubpassRenderPassDescriptor,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    let layout = descriptor.pass_layout.descriptor();
    if descriptor.pass_layout.is_error() {
        return Some("subpass render pass layout must not be an error layout".to_owned());
    }
    if descriptor.extent.width == 0
        || descriptor.extent.height == 0
        || descriptor.extent.depth_or_array_layers != 1
    {
        return Some("subpass render pass extent must be non-zero 2D".to_owned());
    }
    if descriptor.color_attachments.len() != layout.color_attachments.len() {
        return Some("subpass render pass color attachment count must match layout".to_owned());
    }
    if descriptor.depth_stencil_attachment.is_some() != layout.depth_stencil_attachment.is_some() {
        return Some("subpass render pass depth-stencil attachment must match layout".to_owned());
    }
    for subpass in &layout.subpasses {
        for &index in &subpass.color_attachment_indices {
            if descriptor.color_attachments.get(index as usize).is_none() {
                return Some("subpass render pass missing used color attachment".to_owned());
            }
        }
    }
    None
}

/// Resolves match-target transients and validates all retained attachment resources.
pub(crate) fn resolve_subpass_render_pass_resources(
    _device: &Device,
    hal: &HalDevice,
    descriptor: &SubpassRenderPassDescriptor,
) -> Result<(), String> {
    if !tiled_features_supported(hal.backend())
        && !matches!(hal.backend(), yawgpu_hal::HalBackend::Noop)
    {
        return Err("subpass render pass backend is unsupported".to_owned());
    }
    for attachment in &descriptor.color_attachments {
        resolve_resource(hal, &attachment.resource, descriptor.extent)?;
    }
    if let Some(depth) = &descriptor.depth_stencil_attachment {
        resolve_resource(hal, &depth.resource, descriptor.extent)?;
    }
    Ok(())
}

fn resolve_resource(
    hal: &HalDevice,
    resource: &SubpassAttachmentResource,
    extent: Extent3d,
) -> Result<(), String> {
    match resource {
        SubpassAttachmentResource::Persistent {
            view,
            resolve_target,
        } => {
            if view.is_error() {
                return Err("subpass render pass cannot use an error texture view".to_owned());
            }
            if let Some(resolve_target) = resolve_target {
                if resolve_target.is_error() {
                    return Err("subpass render pass cannot use an error resolve target".to_owned());
                }
            }
        }
        SubpassAttachmentResource::Transient(attachment) => {
            if attachment.is_error() {
                return Err(
                    "subpass render pass cannot use an error transient attachment".to_owned(),
                );
            }
            if matches!(attachment.descriptor().size, TransientSizeMode::MatchTarget) {
                let descriptor = attachment.descriptor();
                let hal_descriptor =
                    hal_transient_attachment_descriptor(&descriptor, extent.width, extent.height);
                let hal_attachment = hal
                    .create_transient_attachment(&hal_descriptor)
                    .map_err(|error| error.to_string())?;
                attachment.set_hal_for_match_target(hal_attachment);
            }
        }
    }
    Ok(())
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
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

    fn attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: rgba8_unorm(),
            sample_count: 1,
        }
    }

    fn two_subpass_layout_descriptor() -> SubpassPassLayoutDescriptor {
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

    fn render_attachment_view(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT,
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
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn persistent_color(device: &Device) -> SubpassColorAttachmentBinding {
        SubpassColorAttachmentBinding {
            resource: SubpassAttachmentResource::Persistent {
                view: render_attachment_view(device),
                resolve_target: None,
            },
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_value: black(),
        }
    }

    fn black() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    #[test]
    fn subpass_pass_layout_validates_inputs_and_counts() {
        let device = noop_device();

        let valid = device.create_subpass_pass_layout(two_subpass_layout_descriptor());
        assert!(!valid.is_error());
        assert_eq!(valid.descriptor().subpasses.len(), 2);

        device.push_error_scope(ErrorFilter::Validation);
        let mut bad_source = two_subpass_layout_descriptor();
        bad_source.subpasses[1].input_attachments[0].source_subpass = 1;
        let layout = device.create_subpass_pass_layout(bad_source);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("bad source should be scoped");
        assert!(layout.is_error());
        assert_eq!(
            scoped.message,
            "subpass input sourceSubpass must refer to a prior subpass"
        );

        device.push_error_scope(ErrorFilter::Validation);
        let mut too_many = two_subpass_layout_descriptor();
        too_many.subpasses = vec![
            SubpassLayoutDesc {
                color_attachment_indices: Vec::new(),
                uses_depth_stencil: false,
                input_attachments: Vec::new(),
            };
            5
        ];
        let layout = device.create_subpass_pass_layout(too_many);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("over-capability count should be scoped");
        assert!(layout.is_error());
        assert_eq!(scoped.message, "subpass count exceeds tiled capabilities");
    }

    #[test]
    fn subpass_render_pass_requires_first_encoder_operation() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        encoder.copy_buffer_to_buffer(Arc::clone(&buffer), 0, Arc::clone(&buffer), 0, 0);

        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(pass.is_error());
        drop(pass);
        let (command_buffer, _) = encoder.finish();
        assert!(command_buffer.is_error());

        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(!pass.is_error());
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn subpass_render_pass_validates_attachment_consistency() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(pass.is_error());
        drop(pass);
        let (command_buffer, _) = encoder.finish();
        assert!(command_buffer.is_error());
    }

    #[test]
    fn subpass_render_pass_lifecycle_retains_resources_and_resolves_match_target() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let transient = Arc::new(device.create_transient_attachment(
            TransientAttachmentDescriptor {
                format: rgba8_unorm(),
                size: TransientSizeMode::MatchTarget,
                sample_count: 1,
            },
        ));
        assert!(transient.hal().is_none());
        let before_count = Arc::strong_count(&transient);
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![
                    persistent_color(&device),
                    SubpassColorAttachmentBinding {
                        resource: SubpassAttachmentResource::Transient(Arc::clone(&transient)),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Discard,
                        clear_value: black(),
                    },
                ],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(!pass.is_error());
        assert!(transient.hal().is_some());
        assert!(Arc::strong_count(&transient) > before_count);
        assert_eq!(pass.color_attachment_count(), 2);
        assert!(!pass.has_depth_stencil_attachment());
        assert_eq!(pass.active_subpass(), 0);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.active_subpass(), 1);
        assert_eq!(pass.end(), None);

        let encoder = device.create_command_encoder();
        let (drop_pass, _) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        drop(drop_pass);
    }
}
