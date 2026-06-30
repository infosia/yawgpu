use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::{HalTextureAspect, HalTextureDimension, HalTextureViewDimension};

/// Stores vulkan compute pipeline data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanComputePipeline {
    pub(super) inner: Arc<VulkanComputePipelineInner>,
}

/// Holds shared state for the vulkan compute pipeline handle.
#[derive(Debug)]
pub(super) struct VulkanComputePipelineInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(super) descriptor_bindings: Vec<HalDescriptorBinding>,
    pub(super) shader_module: vk::ShaderModule,
}

impl Drop for VulkanComputePipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for layout in &self.descriptor_set_layouts {
                self.device
                    .device
                    .destroy_descriptor_set_layout(*layout, None);
            }
            self.device
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

/// Stores vulkan render pipeline data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanRenderPipeline {
    pub(super) inner: Arc<VulkanRenderPipelineInner>,
}

/// Holds shared state for the vulkan render pipeline handle.
#[derive(Debug)]
pub(super) struct VulkanRenderPipelineInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) pipeline: vk::Pipeline,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) render_pass: vk::RenderPass,
    pub(super) render_pass_owned: bool,
    pub(super) descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(super) descriptor_bindings: Vec<HalDescriptorBinding>,
    pub(super) vertex_shader_module: vk::ShaderModule,
    pub(super) fragment_shader_module: Option<vk::ShaderModule>,
}

impl Drop for VulkanRenderPipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            if self.render_pass_owned {
                self.device
                    .device
                    .destroy_render_pass(self.render_pass, None);
            }
            for layout in &self.descriptor_set_layouts {
                self.device
                    .device
                    .destroy_descriptor_set_layout(*layout, None);
            }
            if let Some(fragment_shader_module) = self.fragment_shader_module {
                self.device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            self.device
                .device
                .destroy_shader_module(self.vertex_shader_module, None);
        }
    }
}

/// Creates compute pipeline and reports validation errors through the owning device.
pub(super) fn create_compute_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    entry_point: &str,
    bindings: &[HalDescriptorBinding],
) -> Result<VulkanComputePipeline, HalError> {
    let HalShaderSource::SpirV(code) = shader else {
        return Err(shader_error("Vulkan compute pipeline requires SPIR-V"));
    };
    let entry_point =
        CString::new(entry_point).map_err(|_| shader_error("compute entry point contains NUL"))?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);
    let shader_module = unsafe { device.device.create_shader_module(&shader_info, None) }
        .map_err(|_| shader_error("shader module creation failed"))?;
    let descriptor_set_layouts =
        match create_descriptor_set_layouts(&device, bindings, vk::ShaderStageFlags::COMPUTE) {
            Ok(layouts) => layouts,
            Err(error) => {
                unsafe {
                    device.device.destroy_shader_module(shader_module, None);
                }
                return Err(error);
            }
        };
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
    let pipeline_layout = match unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(_) => {
            unsafe {
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device.device.destroy_shader_module(shader_module, None);
            }
            return Err(shader_error("pipeline layout creation failed"));
        }
    };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_point);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipelines = match unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    } {
        Ok(pipelines) => pipelines,
        Err((pipelines, _)) => {
            unsafe {
                for pipeline in pipelines {
                    device.device.destroy_pipeline(pipeline, None);
                }
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device.device.destroy_shader_module(shader_module, None);
            }
            return Err(shader_error("compute pipeline creation failed"));
        }
    };
    let Some(&pipeline) = pipelines.first() else {
        unsafe {
            device.device.destroy_pipeline_layout(pipeline_layout, None);
            destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
            device.device.destroy_shader_module(shader_module, None);
        }
        return Err(shader_error(
            "compute pipeline creation returned no pipeline",
        ));
    };
    Ok(VulkanComputePipeline {
        inner: Arc::new(VulkanComputePipelineInner {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            descriptor_bindings: bindings.to_vec(),
            shader_module,
        }),
    })
}

/// Creates render pipeline and reports validation errors through the owning device.
pub(super) fn create_render_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: Option<&str>,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
) -> Result<VulkanRenderPipeline, HalError> {
    let HalShaderSource::SpirVStages { vertex, fragment } = shader else {
        return Err(shader_error(
            "Vulkan render pipeline requires render SPIR-V stages",
        ));
    };
    let vertex_entry = CString::new(vertex_entry_point)
        .map_err(|_| shader_error("vertex entry point contains NUL"))?;
    let fragment_entry = fragment_entry_point
        .map(CString::new)
        .transpose()
        .map_err(|_| shader_error("fragment entry point contains NUL"))?;
    let vertex_shader_module = create_shader_module(&device, &vertex)?;
    let fragment_shader_module = match fragment
        .as_deref()
        .map(|code| create_shader_module(&device, code))
        .transpose()
    {
        Ok(module) => module,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    if fragment_entry.is_some() != fragment_shader_module.is_some() {
        unsafe {
            if let Some(fragment_shader_module) = fragment_shader_module {
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            device
                .device
                .destroy_shader_module(vertex_shader_module, None);
        }
        return Err(shader_error(
            "Vulkan render pipeline fragment entry and SPIR-V stage must match",
        ));
    }
    let shader_stage_flags = if fragment_shader_module.is_some() {
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
    } else {
        vk::ShaderStageFlags::VERTEX
    };
    let descriptor_set_layouts =
        match create_descriptor_set_layouts(&device, bindings, shader_stage_flags) {
            Ok(layouts) => layouts,
            Err(error) => {
                unsafe {
                    if let Some(fragment_shader_module) = fragment_shader_module {
                        device
                            .device
                            .destroy_shader_module(fragment_shader_module, None);
                    }
                    device
                        .device
                        .destroy_shader_module(vertex_shader_module, None);
                }
                return Err(error);
            }
        };
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
    let pipeline_layout = match unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(_) => {
            unsafe {
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                if let Some(fragment_shader_module) = fragment_shader_module {
                    device
                        .device
                        .destroy_shader_module(fragment_shader_module, None);
                }
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(shader_error("render pipeline layout creation failed"));
        }
    };
    let render_pass = match create_render_pass(&device, descriptor, bindings) {
        Ok(render_pass) => render_pass,
        Err(error) => {
            unsafe {
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                if let Some(fragment_shader_module) = fragment_shader_module {
                    device
                        .device
                        .destroy_shader_module(fragment_shader_module, None);
                }
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    let pipeline = match create_graphics_pipeline(
        &device,
        descriptor,
        pipeline_layout,
        render_pass,
        0,
        vertex_shader_module,
        fragment_shader_module,
        &vertex_entry,
        fragment_entry.as_deref(),
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                device.device.destroy_render_pass(render_pass, None);
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                if let Some(fragment_shader_module) = fragment_shader_module {
                    device
                        .device
                        .destroy_shader_module(fragment_shader_module, None);
                }
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    Ok(VulkanRenderPipeline {
        inner: Arc::new(VulkanRenderPipelineInner {
            device,
            pipeline,
            pipeline_layout,
            render_pass,
            render_pass_owned: true,
            descriptor_set_layouts,
            descriptor_bindings: bindings.to_vec(),
            vertex_shader_module,
            fragment_shader_module,
        }),
    })
}

/// Creates a subpass-compatible render pipeline.
#[cfg(feature = "tiled")]
#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn create_subpass_render_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: Option<&str>,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
    pass_layout: &HalSubpassPassLayout,
    subpass_index: u32,
) -> Result<VulkanRenderPipeline, HalError> {
    let HalShaderSource::SpirVStages { vertex, fragment } = shader else {
        return Err(shader_error(
            "Vulkan subpass render pipeline requires render SPIR-V stages",
        ));
    };
    let vertex_entry = CString::new(vertex_entry_point)
        .map_err(|_| shader_error("vertex entry point contains NUL"))?;
    let fragment_entry = fragment_entry_point
        .map(CString::new)
        .transpose()
        .map_err(|_| shader_error("fragment entry point contains NUL"))?;
    let vertex_shader_module = create_shader_module(&device, &vertex)?;
    let fragment_shader_module = match fragment
        .as_deref()
        .map(|code| create_shader_module(&device, code))
        .transpose()
    {
        Ok(module) => module,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    if fragment_entry.is_some() != fragment_shader_module.is_some() {
        unsafe {
            if let Some(fragment_shader_module) = fragment_shader_module {
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            device
                .device
                .destroy_shader_module(vertex_shader_module, None);
        }
        return Err(shader_error(
            "Vulkan subpass render pipeline fragment entry and SPIR-V stage must match",
        ));
    }
    let shader_stage_flags = if fragment_shader_module.is_some() {
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
    } else {
        vk::ShaderStageFlags::VERTEX
    };
    let descriptor_set_layouts =
        match create_descriptor_set_layouts(&device, bindings, shader_stage_flags) {
            Ok(layouts) => layouts,
            Err(error) => {
                unsafe {
                    if let Some(fragment_shader_module) = fragment_shader_module {
                        device
                            .device
                            .destroy_shader_module(fragment_shader_module, None);
                    }
                    device
                        .device
                        .destroy_shader_module(vertex_shader_module, None);
                }
                return Err(error);
            }
        };
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
    let pipeline_layout = match unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(_) => {
            unsafe {
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                if let Some(fragment_shader_module) = fragment_shader_module {
                    device
                        .device
                        .destroy_shader_module(fragment_shader_module, None);
                }
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(shader_error("render pipeline layout creation failed"));
        }
    };
    let render_pass =
        match super::encode::create_subpass_render_pass_for_layout(&device.device, pass_layout) {
            Ok(render_pass) => render_pass,
            Err(error) => {
                unsafe {
                    device.device.destroy_pipeline_layout(pipeline_layout, None);
                    destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                    if let Some(fragment_shader_module) = fragment_shader_module {
                        device
                            .device
                            .destroy_shader_module(fragment_shader_module, None);
                    }
                    device
                        .device
                        .destroy_shader_module(vertex_shader_module, None);
                }
                return Err(error);
            }
        };
    let pipeline = match create_graphics_pipeline(
        &device,
        descriptor,
        pipeline_layout,
        render_pass,
        subpass_index,
        vertex_shader_module,
        fragment_shader_module,
        &vertex_entry,
        fragment_entry.as_deref(),
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                device.device.destroy_render_pass(render_pass, None);
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                if let Some(fragment_shader_module) = fragment_shader_module {
                    device
                        .device
                        .destroy_shader_module(fragment_shader_module, None);
                }
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    Ok(VulkanRenderPipeline {
        inner: Arc::new(VulkanRenderPipelineInner {
            device,
            pipeline,
            pipeline_layout,
            render_pass,
            render_pass_owned: true,
            descriptor_set_layouts,
            descriptor_bindings: bindings.to_vec(),
            vertex_shader_module,
            fragment_shader_module,
        }),
    })
}

/// Creates shader module and reports validation errors through the owning device.
pub(super) fn create_shader_module(
    device: &VulkanDeviceInner,
    code: &[u32],
) -> Result<vk::ShaderModule, HalError> {
    let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.device.create_shader_module(&shader_info, None) }
        .map_err(|_| shader_error("shader module creation failed"))
}

/// Creates render pass and reports validation errors through the owning device.
pub(super) fn create_render_pass(
    device: &VulkanDeviceInner,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
) -> Result<vk::RenderPass, HalError> {
    create_render_pass_for_descriptor(&device.device, descriptor, bindings)
}

fn create_render_pass_for_descriptor(
    device: &ash::Device,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
) -> Result<vk::RenderPass, HalError> {
    if !descriptor.color_targets.iter().any(Option::is_some) && descriptor.depth_stencil.is_none() {
        return Err(shader_error(
            "render pipeline requires a color target or depth-stencil state",
        ));
    }
    let mut attachments = Vec::new();
    let mut color_references = Vec::new();
    let framebuffer_fetch_color_slots = framebuffer_fetch_color_slots(bindings);
    for (slot, color_target) in descriptor.color_targets.iter().enumerate() {
        let Some(color_target) = color_target else {
            color_references.push(
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED),
            );
            continue;
        };
        let (format, _) = map_texture_format(color_target.format)?;
        let color_slot =
            u32::try_from(slot).map_err(|_| shader_error("color attachment slot is too large"))?;
        let framebuffer_fetch = framebuffer_fetch_color_slots.contains(&color_slot);
        let index = u32::try_from(attachments.len())
            .map_err(|_| shader_error("color attachment index is too large"))?;
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(descriptor.sample_count)?)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(super::encode::render_color_attachment_layout(
                    framebuffer_fetch,
                ))
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        );
        color_references.push(vk::AttachmentReference::default().attachment(index).layout(
            super::encode::render_color_attachment_layout(framebuffer_fetch),
        ));
    }
    let color_target_present = descriptor
        .color_targets
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let input_references = super::encode::input_attachment_references(
        &framebuffer_fetch_color_slots,
        &color_target_present,
        &color_references,
    )?;
    let depth_reference = if let Some(depth_stencil) = descriptor.depth_stencil {
        let (format, _) = map_texture_format(depth_stencil.format)?;
        let has_depth = format_has_depth_aspect(depth_stencil.format);
        let has_stencil = format_has_stencil_aspect(depth_stencil.format);
        let index = u32::try_from(attachments.len())
            .map_err(|_| shader_error("depth attachment index is too large"))?;
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(descriptor.sample_count)?)
                .load_op(if has_depth {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::DONT_CARE
                })
                .store_op(if has_depth {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .stencil_load_op(if has_stencil {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::DONT_CARE
                })
                .stencil_store_op(if has_stencil {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        Some(
            vk::AttachmentReference::default()
                .attachment(index)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        )
    } else {
        None
    };
    let vk_subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references);
    let vk_subpass = if input_references.is_empty() {
        vk_subpass
    } else {
        vk_subpass.input_attachments(&input_references)
    };
    let vk_subpass = if let Some(depth_reference) = depth_reference.as_ref() {
        vk_subpass.depth_stencil_attachment(depth_reference)
    } else {
        vk_subpass
    };
    let vk_subpasses = [vk_subpass];
    let dependencies = super::encode::render_pass_dependencies(!input_references.is_empty());
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&vk_subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("render pass creation failed"))
}

fn framebuffer_fetch_color_slots(bindings: &[HalDescriptorBinding]) -> Vec<u32> {
    let mut slots = bindings
        .iter()
        .filter_map(|binding| match binding.kind {
            HalDescriptorBindingKind::InputAttachment { color_slot } => Some(color_slot),
            _ => None,
        })
        .collect::<Vec<_>>();
    slots.sort_unstable();
    slots.dedup();
    slots
}

fn vk_sample_count(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(shader_error("unsupported render pipeline sample count")),
    }
}

fn is_strip_topology(topology: HalPrimitiveTopology) -> bool {
    matches!(
        topology,
        HalPrimitiveTopology::LineStrip | HalPrimitiveTopology::TriangleStrip
    )
}

fn depth_clamp_and_clip(depth_clip_control: bool, unclipped_depth: bool) -> (bool, Option<bool>) {
    if depth_clip_control {
        (true, Some(!unclipped_depth))
    } else {
        (unclipped_depth, None)
    }
}

/// Creates graphics pipeline and reports validation errors through the owning device.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_graphics_pipeline(
    device: &VulkanDeviceInner,
    descriptor: &HalRenderPipelineDescriptor,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    subpass_index: u32,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: Option<vk::ShaderModule>,
    vertex_entry: &CStr,
    fragment_entry: Option<&CStr>,
) -> Result<vk::Pipeline, HalError> {
    let mut shader_stages = vec![vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(vertex_entry)];
    if let (Some(fragment_shader_module), Some(fragment_entry)) =
        (fragment_shader_module, fragment_entry)
    {
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(fragment_entry),
        );
    }
    let binding_descriptions = descriptor
        .vertex_buffers
        .iter()
        .map(|layout| {
            Ok(vk::VertexInputBindingDescription::default()
                .binding(layout.slot)
                .stride(
                    u32::try_from(layout.array_stride)
                        .map_err(|_| shader_error("vertex array stride is too large"))?,
                )
                .input_rate(match layout.step_mode {
                    HalVertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    HalVertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                }))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    let mut attribute_descriptions = Vec::new();
    for layout in &descriptor.vertex_buffers {
        for attribute in &layout.attributes {
            attribute_descriptions.push(
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.shader_location)
                    .binding(layout.slot)
                    .format(map_vertex_format(attribute.format)?)
                    .offset(
                        u32::try_from(attribute.offset)
                            .map_err(|_| shader_error("vertex attribute offset is too large"))?,
                    ),
            );
        }
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(map_primitive_topology(descriptor.primitive_topology))
        .primitive_restart_enable(is_strip_topology(descriptor.primitive_topology));
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let has_depth_bias = descriptor.depth_stencil.is_some_and(|depth_stencil| {
        depth_stencil.depth_bias != 0
            || depth_stencil.depth_bias_slope_scale != 0.0
            || depth_stencil.depth_bias_clamp != 0.0
    });
    let (depth_clamp_enable, depth_clip_enable) =
        depth_clamp_and_clip(device.depth_clip_control, descriptor.unclipped_depth);
    let mut depth_clip_state = depth_clip_enable.map(|depth_clip_enable| {
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default()
            .depth_clip_enable(depth_clip_enable)
    });
    let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(depth_clamp_enable)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk_cull_mode(descriptor.cull_mode))
        .front_face(vk_front_face(descriptor.front_face))
        .depth_bias_enable(has_depth_bias)
        .depth_bias_constant_factor(
            descriptor
                .depth_stencil
                .map_or(0.0, |depth_stencil| depth_stencil.depth_bias as f32),
        )
        .depth_bias_slope_factor(
            descriptor
                .depth_stencil
                .map_or(0.0, |depth_stencil| depth_stencil.depth_bias_slope_scale),
        )
        .depth_bias_clamp(
            descriptor
                .depth_stencil
                .map_or(0.0, |depth_stencil| depth_stencil.depth_bias_clamp),
        )
        .line_width(1.0);
    if let Some(depth_clip_state) = depth_clip_state.as_mut() {
        rasterization = rasterization.push_next(depth_clip_state);
    }
    let sample_mask = [descriptor.sample_mask];
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk_sample_count(descriptor.sample_count)?)
        .sample_shading_enable(false)
        .sample_mask(&sample_mask)
        .alpha_to_coverage_enable(descriptor.alpha_to_coverage_enabled);
    let color_attachments = descriptor
        .color_targets
        .iter()
        .map(|target| target.map_or_else(color_blend_hole_attachment, color_blend_attachment))
        .collect::<Vec<_>>();
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_attachments);
    let depth_stencil = descriptor
        .depth_stencil
        .map(vk_pipeline_depth_stencil_state);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::BLEND_CONSTANTS,
        vk::DynamicState::STENCIL_REFERENCE,
    ];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(subpass_index);
    if let Some(depth_stencil) = depth_stencil.as_ref() {
        pipeline_info = pipeline_info.depth_stencil_state(depth_stencil);
    }
    let pipelines = unsafe {
        device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    };
    match pipelines {
        Ok(pipelines) => pipelines
            .first()
            .copied()
            .ok_or_else(|| shader_error("graphics pipeline creation returned no pipeline")),
        Err((pipelines, _)) => {
            unsafe {
                for pipeline in pipelines {
                    device.device.destroy_pipeline(pipeline, None);
                }
            }
            Err(shader_error("graphics pipeline creation failed"))
        }
    }
}

fn vk_front_face(front_face: HalFrontFace) -> vk::FrontFace {
    match front_face {
        HalFrontFace::Ccw => vk::FrontFace::COUNTER_CLOCKWISE,
        HalFrontFace::Cw => vk::FrontFace::CLOCKWISE,
    }
}

fn vk_cull_mode(cull_mode: HalCullMode) -> vk::CullModeFlags {
    match cull_mode {
        HalCullMode::None => vk::CullModeFlags::NONE,
        HalCullMode::Front => vk::CullModeFlags::FRONT,
        HalCullMode::Back => vk::CullModeFlags::BACK,
    }
}

fn color_blend_attachment(target: HalColorTargetState) -> vk::PipelineColorBlendAttachmentState {
    let mut attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk_color_write_mask(target.write_mask));
    if let Some(blend) = target.blend {
        attachment = attachment
            .blend_enable(true)
            .src_color_blend_factor(vk_blend_factor(blend.color.src_factor, false))
            .dst_color_blend_factor(vk_blend_factor(blend.color.dst_factor, false))
            .color_blend_op(vk_blend_operation(blend.color.operation))
            .src_alpha_blend_factor(vk_blend_factor(blend.alpha.src_factor, true))
            .dst_alpha_blend_factor(vk_blend_factor(blend.alpha.dst_factor, true))
            .alpha_blend_op(vk_blend_operation(blend.alpha.operation));
    }
    attachment
}

fn color_blend_hole_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::empty())
}

fn vk_color_write_mask(write_mask: u32) -> vk::ColorComponentFlags {
    let mut mask = vk::ColorComponentFlags::empty();
    if write_mask & 0x1 != 0 {
        mask |= vk::ColorComponentFlags::R;
    }
    if write_mask & 0x2 != 0 {
        mask |= vk::ColorComponentFlags::G;
    }
    if write_mask & 0x4 != 0 {
        mask |= vk::ColorComponentFlags::B;
    }
    if write_mask & 0x8 != 0 {
        mask |= vk::ColorComponentFlags::A;
    }
    mask
}

fn vk_blend_operation(operation: HalBlendOperation) -> vk::BlendOp {
    match operation {
        HalBlendOperation::Add => vk::BlendOp::ADD,
        HalBlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        HalBlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        HalBlendOperation::Min => vk::BlendOp::MIN,
        HalBlendOperation::Max => vk::BlendOp::MAX,
    }
}

fn vk_blend_factor(factor: HalBlendFactor, alpha: bool) -> vk::BlendFactor {
    match factor {
        HalBlendFactor::Zero => vk::BlendFactor::ZERO,
        HalBlendFactor::One => vk::BlendFactor::ONE,
        HalBlendFactor::Src => {
            if alpha {
                vk::BlendFactor::SRC_ALPHA
            } else {
                vk::BlendFactor::SRC_COLOR
            }
        }
        HalBlendFactor::OneMinusSrc => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_SRC_COLOR
            }
        }
        HalBlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        HalBlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        HalBlendFactor::Dst => {
            if alpha {
                vk::BlendFactor::DST_ALPHA
            } else {
                vk::BlendFactor::DST_COLOR
            }
        }
        HalBlendFactor::OneMinusDst => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_DST_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_DST_COLOR
            }
        }
        HalBlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        HalBlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        HalBlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        HalBlendFactor::Constant => {
            if alpha {
                vk::BlendFactor::CONSTANT_ALPHA
            } else {
                vk::BlendFactor::CONSTANT_COLOR
            }
        }
        HalBlendFactor::OneMinusConstant => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR
            }
        }
        HalBlendFactor::Src1 => {
            if alpha {
                vk::BlendFactor::SRC1_ALPHA
            } else {
                vk::BlendFactor::SRC1_COLOR
            }
        }
        HalBlendFactor::OneMinusSrc1 => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_SRC1_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_SRC1_COLOR
            }
        }
        HalBlendFactor::Src1Alpha => vk::BlendFactor::SRC1_ALPHA,
        HalBlendFactor::OneMinusSrc1Alpha => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
    }
}

fn vk_pipeline_depth_stencil_state(
    depth_stencil: HalDepthStencilState,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let depth_test_enabled = depth_stencil.depth_write_enabled
        || !matches!(depth_stencil.depth_compare, HalCompareFunction::Always);
    let stencil_enabled = stencil_face_uses_stencil(depth_stencil.stencil_front)
        || stencil_face_uses_stencil(depth_stencil.stencil_back)
        || depth_stencil.stencil_read_mask != u32::MAX
        || depth_stencil.stencil_write_mask != u32::MAX;
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test_enabled)
        .depth_write_enable(depth_stencil.depth_write_enabled)
        .depth_compare_op(map_compare_function(depth_stencil.depth_compare))
        .depth_bounds_test_enable(false)
        .stencil_test_enable(stencil_enabled)
        .front(vk_stencil_op_state(
            depth_stencil.stencil_front,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
        ))
        .back(vk_stencil_op_state(
            depth_stencil.stencil_back,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
        ))
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
}

fn vk_stencil_op_state(
    face: crate::HalStencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> vk::StencilOpState {
    vk::StencilOpState::default()
        .fail_op(map_stencil_operation(face.fail_op))
        .pass_op(map_stencil_operation(face.pass_op))
        .depth_fail_op(map_stencil_operation(face.depth_fail_op))
        .compare_op(map_compare_function(face.compare))
        .compare_mask(read_mask)
        .write_mask(write_mask)
        .reference(0)
}

fn stencil_face_uses_stencil(face: crate::HalStencilFaceState) -> bool {
    !matches!(face.compare, HalCompareFunction::Always)
        || !matches!(face.fail_op, HalStencilOperation::Keep)
        || !matches!(face.depth_fail_op, HalStencilOperation::Keep)
        || !matches!(face.pass_op, HalStencilOperation::Keep)
}

fn map_stencil_operation(operation: HalStencilOperation) -> vk::StencilOp {
    match operation {
        HalStencilOperation::Keep => vk::StencilOp::KEEP,
        HalStencilOperation::Zero => vk::StencilOp::ZERO,
        HalStencilOperation::Replace => vk::StencilOp::REPLACE,
        HalStencilOperation::Invert => vk::StencilOp::INVERT,
        HalStencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        HalStencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        HalStencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        HalStencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

/// Creates descriptor set layouts and reports validation errors through the owning device.
pub(super) fn create_descriptor_set_layouts(
    device: &VulkanDeviceInner,
    bindings: &[HalDescriptorBinding],
    stage_flags: vk::ShaderStageFlags,
) -> Result<Vec<vk::DescriptorSetLayout>, HalError> {
    let Some(max_group) = bindings.iter().map(|binding| binding.group).max() else {
        return Ok(Vec::new());
    };
    let mut layouts = Vec::new();
    for group in 0..=max_group {
        let layout_bindings = bindings
            .iter()
            .filter(|binding| binding.group == group)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(descriptor_type(binding.kind))
                    .descriptor_count(1)
                    .stage_flags(binding_stage_flags(binding.kind, stage_flags))
            })
            .collect::<Vec<_>>();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        match unsafe {
            device
                .device
                .create_descriptor_set_layout(&layout_info, None)
        } {
            Ok(layout) => layouts.push(layout),
            Err(_) => {
                unsafe {
                    destroy_descriptor_set_layouts(&device.device, &layouts);
                }
                return Err(shader_error("descriptor set layout creation failed"));
            }
        }
    }
    Ok(layouts)
}

unsafe fn destroy_descriptor_set_layouts(
    device: &ash::Device,
    layouts: &[vk::DescriptorSetLayout],
) {
    for layout in layouts {
        device.destroy_descriptor_set_layout(*layout, None);
    }
}

/// Returns descriptor type.
pub(super) fn descriptor_type(kind: HalDescriptorBindingKind) -> vk::DescriptorType {
    match kind {
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform) => {
            vk::DescriptorType::UNIFORM_BUFFER
        }
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage) => {
            vk::DescriptorType::STORAGE_BUFFER
        }
        HalDescriptorBindingKind::Texture => vk::DescriptorType::SAMPLED_IMAGE,
        HalDescriptorBindingKind::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        HalDescriptorBindingKind::Sampler => vk::DescriptorType::SAMPLER,
        HalDescriptorBindingKind::InputAttachment { .. } => vk::DescriptorType::INPUT_ATTACHMENT,
    }
}

/// Returns the descriptor set layout stage flags for one binding.
///
/// Input attachments are fragment-stage only per Vulkan descriptor layout rules.
fn binding_stage_flags(
    kind: HalDescriptorBindingKind,
    default: vk::ShaderStageFlags,
) -> vk::ShaderStageFlags {
    match kind {
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform)
        | HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage)
        | HalDescriptorBindingKind::Texture
        | HalDescriptorBindingKind::StorageTexture { .. }
        | HalDescriptorBindingKind::Sampler => default,
        HalDescriptorBindingKind::InputAttachment { .. } => vk::ShaderStageFlags::FRAGMENT,
    }
}

/// Creates compute descriptor pool and reports validation errors through the owning device.
pub(super) fn create_compute_descriptor_pool(
    device: &ash::Device,
    pipeline: &VulkanComputePipeline,
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if pipeline.inner.descriptor_set_layouts.is_empty() {
        return Ok(None);
    }
    let uniform_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform)
            )
        })
        .count();
    let storage_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage)
            )
        })
        .count();
    let texture_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalDescriptorBindingKind::Texture))
        .count();
    let storage_texture_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::StorageTexture { .. }
            )
        })
        .count();
    let sampler_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalDescriptorBindingKind::Sampler))
        .count();
    let mut pool_sizes = Vec::new();
    if uniform_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(
                    u32::try_from(uniform_count)
                        .map_err(|_| shader_error("uniform descriptor count is too large"))?,
                ),
        );
    }
    if storage_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(
                    u32::try_from(storage_count)
                        .map_err(|_| shader_error("storage descriptor count is too large"))?,
                ),
        );
    }
    if texture_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(
                    u32::try_from(texture_count)
                        .map_err(|_| shader_error("texture descriptor count is too large"))?,
                ),
        );
    }
    if storage_texture_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(
                    u32::try_from(storage_texture_count).map_err(|_| {
                        shader_error("storage texture descriptor count is too large")
                    })?,
                ),
        );
    }
    if sampler_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(
                    u32::try_from(sampler_count)
                        .map_err(|_| shader_error("sampler descriptor count is too large"))?,
                ),
        );
    }
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(
            u32::try_from(pipeline.inner.descriptor_set_layouts.len())
                .map_err(|_| shader_error("descriptor set count is too large"))?,
        )
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|_| shader_error("descriptor pool creation failed"))?;
    Ok(Some(pool))
}

/// Returns allocate compute descriptor sets.
pub(super) fn allocate_compute_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    pipeline: &VulkanComputePipeline,
) -> Result<Vec<vk::DescriptorSet>, HalError> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&pipeline.inner.descriptor_set_layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map_err(|_| shader_error("descriptor set allocation failed"))
}

/// Returns update compute descriptor sets.
pub(super) fn update_compute_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanComputePipeline,
    pass: &HalComputePass,
    descriptor_sets: &[vk::DescriptorSet],
) -> Result<Vec<vk::ImageView>, HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(Vec::new());
    }
    let mut buffer_infos = Vec::new();
    let mut image_infos = Vec::new();
    let mut image_views = Vec::new();
    let mut write_specs = Vec::new();
    let result = (|| {
        {
            let mut scratch = DescriptorUpdateScratch {
                device,
                buffer_infos: &mut buffer_infos,
                image_infos: &mut image_infos,
                image_views: &mut image_views,
            };
            for descriptor in &pipeline.inner.descriptor_bindings {
                let info = descriptor_info(
                    descriptor,
                    &pass.bind_buffers,
                    &pass.bind_textures,
                    &pass.bind_samplers,
                    &mut scratch,
                    "compute",
                )?;
                write_specs.push((
                    info,
                    descriptor.group,
                    descriptor.binding,
                    descriptor_type(descriptor.kind),
                ));
            }
        }
        let writes = write_specs
            .iter()
            .map(|(info, group, binding, descriptor_type)| {
                let group = usize::try_from(*group)
                    .map_err(|_| shader_error("descriptor group index is too large"))?;
                let descriptor_set = descriptor_sets
                    .get(group)
                    .copied()
                    .ok_or_else(|| shader_error("descriptor set is missing"))?;
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(*descriptor_type);
                Ok(match info {
                    DescriptorInfo::Buffer(index) => {
                        write.buffer_info(std::slice::from_ref(&buffer_infos[*index]))
                    }
                    DescriptorInfo::Image(index) => {
                        write.image_info(std::slice::from_ref(&image_infos[*index]))
                    }
                })
            })
            .collect::<Result<Vec<_>, HalError>>()?;
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    })();
    if let Err(error) = result {
        destroy_descriptor_image_views(device, &image_views);
        return Err(error);
    }
    Ok(image_views)
}

/// Creates render descriptor pool and reports validation errors through the owning device.
pub(super) fn create_render_descriptor_pool(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if pipeline.inner.descriptor_set_layouts.is_empty() {
        return Ok(None);
    }
    create_descriptor_pool(
        device,
        pipeline.inner.descriptor_set_layouts.len(),
        &pipeline.inner.descriptor_bindings,
    )
}

/// Returns allocate render descriptor sets.
pub(super) fn allocate_render_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    pipeline: &VulkanRenderPipeline,
) -> Result<Vec<vk::DescriptorSet>, HalError> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&pipeline.inner.descriptor_set_layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map_err(|_| shader_error("descriptor set allocation failed"))
}

/// Returns update render descriptor sets.
pub(super) fn update_render_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
    pass: &HalRenderPass,
    color_attachment_views: &[Option<vk::ImageView>],
    descriptor_sets: &[vk::DescriptorSet],
) -> Result<Vec<vk::ImageView>, HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(Vec::new());
    }
    let mut buffer_infos = Vec::new();
    let mut image_infos = Vec::new();
    let mut image_views = Vec::new();
    let mut write_specs = Vec::new();
    let result = (|| {
        {
            let mut scratch = DescriptorUpdateScratch {
                device,
                buffer_infos: &mut buffer_infos,
                image_infos: &mut image_infos,
                image_views: &mut image_views,
            };
            for descriptor in &pipeline.inner.descriptor_bindings {
                let info = match descriptor.kind {
                    HalDescriptorBindingKind::InputAttachment { color_slot } => {
                        let view = usize::try_from(color_slot)
                            .ok()
                            .and_then(|slot| color_attachment_views.get(slot).copied().flatten())
                            .ok_or_else(|| {
                                shader_error("render input attachment color target is missing")
                            })?;
                        scratch.image_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::GENERAL),
                        );
                        DescriptorInfo::Image(scratch.image_infos.len() - 1)
                    }
                    _ => descriptor_info(
                        descriptor,
                        &pass.bind_buffers,
                        &pass.bind_textures,
                        &pass.bind_samplers,
                        &mut scratch,
                        "render",
                    )?,
                };
                write_specs.push((
                    info,
                    descriptor.group,
                    descriptor.binding,
                    descriptor_type(descriptor.kind),
                ));
            }
        }
        let writes = write_specs
            .iter()
            .map(|(info, group, binding, descriptor_type)| {
                let group = usize::try_from(*group)
                    .map_err(|_| shader_error("descriptor group index is too large"))?;
                let descriptor_set = descriptor_sets
                    .get(group)
                    .copied()
                    .ok_or_else(|| shader_error("descriptor set is missing"))?;
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(*descriptor_type);
                Ok(match info {
                    DescriptorInfo::Buffer(index) => {
                        write.buffer_info(std::slice::from_ref(&buffer_infos[*index]))
                    }
                    DescriptorInfo::Image(index) => {
                        write.image_info(std::slice::from_ref(&image_infos[*index]))
                    }
                })
            })
            .collect::<Result<Vec<_>, HalError>>()?;
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    })();
    if let Err(error) = result {
        destroy_descriptor_image_views(device, &image_views);
        return Err(error);
    }
    Ok(image_views)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DescriptorInfo {
    Buffer(usize),
    Image(usize),
}

#[derive(Debug, Clone, Copy)]
struct DescriptorImageView {
    view: vk::ImageView,
    owned: bool,
}

pub(super) struct DescriptorUpdateScratch<'a> {
    pub(super) device: &'a ash::Device,
    pub(super) buffer_infos: &'a mut Vec<vk::DescriptorBufferInfo>,
    pub(super) image_infos: &'a mut Vec<vk::DescriptorImageInfo>,
    pub(super) image_views: &'a mut Vec<vk::ImageView>,
}

pub(super) fn descriptor_info(
    descriptor: &HalDescriptorBinding,
    buffers: &[HalBoundBuffer],
    textures: &[HalBoundTexture],
    samplers: &[HalBoundSampler],
    scratch: &mut DescriptorUpdateScratch<'_>,
    pass_name: &'static str,
) -> Result<DescriptorInfo, HalError> {
    match descriptor.kind {
        HalDescriptorBindingKind::Buffer(_) => {
            let bound = buffers
                .iter()
                .find(|bound| {
                    bound.group == descriptor.group && bound.binding == descriptor.binding
                })
                .ok_or_else(|| descriptor_missing_error(pass_name, "buffer"))?;
            scratch.buffer_infos.push(descriptor_buffer_info(bound)?);
            Ok(DescriptorInfo::Buffer(scratch.buffer_infos.len() - 1))
        }
        HalDescriptorBindingKind::Texture => {
            let bound = textures
                .iter()
                .find(|bound| {
                    bound.group == descriptor.group && bound.binding == descriptor.binding
                })
                .ok_or_else(|| descriptor_missing_error(pass_name, "texture"))?;
            let HalTexture::Vulkan(texture) = &bound.texture else {
                return Err(shader_error("descriptor texture is not Vulkan-backed"));
            };
            let image_view = create_sampled_texture_image_view(scratch.device, texture, bound)?;
            scratch.image_views.push(image_view);
            scratch.image_infos.push(
                vk::DescriptorImageInfo::default()
                    .image_view(image_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            );
            Ok(DescriptorInfo::Image(scratch.image_infos.len() - 1))
        }
        HalDescriptorBindingKind::StorageTexture { .. } => {
            let bound = textures
                .iter()
                .find(|bound| {
                    bound.group == descriptor.group && bound.binding == descriptor.binding
                })
                .ok_or_else(|| descriptor_missing_error(pass_name, "texture"))?;
            let HalTexture::Vulkan(texture) = &bound.texture else {
                return Err(shader_error("descriptor texture is not Vulkan-backed"));
            };
            let image_view = create_storage_texture_image_view(scratch.device, texture, bound)?;
            if image_view.owned {
                scratch.image_views.push(image_view.view);
            }
            scratch.image_infos.push(
                vk::DescriptorImageInfo::default()
                    .image_view(image_view.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            );
            Ok(DescriptorInfo::Image(scratch.image_infos.len() - 1))
        }
        HalDescriptorBindingKind::Sampler => {
            let bound = samplers
                .iter()
                .find(|bound| {
                    bound.group == descriptor.group && bound.binding == descriptor.binding
                })
                .ok_or_else(|| descriptor_missing_error(pass_name, "sampler"))?;
            let HalSampler::Vulkan(sampler) = &bound.sampler else {
                return Err(shader_error("descriptor sampler is not Vulkan-backed"));
            };
            let sampler = sampler
                ._inner
                .as_ref()
                .ok_or_else(|| shader_error("sampler allocation failed"))?;
            scratch
                .image_infos
                .push(vk::DescriptorImageInfo::default().sampler(sampler.sampler));
            Ok(DescriptorInfo::Image(scratch.image_infos.len() - 1))
        }
        HalDescriptorBindingKind::InputAttachment { .. } => Err(shader_error(
            "input attachment descriptor requires render color attachment views",
        )),
    }
}

fn create_sampled_texture_image_view(
    device: &ash::Device,
    texture: &VulkanTexture,
    bound: &HalBoundTexture,
) -> Result<vk::ImageView, HalError> {
    // A combined depth-stencil image must be viewed through its own (combined)
    // VkFormat with the desired aspect selected by the subresource aspect mask.
    // After the core fix, a DepthOnly/StencilOnly view of a combined texture
    // arrives with the aspect-specific `bound.format` (e.g. `Depth32Float` /
    // `Stencil8`); using that aspect VkFormat (e.g. `D32_SFLOAT`) for a view of a
    // `D32_SFLOAT_S8_UINT` image is invalid. Derive the VkFormat from the
    // texture's format in that case; otherwise honor the view's own format.
    let view_format = sampled_texture_view_format(texture.format, bound.format);
    let (format, _) = map_texture_format(view_format)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(texture.inner()?.image)
        .view_type(sampled_texture_view_type(bound.dimension))
        .format(format)
        .subresource_range(sampled_texture_subresource_range(bound));
    unsafe { device.create_image_view(&view_info, None) }
        .map_err(|_| texture_error("sampled texture view creation failed"))
}

fn create_storage_texture_image_view(
    device: &ash::Device,
    texture: &VulkanTexture,
    bound: &HalBoundTexture,
) -> Result<DescriptorImageView, HalError> {
    let inner = texture.inner()?;
    let view_type = sampled_texture_view_type(bound.dimension);
    let subresource_range = sampled_texture_subresource_range(bound);
    if storage_texture_uses_cached_bgra8_view(texture, inner, bound, view_type, subresource_range) {
        return Ok(DescriptorImageView {
            view: inner.bgra8_storage_view,
            owned: false,
        });
    }

    let view_format = storage_texture_view_format(bound.format);
    let (format, _) = map_texture_format(view_format)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(inner.image)
        .view_type(view_type)
        .format(format)
        .subresource_range(subresource_range);
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|_| texture_error("storage texture view creation failed"))?;
    Ok(DescriptorImageView { view, owned: true })
}

fn storage_texture_uses_cached_bgra8_view(
    texture: &VulkanTexture,
    inner: &VulkanTextureInner,
    bound: &HalBoundTexture,
    view_type: vk::ImageViewType,
    subresource_range: vk::ImageSubresourceRange,
) -> bool {
    storage_texture_can_use_cached_bgra8_view(
        bound.format,
        inner.bgra8_storage_view,
        view_type,
        default_texture_image_view_type(texture),
        subresource_range,
        color_subresource_range(inner.mip_level_count, inner.array_layers),
    )
}

fn storage_texture_can_use_cached_bgra8_view(
    bound_format: HalTextureFormat,
    cached_view: vk::ImageView,
    view_type: vk::ImageViewType,
    default_view_type: vk::ImageViewType,
    subresource_range: vk::ImageSubresourceRange,
    canonical_range: vk::ImageSubresourceRange,
) -> bool {
    bound_format == HalTextureFormat::Bgra8Unorm
        && cached_view != vk::ImageView::null()
        && view_type == default_view_type
        && image_subresource_ranges_equal(subresource_range, canonical_range)
}

fn storage_texture_view_format(format: HalTextureFormat) -> HalTextureFormat {
    if format == HalTextureFormat::Bgra8Unorm {
        HalTextureFormat::Rgba8Unorm
    } else {
        format
    }
}

fn default_texture_image_view_type(texture: &VulkanTexture) -> vk::ImageViewType {
    match texture.dimension {
        HalTextureDimension::D1 => vk::ImageViewType::TYPE_1D,
        HalTextureDimension::D2 if texture.depth_or_array_layers > 1 => {
            vk::ImageViewType::TYPE_2D_ARRAY
        }
        HalTextureDimension::D2 => vk::ImageViewType::TYPE_2D,
        HalTextureDimension::D3 => vk::ImageViewType::TYPE_3D,
    }
}

fn image_subresource_ranges_equal(
    lhs: vk::ImageSubresourceRange,
    rhs: vk::ImageSubresourceRange,
) -> bool {
    lhs.aspect_mask == rhs.aspect_mask
        && lhs.base_mip_level == rhs.base_mip_level
        && lhs.level_count == rhs.level_count
        && lhs.base_array_layer == rhs.base_array_layer
        && lhs.layer_count == rhs.layer_count
}

/// Selects the `HalTextureFormat` whose `VkFormat` a sampled image view must
/// use. For a combined depth-stencil texture (both depth and stencil aspects),
/// an aspect view must keep the texture's combined VkFormat — the aspect is
/// chosen via the subresource aspect mask, not by reinterpreting to an
/// aspect-only VkFormat (which Vulkan rejects as format-incompatible). For all
/// other textures the view's own format is used.
fn sampled_texture_view_format(
    texture_format: HalTextureFormat,
    view_format: HalTextureFormat,
) -> HalTextureFormat {
    if format_has_depth_aspect(texture_format) && format_has_stencil_aspect(texture_format) {
        texture_format
    } else {
        view_format
    }
}

fn sampled_texture_view_type(dimension: HalTextureViewDimension) -> vk::ImageViewType {
    match dimension {
        HalTextureViewDimension::D1 => vk::ImageViewType::TYPE_1D,
        HalTextureViewDimension::D2 => vk::ImageViewType::TYPE_2D,
        HalTextureViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
        HalTextureViewDimension::Cube => vk::ImageViewType::CUBE,
        HalTextureViewDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        HalTextureViewDimension::D3 => vk::ImageViewType::TYPE_3D,
    }
}

fn sampled_texture_subresource_range(bound: &HalBoundTexture) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(sampled_texture_aspect_flags(bound.format, bound.aspect))
        .base_mip_level(bound.base_mip_level)
        .level_count(bound.mip_level_count)
        .base_array_layer(bound.base_array_layer)
        .layer_count(bound.array_layer_count)
}

fn sampled_texture_aspect_flags(
    format: HalTextureFormat,
    aspect: HalTextureAspect,
) -> vk::ImageAspectFlags {
    match aspect {
        HalTextureAspect::All => {
            let mut flags = vk::ImageAspectFlags::empty();
            if format_has_depth_aspect(format) {
                flags |= vk::ImageAspectFlags::DEPTH;
            }
            if format_has_stencil_aspect(format) {
                flags |= vk::ImageAspectFlags::STENCIL;
            }
            if flags.is_empty() {
                vk::ImageAspectFlags::COLOR
            } else {
                flags
            }
        }
        HalTextureAspect::DepthOnly => vk::ImageAspectFlags::DEPTH,
        HalTextureAspect::StencilOnly => vk::ImageAspectFlags::STENCIL,
    }
}

pub(super) fn destroy_descriptor_image_views(device: &ash::Device, views: &[vk::ImageView]) {
    unsafe {
        for &view in views {
            device.destroy_image_view(view, None);
        }
    }
}

fn descriptor_missing_error(pass_name: &'static str, resource: &'static str) -> HalError {
    match (pass_name, resource) {
        ("compute", "buffer") => shader_error("compute buffer descriptor binding is missing"),
        ("compute", "texture") => shader_error("compute texture descriptor binding is missing"),
        ("compute", "sampler") => shader_error("compute sampler descriptor binding is missing"),
        ("render", "buffer") => shader_error("render buffer descriptor binding is missing"),
        ("render", "texture") => shader_error("render texture descriptor binding is missing"),
        ("render", "sampler") => shader_error("render sampler descriptor binding is missing"),
        _ => shader_error("descriptor binding is missing"),
    }
}

/// Returns bind render descriptor sets.
pub(super) fn bind_render_descriptor_sets(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: &VulkanRenderPipeline,
    descriptor_sets: &[vk::DescriptorSet],
) {
    if descriptor_sets.is_empty() {
        return;
    }
    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.inner.pipeline_layout,
            0,
            descriptor_sets,
            &[],
        );
    }
}

/// Returns bind vertex buffers.
pub(super) fn bind_vertex_buffers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    for bound in &pass.vertex_buffers {
        let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
            return Err(buffer_error("vertex buffer is not Vulkan-backed"));
        };
        let inner = buffer.inner()?;
        validate_bound_buffer_range(bound)?;
        let buffers = [inner.buffer];
        let offsets = [bound.offset];
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, bound.binding, &buffers, &offsets);
        }
    }
    Ok(())
}

/// Validates bound buffer range and returns a descriptive error on failure.
pub(super) fn validate_bound_buffer_range(bound: &HalBoundBuffer) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    bound_buffer_range(bound, buffer.size()).map(|_| ())
}

/// Returns bound buffer range.
pub(super) fn bound_buffer_range(
    bound: &HalBoundBuffer,
    buffer_size: u64,
) -> Result<u64, HalError> {
    if bound.offset > buffer_size {
        return Err(buffer_error("buffer offset exceeds buffer size"));
    }
    let range = if bound.size == u64::MAX {
        buffer_size
            .checked_sub(bound.offset)
            .ok_or_else(|| buffer_error("buffer range exceeds buffer size"))?
    } else {
        bound.size
    };
    let end = bound
        .offset
        .checked_add(range)
        .ok_or_else(|| buffer_error("buffer range overflows"))?;
    if end > buffer_size {
        return Err(buffer_error("buffer range exceeds buffer size"));
    }
    Ok(range)
}

/// Creates descriptor pool and reports validation errors through the owning device.
pub(super) fn create_descriptor_pool(
    device: &ash::Device,
    descriptor_set_count: usize,
    bindings: &[HalDescriptorBinding],
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if descriptor_set_count == 0 {
        return Ok(None);
    }
    let uniform_count = bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform)
            )
        })
        .count();
    let storage_count = bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage)
            )
        })
        .count();
    let texture_count = bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalDescriptorBindingKind::Texture))
        .count();
    let storage_texture_count = bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::StorageTexture { .. }
            )
        })
        .count();
    let sampler_count = bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalDescriptorBindingKind::Sampler))
        .count();
    let input_attachment_count = bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                HalDescriptorBindingKind::InputAttachment { .. }
            )
        })
        .count();
    let mut pool_sizes = Vec::new();
    if uniform_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(
                    u32::try_from(uniform_count)
                        .map_err(|_| shader_error("uniform descriptor count is too large"))?,
                ),
        );
    }
    if storage_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(
                    u32::try_from(storage_count)
                        .map_err(|_| shader_error("storage descriptor count is too large"))?,
                ),
        );
    }
    if texture_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(
                    u32::try_from(texture_count)
                        .map_err(|_| shader_error("texture descriptor count is too large"))?,
                ),
        );
    }
    if storage_texture_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(
                    u32::try_from(storage_texture_count).map_err(|_| {
                        shader_error("storage texture descriptor count is too large")
                    })?,
                ),
        );
    }
    if sampler_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(
                    u32::try_from(sampler_count)
                        .map_err(|_| shader_error("sampler descriptor count is too large"))?,
                ),
        );
    }
    if input_attachment_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(
                    u32::try_from(input_attachment_count).map_err(|_| {
                        shader_error("input attachment descriptor count is too large")
                    })?,
                ),
        );
    }
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(
            u32::try_from(descriptor_set_count)
                .map_err(|_| shader_error("descriptor set count is too large"))?,
        )
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|_| shader_error("descriptor pool creation failed"))?;
    Ok(Some(pool))
}

/// Returns descriptor buffer info.
pub(super) fn descriptor_buffer_info(
    bound: &HalBoundBuffer,
) -> Result<vk::DescriptorBufferInfo, HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let inner = buffer.inner()?;
    let range = bound_buffer_range(bound, buffer.size())?;
    Ok(vk::DescriptorBufferInfo::default()
        .buffer(inner.buffer)
        .offset(bound.offset)
        .range(range))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{noop, HalTextureDescriptor, HalTextureDimension, HalTextureUsage};
    use ash::vk::Handle;

    fn dummy_bound_texture(format: HalTextureFormat, aspect: HalTextureAspect) -> HalBoundTexture {
        let device = noop::NoopDevice::new();
        HalBoundTexture {
            group: 0,
            binding: 1,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            texture: HalTexture::Noop(
                device
                    .create_texture(&HalTextureDescriptor {
                        dimension: HalTextureDimension::D2,
                        format,
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 3,
                        mip_level_count: 5,
                        sample_count: 1,
                        usage: HalTextureUsage {
                            copy_src: false,
                            copy_dst: false,
                            texture_binding: true,
                            storage_binding: false,
                            render_attachment: false,
                        },
                    })
                    .expect("Noop texture allocation should succeed"),
            ),
            format,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 2,
            mip_level_count: 3,
            base_array_layer: 1,
            array_layer_count: 1,
            aspect,
            storage_access: None,
        }
    }

    #[test]
    fn primitive_restart_is_enabled_only_for_strip_topologies() {
        assert!(!is_strip_topology(HalPrimitiveTopology::PointList));
        assert!(!is_strip_topology(HalPrimitiveTopology::LineList));
        assert!(is_strip_topology(HalPrimitiveTopology::LineStrip));
        assert!(!is_strip_topology(HalPrimitiveTopology::TriangleList));
        assert!(is_strip_topology(HalPrimitiveTopology::TriangleStrip));
    }

    #[test]
    fn depth_clamp_and_clip_uses_ext_when_depth_clip_control_is_available() {
        assert_eq!(depth_clamp_and_clip(true, false), (true, Some(true)));
        assert_eq!(depth_clamp_and_clip(true, true), (true, Some(false)));
        assert_eq!(depth_clamp_and_clip(false, false), (false, None));
        assert_eq!(depth_clamp_and_clip(false, true), (true, None));
    }

    #[test]
    fn sampled_texture_view_uses_bound_subresource_range() {
        let bound = dummy_bound_texture(HalTextureFormat::Depth32Float, HalTextureAspect::All);

        let range = sampled_texture_subresource_range(&bound);

        assert_eq!(
            sampled_texture_view_type(bound.dimension),
            vk::ImageViewType::TYPE_2D
        );
        assert_eq!(range.aspect_mask, vk::ImageAspectFlags::DEPTH);
        assert_eq!(range.base_mip_level, 2);
        assert_eq!(range.level_count, 3);
        assert_eq!(range.base_array_layer, 1);
        assert_eq!(range.layer_count, 1);
    }

    #[test]
    fn sampled_texture_aspect_flags_respect_explicit_stencil_view() {
        let bound = dummy_bound_texture(
            HalTextureFormat::Depth32FloatStencil8,
            HalTextureAspect::StencilOnly,
        );

        let range = sampled_texture_subresource_range(&bound);

        assert_eq!(range.aspect_mask, vk::ImageAspectFlags::STENCIL);
    }

    #[test]
    fn sampled_view_format_keeps_combined_format_for_aspect_views() {
        // After the core fix, a DepthOnly view of a combined texture arrives with
        // the aspect-specific view format (e.g. Depth32Float / Depth24Plus); a
        // StencilOnly view arrives with Stencil8. The image view must still use
        // the texture's combined VkFormat (the aspect is selected by the aspect
        // mask), so the view format here resolves to the combined texture format.
        for combined in [
            HalTextureFormat::Depth24PlusStencil8,
            HalTextureFormat::Depth32FloatStencil8,
        ] {
            let depth_view = match combined {
                HalTextureFormat::Depth24PlusStencil8 => HalTextureFormat::Depth24Plus,
                HalTextureFormat::Depth32FloatStencil8 => HalTextureFormat::Depth32Float,
                _ => unreachable!(),
            };
            assert_eq!(
                sampled_texture_view_format(combined, depth_view),
                combined,
                "depth-aspect view of {combined:?} must keep the combined format"
            );
            assert_eq!(
                sampled_texture_view_format(combined, HalTextureFormat::Stencil8),
                combined,
                "stencil-aspect view of {combined:?} must keep the combined format"
            );
            assert_eq!(sampled_texture_view_format(combined, combined), combined);
        }
    }

    #[test]
    fn sampled_view_format_keeps_view_format_for_non_combined_textures() {
        // Non-combined textures (color, depth-only, stencil-only) keep the view's
        // own format unchanged.
        assert_eq!(
            sampled_texture_view_format(HalTextureFormat::Rgba8Unorm, HalTextureFormat::Rgba8Unorm),
            HalTextureFormat::Rgba8Unorm
        );
        assert_eq!(
            sampled_texture_view_format(
                HalTextureFormat::Depth32Float,
                HalTextureFormat::Depth32Float
            ),
            HalTextureFormat::Depth32Float
        );
        assert_eq!(
            sampled_texture_view_format(HalTextureFormat::Stencil8, HalTextureFormat::Stencil8),
            HalTextureFormat::Stencil8
        );
    }

    #[test]
    fn storage_view_format_remaps_bgra8unorm_to_rgba8unorm() {
        assert_eq!(
            storage_texture_view_format(HalTextureFormat::Bgra8Unorm),
            HalTextureFormat::Rgba8Unorm
        );
        assert_eq!(
            storage_texture_view_format(HalTextureFormat::Rgba8Unorm),
            HalTextureFormat::Rgba8Unorm
        );
        assert_eq!(
            storage_texture_view_format(HalTextureFormat::Bgra8UnormSrgb),
            HalTextureFormat::Bgra8UnormSrgb
        );
    }

    #[test]
    fn cached_bgra8_storage_view_requires_canonical_range_and_view_type() {
        let cached = vk::ImageView::from_raw(1);
        let canonical = color_subresource_range(2, 3);

        assert!(storage_texture_can_use_cached_bgra8_view(
            HalTextureFormat::Bgra8Unorm,
            cached,
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageViewType::TYPE_2D_ARRAY,
            canonical,
            canonical,
        ));
        assert!(!storage_texture_can_use_cached_bgra8_view(
            HalTextureFormat::Rgba8Unorm,
            cached,
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageViewType::TYPE_2D_ARRAY,
            canonical,
            canonical,
        ));
        assert!(!storage_texture_can_use_cached_bgra8_view(
            HalTextureFormat::Bgra8Unorm,
            vk::ImageView::null(),
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageViewType::TYPE_2D_ARRAY,
            canonical,
            canonical,
        ));
        assert!(!storage_texture_can_use_cached_bgra8_view(
            HalTextureFormat::Bgra8Unorm,
            cached,
            vk::ImageViewType::CUBE,
            vk::ImageViewType::TYPE_2D_ARRAY,
            canonical,
            canonical,
        ));

        let mut subrange = canonical;
        subrange.base_mip_level = 1;
        assert!(!storage_texture_can_use_cached_bgra8_view(
            HalTextureFormat::Bgra8Unorm,
            cached,
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageViewType::TYPE_2D_ARRAY,
            subrange,
            canonical,
        ));
    }
}
