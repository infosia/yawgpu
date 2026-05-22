use super::*;

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
    pub(super) fragment_shader_module: vk::ShaderModule,
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
            self.device
                .device
                .destroy_shader_module(self.fragment_shader_module, None);
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
    fragment_entry_point: &str,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
) -> Result<VulkanRenderPipeline, HalError> {
    let HalShaderSource::SpirVStages { vertex, fragment } = shader else {
        return Err(shader_error(
            "Vulkan render pipeline requires vertex and fragment SPIR-V",
        ));
    };
    let vertex_entry = CString::new(vertex_entry_point)
        .map_err(|_| shader_error("vertex entry point contains NUL"))?;
    let fragment_entry = CString::new(fragment_entry_point)
        .map_err(|_| shader_error("fragment entry point contains NUL"))?;
    let vertex_shader_module = create_shader_module(&device, &vertex)?;
    let fragment_shader_module = match create_shader_module(&device, &fragment) {
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
    let descriptor_set_layouts = match create_descriptor_set_layouts(
        &device,
        bindings,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
    ) {
        Ok(layouts) => layouts,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(shader_error("render pipeline layout creation failed"));
        }
    };
    let render_pass = match create_render_pass(&device, descriptor) {
        Ok(render_pass) => render_pass,
        Err(error) => {
            unsafe {
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
        &fragment_entry,
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                device.device.destroy_render_pass(render_pass, None);
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
#[allow(clippy::too_many_arguments)]
pub(super) fn create_subpass_render_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
    pass_layout: &HalSubpassPassLayout,
    subpass_index: u32,
) -> Result<VulkanRenderPipeline, HalError> {
    let HalShaderSource::SpirVStages { vertex, fragment } = shader else {
        return Err(shader_error(
            "Vulkan subpass render pipeline requires vertex and fragment SPIR-V",
        ));
    };
    let vertex_entry = CString::new(vertex_entry_point)
        .map_err(|_| shader_error("vertex entry point contains NUL"))?;
    let fragment_entry = CString::new(fragment_entry_point)
        .map_err(|_| shader_error("fragment entry point contains NUL"))?;
    let vertex_shader_module = create_shader_module(&device, &vertex)?;
    let fragment_shader_module = match create_shader_module(&device, &fragment) {
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
    let descriptor_set_layouts = match create_descriptor_set_layouts(
        &device,
        bindings,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
    ) {
        Ok(layouts) => layouts,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(shader_error("render pipeline layout creation failed"));
        }
    };
    let render_pass = match cached_subpass_render_pass_for_layout(&device, pass_layout) {
        Ok(render_pass) => render_pass,
        Err(error) => {
            unsafe {
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
        &fragment_entry,
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
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
            render_pass_owned: false,
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
) -> Result<vk::RenderPass, HalError> {
    let color_format = descriptor
        .color_formats
        .first()
        .copied()
        .ok_or_else(|| shader_error("render pipeline requires a color target"))?;
    create_render_pass_for_format(&device.device, color_format)
}

/// Creates render pass for format and reports validation errors through the owning device.
pub(super) fn create_render_pass_for_format(
    device: &ash::Device,
    color_format: HalTextureFormat,
) -> Result<vk::RenderPass, HalError> {
    let (format, _) = map_texture_format(color_format)?;
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let color_reference = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_references = [color_reference];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references);
    let dependency_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let dependency_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let attachments = [attachment];
    let subpasses = [subpass];
    let dependencies = [dependency_in, dependency_out];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("render pass creation failed"))
}

#[cfg(feature = "tiled")]
fn cached_subpass_render_pass_for_layout(
    device: &VulkanDeviceInner,
    layout: &HalSubpassPassLayout,
) -> Result<vk::RenderPass, HalError> {
    if let Ok(cache) = device.subpass_render_pass_cache.lock() {
        if let Some(&render_pass) = cache.get(layout) {
            return Ok(render_pass);
        }
    }
    let render_pass = create_subpass_render_pass_for_layout(&device.device, layout)?;
    match device.subpass_render_pass_cache.lock() {
        Ok(mut cache) => {
            let entry = cache.entry(layout.clone()).or_insert(render_pass);
            if *entry != render_pass {
                unsafe {
                    device.device.destroy_render_pass(render_pass, None);
                }
            }
            Ok(*entry)
        }
        Err(_) => Ok(render_pass),
    }
}

#[cfg(feature = "tiled")]
fn create_subpass_render_pass_for_layout(
    device: &ash::Device,
    layout: &HalSubpassPassLayout,
) -> Result<vk::RenderPass, HalError> {
    let mut attachments = Vec::new();
    for attachment in &layout.color_attachments {
        let (format, _) = map_texture_format(attachment.format)?;
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(attachment.sample_count)?)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        );
    }
    if let Some(attachment) = layout.depth_stencil_attachment {
        let (format, _) = map_texture_format(attachment.format)?;
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(attachment.sample_count)?)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
    }
    let depth_index = layout.color_attachments.len() as u32;
    let color_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass
                .color_attachment_indices
                .iter()
                .map(|&attachment| {
                    vk::AttachmentReference::default()
                        .attachment(attachment)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| {
                    let (attachment, image_layout) = if input.source_attachment == u32::MAX {
                        (
                            depth_index,
                            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        )
                    } else {
                        (
                            input.source_attachment,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )
                    };
                    vk::AttachmentReference::default()
                        .attachment(attachment)
                        .layout(image_layout)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let depth_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass.uses_depth_stencil.then(|| {
                vk::AttachmentReference::default()
                    .attachment(depth_index)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            })
        })
        .collect::<Vec<_>>();
    let mut subpasses = Vec::new();
    for index in 0..layout.subpasses.len() {
        let mut description = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs[index])
            .input_attachments(&input_refs[index]);
        if let Some(depth_ref) = depth_refs[index].as_ref() {
            description = description.depth_stencil_attachment(depth_ref);
        }
        subpasses.push(description);
    }
    let dependencies = subpass_dependencies_for_layout(layout);
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("subpass render pass creation failed"))
}

#[cfg(feature = "tiled")]
fn subpass_dependencies_for_layout(layout: &HalSubpassPassLayout) -> Vec<vk::SubpassDependency> {
    let mut dependencies = vec![vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    dependencies.extend(layout.dependencies.iter().map(|dependency| {
        let (src_stage, src_access, dst_stage, dst_access) = match dependency.dependency_type {
            HalSubpassDependencyType::ColorToInput => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
            HalSubpassDependencyType::DepthToInput => (
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
            HalSubpassDependencyType::ColorDepthToInput => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
        };
        vk::SubpassDependency::default()
            .src_subpass(dependency.src_subpass)
            .dst_subpass(dependency.dst_subpass)
            .src_stage_mask(src_stage)
            .dst_stage_mask(dst_stage)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .dependency_flags(if dependency.by_region {
                vk::DependencyFlags::BY_REGION
            } else {
                vk::DependencyFlags::empty()
            })
    }));
    dependencies.push(
        vk::SubpassDependency::default()
            .src_subpass(layout.subpasses.len().saturating_sub(1) as u32)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    );
    dependencies
}

#[cfg(feature = "tiled")]
fn vk_sample_count(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(shader_error("unsupported subpass sample count")),
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
    fragment_shader_module: vk::ShaderModule,
    vertex_entry: &CStr,
    fragment_entry: &CStr,
) -> Result<vk::Pipeline, HalError> {
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(vertex_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(fragment_entry),
    ];
    let binding_descriptions = descriptor
        .vertex_buffers
        .iter()
        .enumerate()
        .map(|(slot, layout)| {
            let slot =
                u32::try_from(slot).map_err(|_| shader_error("vertex buffer slot is too large"))?;
            Ok(vk::VertexInputBindingDescription::default()
                .binding(slot)
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
    for (slot, layout) in descriptor.vertex_buffers.iter().enumerate() {
        let slot =
            u32::try_from(slot).map_err(|_| shader_error("vertex buffer slot is too large"))?;
        for attribute in &layout.attributes {
            attribute_descriptions.push(
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.shader_location)
                    .binding(slot)
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
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);
    let color_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let color_attachments = [color_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
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
pub(super) fn descriptor_type(kind: HalBufferBindingKind) -> vk::DescriptorType {
    match kind {
        HalBufferBindingKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        HalBufferBindingKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
        #[cfg(feature = "tiled")]
        HalBufferBindingKind::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
    }
}

/// Returns the descriptor set layout stage flags for one binding.
///
/// Input attachments may only be read in the fragment stage
/// (`VUID-VkDescriptorSetLayoutBinding-descriptorType-01510`), so they take
/// `FRAGMENT` regardless of the pipeline-wide default; other bindings use it.
fn binding_stage_flags(
    kind: HalBufferBindingKind,
    default: vk::ShaderStageFlags,
) -> vk::ShaderStageFlags {
    match kind {
        HalBufferBindingKind::Uniform | HalBufferBindingKind::Storage => default,
        #[cfg(feature = "tiled")]
        HalBufferBindingKind::InputAttachment => vk::ShaderStageFlags::FRAGMENT,
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
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Uniform))
        .count();
    let storage_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Storage))
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
) -> Result<(), HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(());
    }
    let mut buffer_infos = Vec::new();
    let mut write_specs = Vec::new();
    for descriptor in &pipeline.inner.descriptor_bindings {
        let bound = pass
            .bind_buffers
            .iter()
            .find(|bound| bound.group == descriptor.group && bound.binding == descriptor.binding)
            .ok_or_else(|| shader_error("compute descriptor binding is missing"))?;
        let buffer_info = descriptor_buffer_info(bound)?;
        buffer_infos.push(buffer_info);
        write_specs.push((
            buffer_infos.len() - 1,
            descriptor.group,
            descriptor.binding,
            descriptor_type(descriptor.kind),
        ));
    }
    let writes = write_specs
        .iter()
        .map(|(info_index, group, binding, descriptor_type)| {
            let group = usize::try_from(*group)
                .map_err(|_| shader_error("descriptor group index is too large"))?;
            let descriptor_set = descriptor_sets
                .get(group)
                .copied()
                .ok_or_else(|| shader_error("descriptor set is missing"))?;
            Ok(vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*binding)
                .descriptor_type(*descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_infos[*info_index])))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    Ok(())
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
    descriptor_sets: &[vk::DescriptorSet],
) -> Result<(), HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(());
    }
    let mut buffer_infos = Vec::new();
    let mut write_specs = Vec::new();
    for descriptor in &pipeline.inner.descriptor_bindings {
        let bound = pass
            .bind_buffers
            .iter()
            .find(|bound| bound.group == descriptor.group && bound.binding == descriptor.binding)
            .ok_or_else(|| shader_error("render descriptor binding is missing"))?;
        let buffer_info = descriptor_buffer_info(bound)?;
        buffer_infos.push(buffer_info);
        write_specs.push((
            buffer_infos.len() - 1,
            descriptor.group,
            descriptor.binding,
            descriptor_type(descriptor.kind),
        ));
    }
    let writes = write_specs
        .iter()
        .map(|(info_index, group, binding, descriptor_type)| {
            let group = usize::try_from(*group)
                .map_err(|_| shader_error("descriptor group index is too large"))?;
            let descriptor_set = descriptor_sets
                .get(group)
                .copied()
                .ok_or_else(|| shader_error("descriptor set is missing"))?;
            Ok(vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*binding)
                .descriptor_type(*descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_infos[*info_index])))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    Ok(())
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
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Uniform))
        .count();
    let storage_count = bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Storage))
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
    #[cfg(feature = "tiled")]
    {
        let input_attachment_count = bindings
            .iter()
            .filter(|binding| matches!(binding.kind, HalBufferBindingKind::InputAttachment))
            .count();
        if input_attachment_count > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                    .descriptor_count(u32::try_from(input_attachment_count).map_err(|_| {
                        shader_error("input attachment descriptor count is too large")
                    })?),
            );
        }
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
