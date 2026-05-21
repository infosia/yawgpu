use super::*;

/// Converts limits to native into the corresponding yawgpu representation.
#[must_use]
pub fn map_limits_to_native(limits: core::Limits) -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: limits.max_texture_dimension_1d,
        maxTextureDimension2D: limits.max_texture_dimension_2d,
        maxTextureDimension3D: limits.max_texture_dimension_3d,
        maxTextureArrayLayers: limits.max_texture_array_layers,
        maxBindGroups: limits.max_bind_groups,
        maxBindGroupsPlusVertexBuffers: limits.max_bind_groups_plus_vertex_buffers,
        maxBindingsPerBindGroup: limits.max_bindings_per_bind_group,
        maxDynamicUniformBuffersPerPipelineLayout: limits
            .max_dynamic_uniform_buffers_per_pipeline_layout,
        maxDynamicStorageBuffersPerPipelineLayout: limits
            .max_dynamic_storage_buffers_per_pipeline_layout,
        maxSampledTexturesPerShaderStage: limits.max_sampled_textures_per_shader_stage,
        maxSamplersPerShaderStage: limits.max_samplers_per_shader_stage,
        maxStorageBuffersPerShaderStage: limits.max_storage_buffers_per_shader_stage,
        maxStorageTexturesPerShaderStage: limits.max_storage_textures_per_shader_stage,
        maxUniformBuffersPerShaderStage: limits.max_uniform_buffers_per_shader_stage,
        maxUniformBufferBindingSize: limits.max_uniform_buffer_binding_size,
        maxStorageBufferBindingSize: limits.max_storage_buffer_binding_size,
        minUniformBufferOffsetAlignment: limits.min_uniform_buffer_offset_alignment,
        minStorageBufferOffsetAlignment: limits.min_storage_buffer_offset_alignment,
        maxVertexBuffers: limits.max_vertex_buffers,
        maxBufferSize: limits.max_buffer_size,
        maxVertexAttributes: limits.max_vertex_attributes,
        maxVertexBufferArrayStride: limits.max_vertex_buffer_array_stride,
        maxInterStageShaderVariables: limits.max_inter_stage_shader_variables,
        maxColorAttachments: limits.max_color_attachments,
        maxColorAttachmentBytesPerSample: limits.max_color_attachment_bytes_per_sample,
        maxComputeWorkgroupStorageSize: limits.max_compute_workgroup_storage_size,
        maxComputeInvocationsPerWorkgroup: limits.max_compute_invocations_per_workgroup,
        maxComputeWorkgroupSizeX: limits.max_compute_workgroup_size_x,
        maxComputeWorkgroupSizeY: limits.max_compute_workgroup_size_y,
        maxComputeWorkgroupSizeZ: limits.max_compute_workgroup_size_z,
        maxComputeWorkgroupsPerDimension: limits.max_compute_workgroups_per_dimension,
        maxImmediateSize: limits.max_immediate_size,
    }
}

/// Converts limits into the corresponding yawgpu representation.
#[must_use]
pub fn map_limits(value: &native::WGPULimits) -> core::Limits {
    let default = core::Limits::DEFAULT;
    let mut limits = default;
    limits.max_texture_dimension_1d = limit_u32(
        value.maxTextureDimension1D,
        default.max_texture_dimension_1d,
    );
    limits.max_texture_dimension_2d = limit_u32(
        value.maxTextureDimension2D,
        default.max_texture_dimension_2d,
    );
    limits.max_texture_dimension_3d = limit_u32(
        value.maxTextureDimension3D,
        default.max_texture_dimension_3d,
    );
    limits.max_texture_array_layers = limit_u32(
        value.maxTextureArrayLayers,
        default.max_texture_array_layers,
    );
    limits.max_bind_groups = limit_u32(value.maxBindGroups, default.max_bind_groups);
    limits.max_bind_groups_plus_vertex_buffers = limit_u32(
        value.maxBindGroupsPlusVertexBuffers,
        default.max_bind_groups_plus_vertex_buffers,
    );
    limits.max_bindings_per_bind_group = limit_u32(
        value.maxBindingsPerBindGroup,
        default.max_bindings_per_bind_group,
    );
    limits.max_dynamic_uniform_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicUniformBuffersPerPipelineLayout,
        default.max_dynamic_uniform_buffers_per_pipeline_layout,
    );
    limits.max_dynamic_storage_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicStorageBuffersPerPipelineLayout,
        default.max_dynamic_storage_buffers_per_pipeline_layout,
    );
    limits.max_sampled_textures_per_shader_stage = limit_u32(
        value.maxSampledTexturesPerShaderStage,
        default.max_sampled_textures_per_shader_stage,
    );
    limits.max_samplers_per_shader_stage = limit_u32(
        value.maxSamplersPerShaderStage,
        default.max_samplers_per_shader_stage,
    );
    limits.max_storage_buffers_per_shader_stage = limit_u32(
        value.maxStorageBuffersPerShaderStage,
        default.max_storage_buffers_per_shader_stage,
    );
    limits.max_storage_textures_per_shader_stage = limit_u32(
        value.maxStorageTexturesPerShaderStage,
        default.max_storage_textures_per_shader_stage,
    );
    limits.max_uniform_buffers_per_shader_stage = limit_u32(
        value.maxUniformBuffersPerShaderStage,
        default.max_uniform_buffers_per_shader_stage,
    );
    limits.max_uniform_buffer_binding_size = limit_u64(
        value.maxUniformBufferBindingSize,
        default.max_uniform_buffer_binding_size,
    );
    limits.max_storage_buffer_binding_size = limit_u64(
        value.maxStorageBufferBindingSize,
        default.max_storage_buffer_binding_size,
    );
    limits.min_uniform_buffer_offset_alignment = limit_u32(
        value.minUniformBufferOffsetAlignment,
        default.min_uniform_buffer_offset_alignment,
    );
    limits.min_storage_buffer_offset_alignment = limit_u32(
        value.minStorageBufferOffsetAlignment,
        default.min_storage_buffer_offset_alignment,
    );
    limits.max_vertex_buffers = limit_u32(value.maxVertexBuffers, default.max_vertex_buffers);
    limits.max_buffer_size = limit_u64(value.maxBufferSize, default.max_buffer_size);
    limits.max_vertex_attributes =
        limit_u32(value.maxVertexAttributes, default.max_vertex_attributes);
    limits.max_vertex_buffer_array_stride = limit_u32(
        value.maxVertexBufferArrayStride,
        default.max_vertex_buffer_array_stride,
    );
    limits.max_inter_stage_shader_variables = limit_u32(
        value.maxInterStageShaderVariables,
        default.max_inter_stage_shader_variables,
    );
    limits.max_color_attachments =
        limit_u32(value.maxColorAttachments, default.max_color_attachments);
    limits.max_color_attachment_bytes_per_sample = limit_u32(
        value.maxColorAttachmentBytesPerSample,
        default.max_color_attachment_bytes_per_sample,
    );
    limits.max_compute_workgroup_storage_size = limit_u32(
        value.maxComputeWorkgroupStorageSize,
        default.max_compute_workgroup_storage_size,
    );
    limits.max_compute_invocations_per_workgroup = limit_u32(
        value.maxComputeInvocationsPerWorkgroup,
        default.max_compute_invocations_per_workgroup,
    );
    limits.max_compute_workgroup_size_x = limit_u32(
        value.maxComputeWorkgroupSizeX,
        default.max_compute_workgroup_size_x,
    );
    limits.max_compute_workgroup_size_y = limit_u32(
        value.maxComputeWorkgroupSizeY,
        default.max_compute_workgroup_size_y,
    );
    limits.max_compute_workgroup_size_z = limit_u32(
        value.maxComputeWorkgroupSizeZ,
        default.max_compute_workgroup_size_z,
    );
    limits.max_compute_workgroups_per_dimension = limit_u32(
        value.maxComputeWorkgroupsPerDimension,
        default.max_compute_workgroups_per_dimension,
    );
    limits.max_immediate_size = limit_u32(value.maxImmediateSize, default.max_immediate_size);
    limits
}

#[must_use]
fn limit_u32(value: u32, default: u32) -> u32 {
    if value == native::WGPU_LIMIT_U32_UNDEFINED {
        default
    } else {
        value
    }
}

#[must_use]
fn limit_u64(value: u64, default: u64) -> u64 {
    if value == native::WGPU_LIMIT_U64_UNDEFINED {
        default
    } else {
        value
    }
}
