#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_bind_groups: u32,
    pub max_bind_groups_plus_vertex_buffers: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_vertex_buffers: u32,
    pub max_buffer_size: u64,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffer_array_stride: u32,
    pub max_inter_stage_shader_variables: u32,
    pub max_color_attachments: u32,
    pub max_color_attachment_bytes_per_sample: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_immediate_size: u32,
}

impl Limits {
    pub const DEFAULT: Self = Self {
        max_texture_dimension_1d: 4096,
        max_texture_dimension_2d: 4096,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 256,
        max_bind_groups: 4,
        max_bind_groups_plus_vertex_buffers: 24,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        max_dynamic_storage_buffers_per_pipeline_layout: 4,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 16_384,
        max_storage_buffer_binding_size: 128 * 1024 * 1024,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_vertex_buffers: 8,
        max_buffer_size: 256 * 1024 * 1024,
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_inter_stage_shader_variables: 15,
        max_color_attachments: 4,
        max_color_attachment_bytes_per_sample: 32,
        max_compute_workgroup_storage_size: 16_384,
        max_compute_invocations_per_workgroup: 128,
        max_compute_workgroup_size_x: 128,
        max_compute_workgroup_size_y: 128,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65_535,
        max_immediate_size: 64,
    };

    pub(crate) fn validate_required_limits(self, required: Option<&Self>) -> Result<Self, String> {
        // Block 00: for the synthetic Noop adapter, supported limits equal
        // the WebGPU spec defaults, so comparisons against `self` collapse to
        // comparisons against `DEFAULT` intentionally.
        let required = required.copied().unwrap_or(Self::DEFAULT);
        let default = Self::DEFAULT;
        let mut effective = default;

        macro_rules! maximum {
            ($field:ident) => {
                if required.$field > self.$field {
                    return Err(format!(
                        "required limit {}={} exceeds supported {}",
                        stringify!($field),
                        required.$field,
                        self.$field
                    ));
                }
                effective.$field = required.$field.max(default.$field);
            };
        }

        macro_rules! alignment {
            ($field:ident) => {
                if required.$field < self.$field {
                    return Err(format!(
                        "required limit {}={} is below supported {}",
                        stringify!($field),
                        required.$field,
                        self.$field
                    ));
                }
                effective.$field = required.$field.min(default.$field);
            };
        }

        maximum!(max_texture_dimension_1d);
        maximum!(max_texture_dimension_2d);
        maximum!(max_texture_dimension_3d);
        maximum!(max_texture_array_layers);
        maximum!(max_bind_groups);
        maximum!(max_bind_groups_plus_vertex_buffers);
        maximum!(max_bindings_per_bind_group);
        maximum!(max_dynamic_uniform_buffers_per_pipeline_layout);
        maximum!(max_dynamic_storage_buffers_per_pipeline_layout);
        maximum!(max_sampled_textures_per_shader_stage);
        maximum!(max_samplers_per_shader_stage);
        maximum!(max_storage_buffers_per_shader_stage);
        maximum!(max_storage_textures_per_shader_stage);
        maximum!(max_uniform_buffers_per_shader_stage);
        maximum!(max_uniform_buffer_binding_size);
        maximum!(max_storage_buffer_binding_size);
        alignment!(min_uniform_buffer_offset_alignment);
        alignment!(min_storage_buffer_offset_alignment);
        maximum!(max_vertex_buffers);
        maximum!(max_buffer_size);
        maximum!(max_vertex_attributes);
        maximum!(max_vertex_buffer_array_stride);
        maximum!(max_inter_stage_shader_variables);
        maximum!(max_color_attachments);
        maximum!(max_color_attachment_bytes_per_sample);
        maximum!(max_compute_workgroup_storage_size);
        maximum!(max_compute_invocations_per_workgroup);
        maximum!(max_compute_workgroup_size_x);
        maximum!(max_compute_workgroup_size_y);
        maximum!(max_compute_workgroup_size_z);
        maximum!(max_compute_workgroups_per_dimension);

        if required.max_immediate_size > self.max_immediate_size {
            return Err(format!(
                "required limit max_immediate_size={} exceeds supported {}",
                required.max_immediate_size, self.max_immediate_size
            ));
        }
        effective.max_immediate_size = self.max_immediate_size;

        Ok(effective)
    }
}
