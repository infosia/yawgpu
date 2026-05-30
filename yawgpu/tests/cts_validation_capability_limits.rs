#[macro_use]
#[path = "cts/validation/capability_checks/limits/common.rs"]
mod common;

#[path = "cts/validation/capability_checks/limits/max_bind_groups.rs"]
mod max_bind_groups;
#[path = "cts/validation/capability_checks/limits/max_bind_groups_plus_vertex_buffers.rs"]
mod max_bind_groups_plus_vertex_buffers;
#[path = "cts/validation/capability_checks/limits/max_bindings_per_bind_group.rs"]
mod max_bindings_per_bind_group;
#[path = "cts/validation/capability_checks/limits/max_buffer_size.rs"]
mod max_buffer_size;
#[path = "cts/validation/capability_checks/limits/max_color_attachment_bytes_per_sample.rs"]
mod max_color_attachment_bytes_per_sample;
#[path = "cts/validation/capability_checks/limits/max_color_attachments.rs"]
mod max_color_attachments;
#[path = "cts/validation/capability_checks/limits/max_compute_invocations_per_workgroup.rs"]
mod max_compute_invocations_per_workgroup;
#[path = "cts/validation/capability_checks/limits/max_compute_workgroup_size_x.rs"]
mod max_compute_workgroup_size_x;
#[path = "cts/validation/capability_checks/limits/max_compute_workgroup_size_y.rs"]
mod max_compute_workgroup_size_y;
#[path = "cts/validation/capability_checks/limits/max_compute_workgroup_size_z.rs"]
mod max_compute_workgroup_size_z;
#[path = "cts/validation/capability_checks/limits/max_compute_workgroup_storage_size.rs"]
mod max_compute_workgroup_storage_size;
#[path = "cts/validation/capability_checks/limits/max_compute_workgroups_per_dimension.rs"]
mod max_compute_workgroups_per_dimension;
#[path = "cts/validation/capability_checks/limits/max_dynamic_storage_buffers_per_pipeline_layout.rs"]
mod max_dynamic_storage_buffers_per_pipeline_layout;
#[path = "cts/validation/capability_checks/limits/max_dynamic_uniform_buffers_per_pipeline_layout.rs"]
mod max_dynamic_uniform_buffers_per_pipeline_layout;
#[path = "cts/validation/capability_checks/limits/max_inter_stage_shader_variables.rs"]
mod max_inter_stage_shader_variables;
#[path = "cts/validation/capability_checks/limits/max_sampled_textures_per_shader_stage.rs"]
mod max_sampled_textures_per_shader_stage;
#[path = "cts/validation/capability_checks/limits/max_samplers_per_shader_stage.rs"]
mod max_samplers_per_shader_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_buffer_binding_size.rs"]
mod max_storage_buffer_binding_size;
#[path = "cts/validation/capability_checks/limits/max_storage_buffers_in_fragment_stage.rs"]
mod max_storage_buffers_in_fragment_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_buffers_in_vertex_stage.rs"]
mod max_storage_buffers_in_vertex_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_buffers_per_shader_stage.rs"]
mod max_storage_buffers_per_shader_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_textures_in_fragment_stage.rs"]
mod max_storage_textures_in_fragment_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_textures_in_vertex_stage.rs"]
mod max_storage_textures_in_vertex_stage;
#[path = "cts/validation/capability_checks/limits/max_storage_textures_per_shader_stage.rs"]
mod max_storage_textures_per_shader_stage;
#[path = "cts/validation/capability_checks/limits/max_texture_array_layers.rs"]
mod max_texture_array_layers;
#[path = "cts/validation/capability_checks/limits/max_texture_dimension_1d.rs"]
mod max_texture_dimension_1d;
#[path = "cts/validation/capability_checks/limits/max_texture_dimension_2d.rs"]
mod max_texture_dimension_2d;
#[path = "cts/validation/capability_checks/limits/max_texture_dimension_3d.rs"]
mod max_texture_dimension_3d;
#[path = "cts/validation/capability_checks/limits/max_uniform_buffer_binding_size.rs"]
mod max_uniform_buffer_binding_size;
#[path = "cts/validation/capability_checks/limits/max_uniform_buffers_per_shader_stage.rs"]
mod max_uniform_buffers_per_shader_stage;
#[path = "cts/validation/capability_checks/limits/max_vertex_attributes.rs"]
mod max_vertex_attributes;
#[path = "cts/validation/capability_checks/limits/max_vertex_buffer_array_stride.rs"]
mod max_vertex_buffer_array_stride;
#[path = "cts/validation/capability_checks/limits/max_vertex_buffers.rs"]
mod max_vertex_buffers;
#[path = "cts/validation/capability_checks/limits/min_storage_buffer_offset_alignment.rs"]
mod min_storage_buffer_offset_alignment;
#[path = "cts/validation/capability_checks/limits/min_uniform_buffer_offset_alignment.rs"]
mod min_uniform_buffer_offset_alignment;
