/// Stores limits metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// Max texture dimension 1d.
    pub max_texture_dimension_1d: u32,
    /// Max texture dimension 2d.
    pub max_texture_dimension_2d: u32,
    /// Max texture dimension 3d.
    pub max_texture_dimension_3d: u32,
    /// Max texture array layers.
    pub max_texture_array_layers: u32,
    /// Max bind groups.
    pub max_bind_groups: u32,
    /// Max bind groups plus vertex buffers.
    pub max_bind_groups_plus_vertex_buffers: u32,
    /// Max bindings per bind group.
    pub max_bindings_per_bind_group: u32,
    /// Max dynamic uniform buffers per pipeline layout.
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// Max dynamic storage buffers per pipeline layout.
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// Max sampled textures per shader stage.
    pub max_sampled_textures_per_shader_stage: u32,
    /// Max samplers per shader stage.
    pub max_samplers_per_shader_stage: u32,
    /// Max storage buffers per shader stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Max storage textures per shader stage.
    pub max_storage_textures_per_shader_stage: u32,
    /// Max storage buffers usable in the vertex stage.
    pub max_storage_buffers_in_vertex_stage: u32,
    /// Max storage buffers usable in the fragment stage.
    pub max_storage_buffers_in_fragment_stage: u32,
    /// Max storage textures usable in the vertex stage.
    pub max_storage_textures_in_vertex_stage: u32,
    /// Max storage textures usable in the fragment stage.
    pub max_storage_textures_in_fragment_stage: u32,
    /// Max uniform buffers per shader stage.
    pub max_uniform_buffers_per_shader_stage: u32,
    /// Max uniform buffer binding size.
    pub max_uniform_buffer_binding_size: u64,
    /// Max storage buffer binding size.
    pub max_storage_buffer_binding_size: u64,
    /// Min uniform buffer offset alignment.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Min storage buffer offset alignment.
    pub min_storage_buffer_offset_alignment: u32,
    /// Max vertex buffers.
    pub max_vertex_buffers: u32,
    /// Max buffer size.
    pub max_buffer_size: u64,
    /// Max vertex attributes.
    pub max_vertex_attributes: u32,
    /// Max vertex buffer array stride.
    pub max_vertex_buffer_array_stride: u32,
    /// Max inter stage shader variables.
    pub max_inter_stage_shader_variables: u32,
    /// Max color attachments.
    pub max_color_attachments: u32,
    /// Max color attachment bytes per sample.
    pub max_color_attachment_bytes_per_sample: u32,
    /// Max compute workgroup storage size.
    pub max_compute_workgroup_storage_size: u32,
    /// Max compute invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Max compute workgroup size x.
    pub max_compute_workgroup_size_x: u32,
    /// Max compute workgroup size y.
    pub max_compute_workgroup_size_y: u32,
    /// Max compute workgroup size z.
    pub max_compute_workgroup_size_z: u32,
    /// Max compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Max immediate size.
    pub max_immediate_size: u32,
}

impl Limits {
    /// Constant value for default.
    pub const DEFAULT: Self = Self {
        max_texture_dimension_1d: 8192,
        max_texture_dimension_2d: 8192,
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
        max_storage_buffers_in_vertex_stage: 8,
        max_storage_buffers_in_fragment_stage: 8,
        max_storage_textures_in_vertex_stage: 4,
        max_storage_textures_in_fragment_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65_536,
        max_storage_buffer_binding_size: 128 * 1024 * 1024,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_vertex_buffers: 8,
        max_buffer_size: 256 * 1024 * 1024,
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_inter_stage_shader_variables: 16,
        max_color_attachments: 8,
        max_color_attachment_bytes_per_sample: 32,
        max_compute_workgroup_storage_size: 16_384,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65_535,
        // Block 94 S3: Dawn's base-tier `maxImmediateSize`
        // (`kMaxImmediateDataBytes`, Limits.cpp v1 base tier) -- flipped from
        // 0 once every Tier-1 backend (Noop S1, Metal S2, Vulkan S3)
        // executes SetImmediates. GLES (Tier 2) still reports 0 through its
        // own adapter limits; `from_hal` deliberately copies the HAL value
        // verbatim (no `.max(default)`) so that sticks.
        max_immediate_size: 64,
    };

    /// Converts backend-reported HAL limits into core limits.
    pub(crate) fn from_hal(hal: yawgpu_hal::HalLimits) -> Self {
        let default = Self::DEFAULT;
        let max_storage_buffers_per_shader_stage = hal
            .max_storage_buffers_per_shader_stage
            .max(default.max_storage_buffers_per_shader_stage);
        let max_storage_textures_per_shader_stage = hal
            .max_storage_textures_per_shader_stage
            .max(default.max_storage_textures_per_shader_stage);

        Self {
            max_texture_dimension_1d: hal
                .max_texture_dimension_1d
                .max(default.max_texture_dimension_1d),
            max_texture_dimension_2d: hal
                .max_texture_dimension_2d
                .max(default.max_texture_dimension_2d),
            max_texture_dimension_3d: hal
                .max_texture_dimension_3d
                .max(default.max_texture_dimension_3d),
            max_texture_array_layers: hal
                .max_texture_array_layers
                .max(default.max_texture_array_layers),
            max_bind_groups: hal.max_bind_groups.max(default.max_bind_groups),
            max_bind_groups_plus_vertex_buffers: hal
                .max_bind_groups_plus_vertex_buffers
                .max(default.max_bind_groups_plus_vertex_buffers),
            max_bindings_per_bind_group: hal
                .max_bindings_per_bind_group
                .max(default.max_bindings_per_bind_group),
            max_dynamic_uniform_buffers_per_pipeline_layout: hal
                .max_dynamic_uniform_buffers_per_pipeline_layout
                .max(default.max_dynamic_uniform_buffers_per_pipeline_layout),
            max_dynamic_storage_buffers_per_pipeline_layout: hal
                .max_dynamic_storage_buffers_per_pipeline_layout
                .max(default.max_dynamic_storage_buffers_per_pipeline_layout),
            max_sampled_textures_per_shader_stage: hal
                .max_sampled_textures_per_shader_stage
                .max(default.max_sampled_textures_per_shader_stage),
            max_samplers_per_shader_stage: hal
                .max_samplers_per_shader_stage
                .max(default.max_samplers_per_shader_stage),
            max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage,
            max_storage_buffers_in_vertex_stage: max_storage_buffers_per_shader_stage,
            max_storage_buffers_in_fragment_stage: max_storage_buffers_per_shader_stage,
            max_storage_textures_in_vertex_stage: max_storage_textures_per_shader_stage,
            max_storage_textures_in_fragment_stage: max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage: hal
                .max_uniform_buffers_per_shader_stage
                .max(default.max_uniform_buffers_per_shader_stage),
            max_uniform_buffer_binding_size: hal
                .max_uniform_buffer_binding_size
                .max(default.max_uniform_buffer_binding_size),
            max_storage_buffer_binding_size: hal
                .max_storage_buffer_binding_size
                .max(default.max_storage_buffer_binding_size),
            min_uniform_buffer_offset_alignment: hal
                .min_uniform_buffer_offset_alignment
                .max(32)
                .min(default.min_uniform_buffer_offset_alignment),
            min_storage_buffer_offset_alignment: hal
                .min_storage_buffer_offset_alignment
                .max(32)
                .min(default.min_storage_buffer_offset_alignment),
            max_vertex_buffers: hal.max_vertex_buffers.max(default.max_vertex_buffers),
            max_buffer_size: hal.max_buffer_size.max(default.max_buffer_size),
            max_vertex_attributes: hal.max_vertex_attributes.max(default.max_vertex_attributes),
            max_vertex_buffer_array_stride: hal
                .max_vertex_buffer_array_stride
                .max(default.max_vertex_buffer_array_stride),
            max_inter_stage_shader_variables: hal
                .max_inter_stage_shader_variables
                .max(default.max_inter_stage_shader_variables),
            max_color_attachments: hal.max_color_attachments.max(default.max_color_attachments),
            max_color_attachment_bytes_per_sample: hal
                .max_color_attachment_bytes_per_sample
                .max(default.max_color_attachment_bytes_per_sample),
            max_compute_workgroup_storage_size: hal
                .max_compute_workgroup_storage_size
                .max(default.max_compute_workgroup_storage_size),
            max_compute_invocations_per_workgroup: hal
                .max_compute_invocations_per_workgroup
                .max(default.max_compute_invocations_per_workgroup),
            max_compute_workgroup_size_x: hal
                .max_compute_workgroup_size_x
                .max(default.max_compute_workgroup_size_x),
            max_compute_workgroup_size_y: hal
                .max_compute_workgroup_size_y
                .max(default.max_compute_workgroup_size_y),
            max_compute_workgroup_size_z: hal
                .max_compute_workgroup_size_z
                .max(default.max_compute_workgroup_size_z),
            max_compute_workgroups_per_dimension: hal
                .max_compute_workgroups_per_dimension
                .max(default.max_compute_workgroups_per_dimension),
            max_immediate_size: hal.max_immediate_size,
        }
    }

    /// Validates required limits and returns a descriptive error on failure.
    pub(crate) fn validate_required_limits(self, required: Option<&Self>) -> Result<Self, String> {
        // Whether the caller actually supplied requirements. A null/absent
        // descriptor is NOT a set of asks (Block 67 "decline only
        // unsatisfiable ASKS"): fields whose spec default may exceed a
        // Tier-2 backend's supported value (`max_immediate_size`, GLES
        // supports 0) must skip their over-ask check entirely in that case
        // rather than fail default device creation.
        let required_specified = required.is_some();
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
                // Alignment-class limits must be powers of two, judged on the
                // REQUESTED value (CTS requestDevice `limit,worse_than_default`:
                // requesting default+1 = 257 must fail even though the clamped
                // effective value would be valid). The relationship checks run
                // on the effective limits and cannot see this.
                if !required.$field.is_power_of_two() {
                    return Err(format!(
                        "required limit {}={} must be a power of two",
                        stringify!($field),
                        required.$field
                    ));
                }
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
        maximum!(max_storage_buffers_in_vertex_stage);
        maximum!(max_storage_buffers_in_fragment_stage);
        maximum!(max_storage_textures_in_vertex_stage);
        maximum!(max_storage_textures_in_fragment_stage);
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

        // Dawn EnforceLimitSpecInvariants: yawgpu exposes core feature level,
        // so per-stage storage limits first raise the per-shader-stage limit
        // and then all stage-specific storage limits are pinned to it.
        effective.max_storage_buffers_per_shader_stage = effective
            .max_storage_buffers_per_shader_stage
            .max(effective.max_storage_buffers_in_vertex_stage)
            .max(effective.max_storage_buffers_in_fragment_stage);
        effective.max_storage_textures_per_shader_stage = effective
            .max_storage_textures_per_shader_stage
            .max(effective.max_storage_textures_in_vertex_stage)
            .max(effective.max_storage_textures_in_fragment_stage);
        effective.max_storage_buffers_in_vertex_stage =
            effective.max_storage_buffers_per_shader_stage;
        effective.max_storage_buffers_in_fragment_stage =
            effective.max_storage_buffers_per_shader_stage;
        effective.max_storage_textures_in_vertex_stage =
            effective.max_storage_textures_per_shader_stage;
        effective.max_storage_textures_in_fragment_stage =
            effective.max_storage_textures_per_shader_stage;

        // Relationship checks are evaluated on the *effective* limits (after
        // per-field maximum/alignment clamping to the spec default floor).
        // Evaluating against `required` causes false rejections for single-field
        // worse-than-default requests because the device reports clamped
        // effective limits.
        // The CTS `adapter,requestDevice:limit,worse_than_default` confirms that
        // any single-field worse-than-default Maximum request must succeed.
        validate_required_limit_relationships(effective)?;

        // `max_immediate_size` over-ask check only when the caller actually
        // specified requirements: `Limits::DEFAULT` (64) exceeds a Tier-2
        // GLES adapter's supported 0, so treating an absent descriptor as
        // "requires 64" would deterministically fail default device creation
        // there. The device's effective value is the adapter's supported
        // value either way (never the ask -- CTS
        // `maxImmediateSize always uses supported limit`).
        if required_specified && required.max_immediate_size > self.max_immediate_size {
            return Err(format!(
                "required limit max_immediate_size={} exceeds supported {}",
                required.max_immediate_size, self.max_immediate_size
            ));
        }
        effective.max_immediate_size = self.max_immediate_size;

        Ok(effective)
    }
}

fn validate_required_limit_relationships(limits: Limits) -> Result<(), String> {
    if limits.max_bind_groups > limits.max_bind_groups_plus_vertex_buffers {
        return Err(
            "required max_bind_groups exceeds max_bind_groups_plus_vertex_buffers".to_owned(),
        );
    }
    if limits.max_vertex_buffers > limits.max_bind_groups_plus_vertex_buffers {
        return Err(
            "required max_vertex_buffers exceeds max_bind_groups_plus_vertex_buffers".to_owned(),
        );
    }
    if !valid_min_buffer_offset_alignment(limits.min_uniform_buffer_offset_alignment) {
        return Err(
            "required min_uniform_buffer_offset_alignment must be a power of two and at least 32"
                .to_owned(),
        );
    }
    if !valid_min_buffer_offset_alignment(limits.min_storage_buffer_offset_alignment) {
        return Err(
            "required min_storage_buffer_offset_alignment must be a power of two and at least 32"
                .to_owned(),
        );
    }
    if limits.max_bindings_per_bind_group == 0 {
        return Err("required max_bindings_per_bind_group must be at least one".to_owned());
    }
    if limits.max_compute_workgroups_per_dimension == 0 {
        return Err(
            "required max_compute_workgroups_per_dimension must be at least one".to_owned(),
        );
    }
    if limits.max_color_attachment_bytes_per_sample == 0 {
        return Err(
            "required max_color_attachment_bytes_per_sample must be at least one".to_owned(),
        );
    }
    Ok(())
}

fn valid_min_buffer_offset_alignment(value: u32) -> bool {
    value >= 32 && value.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_required_limits_rejects_relationship_violations() {
        // maxBindGroups cannot exceed maxBindGroupsPlusVertexBuffers.
        let mut supported = Limits::DEFAULT;
        supported.max_bind_groups = Limits::DEFAULT.max_bind_groups_plus_vertex_buffers + 1;
        let mut required = Limits::DEFAULT;
        required.max_bind_groups = supported.max_bind_groups;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err("required max_bind_groups exceeds max_bind_groups_plus_vertex_buffers".to_owned())
        );

        // maxVertexBuffers cannot exceed maxBindGroupsPlusVertexBuffers.
        let mut supported = Limits::DEFAULT;
        supported.max_vertex_buffers = Limits::DEFAULT.max_bind_groups_plus_vertex_buffers + 1;
        let mut required = Limits::DEFAULT;
        required.max_vertex_buffers = supported.max_vertex_buffers;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err(
                "required max_vertex_buffers exceeds max_bind_groups_plus_vertex_buffers"
                    .to_owned()
            )
        );

        // Alignment relationship: a non-power-of-two alignment value is
        // rejected.  The effective alignment = min(required, default) = 48,
        // which is not a power of two — the check still fires after clamping.
        let mut supported = Limits::DEFAULT;
        supported.min_uniform_buffer_offset_alignment = 32;
        let mut required = supported;
        required.min_uniform_buffer_offset_alignment = 48;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err(
                "required limit min_uniform_buffer_offset_alignment=48 must be a power of two"
                    .to_owned()
            )
        );

        // CTS worse_than_default: requesting maxComputeInvocationsPerWorkgroup
        // worse than the default (all other limits at default) must succeed.
        // The effective limits resolve size_x and invocations to their defaults
        // (both 256) so the relationship holds.
        let supported = Limits::DEFAULT;
        let mut required = Limits::DEFAULT;
        required.max_compute_invocations_per_workgroup = 128; // worse-than-default
        assert!(
            supported.validate_required_limits(Some(&required)).is_ok(),
            "worse-than-default maxComputeInvocationsPerWorkgroup must succeed"
        );

        // CTS worse_than_default: requesting maxBindGroupsPlusVertexBuffers=0
        // (worse than default 24) must succeed.
        let mut required = Limits::DEFAULT;
        required.max_bind_groups_plus_vertex_buffers = 0;
        assert!(
            supported.validate_required_limits(Some(&required)).is_ok(),
            "worse-than-default maxBindGroupsPlusVertexBuffers must succeed"
        );

        // CTS worse_than_default: requesting maxColorAttachmentBytesPerSample=0
        // must succeed (effective clamps to default 32).
        let mut required = Limits::DEFAULT;
        required.max_color_attachment_bytes_per_sample = 0;
        assert!(
            supported.validate_required_limits(Some(&required)).is_ok(),
            "worse-than-default maxColorAttachmentBytesPerSample must succeed"
        );
    }

    #[test]
    fn binding_sizes_above_default_max_buffer_size_are_preserved() {
        let requested_size = Limits::DEFAULT.max_buffer_size * 2;

        let mut supported = Limits::DEFAULT;
        supported.max_uniform_buffer_binding_size = requested_size;
        let mut required = Limits::DEFAULT;
        required.max_uniform_buffer_binding_size = requested_size;
        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("uniform buffer binding size above default maxBufferSize should succeed");
        assert_eq!(effective.max_uniform_buffer_binding_size, requested_size);
        assert_eq!(effective.max_buffer_size, Limits::DEFAULT.max_buffer_size);

        let mut supported = Limits::DEFAULT;
        supported.max_storage_buffer_binding_size = requested_size;
        let mut required = Limits::DEFAULT;
        required.max_storage_buffer_binding_size = requested_size;
        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("storage buffer binding size above default maxBufferSize should succeed");
        assert_eq!(effective.max_storage_buffer_binding_size, requested_size);
        assert_eq!(effective.max_buffer_size, Limits::DEFAULT.max_buffer_size);
    }

    #[test]
    fn compute_workgroup_axis_above_default_invocations_is_allowed() {
        let mut supported = Limits::DEFAULT;
        supported.max_compute_workgroup_size_x = 1024;

        let mut required = Limits::DEFAULT;
        required.max_compute_workgroup_size_x = 1024;

        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("requestDevice should not reject per-axis workgroup size above invocations");

        assert_eq!(effective.max_compute_workgroup_size_x, 1024);
        assert_eq!(
            effective.max_compute_invocations_per_workgroup,
            Limits::DEFAULT.max_compute_invocations_per_workgroup
        );
    }

    #[test]
    fn non_power_of_two_alignment_request_is_rejected() {
        // CTS requestDevice `limit,worse_than_default` (alignment "1,1"):
        // default*1 + 1 = 257 is not a power of two and must fail the request
        // even though min(257, default) would be a valid effective value.
        let supported = Limits::DEFAULT;
        let mut required = supported;
        required.min_uniform_buffer_offset_alignment = 257;
        assert!(supported.validate_required_limits(Some(&required)).is_err());
        let mut required2 = supported;
        required2.min_storage_buffer_offset_alignment = 257;
        assert!(supported
            .validate_required_limits(Some(&required2))
            .is_err());
    }

    /// Block 94 S3: `Limits::DEFAULT.max_immediate_size` is Dawn's base-tier
    /// 64 now that every Tier-1 backend executes SetImmediates. Requiring
    /// more than the supported value still fails; a backend reporting 0
    /// (GLES, Tier 2) still rejects any non-zero requirement because
    /// `from_hal` copies the HAL value verbatim.
    #[test]
    fn default_limits_advertise_dawn_base_tier_immediate_size() {
        let supported = Limits::DEFAULT;
        assert_eq!(supported.max_immediate_size, 64);

        let mut required = supported;
        required.max_immediate_size = 68;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err("required limit max_immediate_size=68 exceeds supported 64".to_owned())
        );

        let mut zero_supported = Limits::DEFAULT;
        zero_supported.max_immediate_size = 0;
        let mut required = Limits::DEFAULT;
        required.max_immediate_size = 4;
        assert_eq!(
            zero_supported.validate_required_limits(Some(&required)),
            Err("required limit max_immediate_size=4 exceeds supported 0".to_owned())
        );
    }

    /// Block 94 Phase Review CRITICAL 1 regression: default device creation
    /// (no required limits) must succeed on an adapter that supports
    /// `max_immediate_size == 0` (GLES, Tier 2) even though
    /// `Limits::DEFAULT` is 64 -- an absent descriptor is NOT an ask.
    /// Explicit asks keep their over-ask semantics: 64 against a
    /// zero-supported adapter still errors, and an explicit 0 always passes.
    /// The effective device limit is the adapter's supported value in every
    /// case.
    #[test]
    fn unspecified_max_immediate_size_is_not_a_requirement() {
        let mut zero_supported = Limits::DEFAULT;
        zero_supported.max_immediate_size = 0;

        // Null/absent descriptor: no ask, creation succeeds, effective 0.
        let effective = zero_supported
            .validate_required_limits(None)
            .expect("default device creation must succeed on a zero-supported adapter");
        assert_eq!(effective.max_immediate_size, 0);

        // Explicit ask of the spec default (64) is a real over-ask: rejected.
        let mut required = Limits::DEFAULT;
        required.max_immediate_size = 64;
        assert_eq!(
            zero_supported.validate_required_limits(Some(&required)),
            Err("required limit max_immediate_size=64 exceeds supported 0".to_owned())
        );

        // Explicit 0 (the FFI mapping of WGPU_LIMIT_U32_UNDEFINED) passes.
        let mut required = Limits::DEFAULT;
        required.max_immediate_size = 0;
        let effective = zero_supported
            .validate_required_limits(Some(&required))
            .expect("an explicit max_immediate_size of 0 is always satisfiable");
        assert_eq!(effective.max_immediate_size, 0);
    }

    #[test]
    fn default_limits_match_request_device_cts_core_table() {
        let limits = Limits::DEFAULT;

        assert_eq!(limits.max_texture_dimension_1d, 8192);
        assert_eq!(limits.max_texture_dimension_2d, 8192);
        assert_eq!(limits.max_uniform_buffer_binding_size, 65_536);
        assert_eq!(limits.max_inter_stage_shader_variables, 16);
        assert_eq!(limits.max_color_attachments, 8);
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
        // Block 94 S3: Dawn base tier (kMaxImmediateDataBytes).
        assert_eq!(limits.max_immediate_size, 64);
    }

    #[test]
    fn from_hal_clamps_default_floor_and_alignment_ceiling() {
        let mut hal = yawgpu_hal::HalLimits::DEFAULT;
        hal.max_texture_dimension_2d = Limits::DEFAULT.max_texture_dimension_2d - 1;
        hal.min_uniform_buffer_offset_alignment =
            Limits::DEFAULT.min_uniform_buffer_offset_alignment * 2;

        let limits = Limits::from_hal(hal);

        assert_eq!(
            limits.max_texture_dimension_2d,
            Limits::DEFAULT.max_texture_dimension_2d
        );
        assert_eq!(
            limits.min_uniform_buffer_offset_alignment,
            Limits::DEFAULT.min_uniform_buffer_offset_alignment
        );
    }

    #[test]
    fn from_hal_preserves_better_alignment_and_above_default_maximum() {
        let mut hal = yawgpu_hal::HalLimits::DEFAULT;
        hal.max_texture_dimension_2d = Limits::DEFAULT.max_texture_dimension_2d * 2;
        hal.min_uniform_buffer_offset_alignment = 4;
        hal.min_storage_buffer_offset_alignment = 32;

        let limits = Limits::from_hal(hal);

        assert_eq!(
            limits.max_texture_dimension_2d,
            Limits::DEFAULT.max_texture_dimension_2d * 2
        );
        assert_eq!(limits.min_uniform_buffer_offset_alignment, 32);
        assert_eq!(limits.min_storage_buffer_offset_alignment, 32);
    }

    #[test]
    fn from_hal_pins_stage_specific_storage_limits_to_per_shader_stage() {
        let mut hal = yawgpu_hal::HalLimits::DEFAULT;
        hal.max_storage_textures_per_shader_stage = 4;
        hal.max_storage_textures_in_fragment_stage = 8;

        let limits = Limits::from_hal(hal);

        assert_eq!(limits.max_storage_textures_per_shader_stage, 4);
        assert_eq!(limits.max_storage_textures_in_fragment_stage, 4);
    }

    #[test]
    fn validate_required_limits_rejects_better_than_supported_limits() {
        let supported = Limits::DEFAULT;
        let mut required = Limits::DEFAULT;
        required.max_color_attachments = supported.max_color_attachments + 1;

        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err(format!(
                "required limit max_color_attachments={} exceeds supported {}",
                required.max_color_attachments, supported.max_color_attachments
            ))
        );

        required = Limits::DEFAULT;
        required.min_uniform_buffer_offset_alignment =
            supported.min_uniform_buffer_offset_alignment / 2;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err(format!(
                "required limit min_uniform_buffer_offset_alignment={} is below supported {}",
                required.min_uniform_buffer_offset_alignment,
                supported.min_uniform_buffer_offset_alignment
            ))
        );
    }

    #[test]
    fn validate_required_limits_reports_supported_and_clamps_worse_than_default() {
        let mut supported = Limits::DEFAULT;
        supported.max_color_attachments = 16;
        supported.min_uniform_buffer_offset_alignment = 128;

        let mut required = Limits::DEFAULT;
        required.max_color_attachments = 16;
        required.min_uniform_buffer_offset_alignment = 128;
        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("supported requested limits should validate");
        assert_eq!(effective.max_color_attachments, 16);
        assert_eq!(effective.min_uniform_buffer_offset_alignment, 128);

        required = Limits::DEFAULT;
        required.max_color_attachments = Limits::DEFAULT.max_color_attachments - 1;
        required.min_uniform_buffer_offset_alignment =
            Limits::DEFAULT.min_uniform_buffer_offset_alignment * 2;
        let effective = Limits::DEFAULT
            .validate_required_limits(Some(&required))
            .expect("legal worse-than-default requested limits should validate");
        assert_eq!(
            effective.max_color_attachments,
            Limits::DEFAULT.max_color_attachments
        );
        assert_eq!(
            effective.min_uniform_buffer_offset_alignment,
            Limits::DEFAULT.min_uniform_buffer_offset_alignment
        );
    }

    /// CTS `adapter,requestDevice:limit,worse_than_default` — three previously
    /// failing cases (F-087 residual 4).  Each limit is set to a
    /// worse-than-default value in isolation; all others are at their defaults.
    /// The request must succeed and the device must report the *default* value
    /// (because effective = max(requested, default) = default).
    #[test]
    fn worse_than_default_limits_accepted_and_report_default() {
        let supported = Limits::DEFAULT;

        // maxComputeInvocationsPerWorkgroup: default=256; request 255 (×2 sub-cases).
        for requested in [255u32, 156u32] {
            let mut required = Limits::DEFAULT;
            required.max_compute_invocations_per_workgroup = requested;
            let effective = supported
                .validate_required_limits(Some(&required))
                .unwrap_or_else(|e| {
                    panic!(
                        "worse-than-default maxComputeInvocationsPerWorkgroup={requested} must succeed: {e}"
                    )
                });
            assert_eq!(
                effective.max_compute_invocations_per_workgroup,
                Limits::DEFAULT.max_compute_invocations_per_workgroup,
                "device must report default for maxComputeInvocationsPerWorkgroup"
            );
        }

        // maxBindGroupsPlusVertexBuffers: default=24; request 23 and 0.
        for requested in [23u32, 0u32] {
            let mut required = Limits::DEFAULT;
            required.max_bind_groups_plus_vertex_buffers = requested;
            let effective = supported
                .validate_required_limits(Some(&required))
                .unwrap_or_else(|e| {
                    panic!(
                        "worse-than-default maxBindGroupsPlusVertexBuffers={requested} must succeed: {e}"
                    )
                });
            assert_eq!(
                effective.max_bind_groups_plus_vertex_buffers,
                Limits::DEFAULT.max_bind_groups_plus_vertex_buffers,
                "device must report default for maxBindGroupsPlusVertexBuffers"
            );
        }

        // maxColorAttachmentBytesPerSample: default=32; request 31 and 0.
        for requested in [31u32, 0u32] {
            let mut required = Limits::DEFAULT;
            required.max_color_attachment_bytes_per_sample = requested;
            let effective = supported
                .validate_required_limits(Some(&required))
                .unwrap_or_else(|e| {
                    panic!(
                        "worse-than-default maxColorAttachmentBytesPerSample={requested} must succeed: {e}"
                    )
                });
            assert_eq!(
                effective.max_color_attachment_bytes_per_sample,
                Limits::DEFAULT.max_color_attachment_bytes_per_sample,
                "device must report default for maxColorAttachmentBytesPerSample"
            );
        }
    }

    /// Genuinely invalid relationship requests still reject.
    ///
    /// A non-power-of-two alignment survives effective-limit clamping (alignment
    /// fields use `min`, not `max`) and must still be rejected.  Use a
    /// supported alignment of 32 (the minimum) so that the per-field check
    /// passes (48 >= 32) and only the power-of-two invariant fires.
    #[test]
    fn genuinely_invalid_alignment_relationship_still_rejects() {
        // Set supported alignment to 32 so values above 32 pass the per-field
        // check but non-powers-of-two still trigger the invariant.
        let mut supported = Limits::DEFAULT;
        supported.min_uniform_buffer_offset_alignment = 32;

        // 48 is not a power of two — the alignment relationship check must
        // still fire after effective-limit clamping.
        let mut required = supported; // start from the new supported baseline
        required.min_uniform_buffer_offset_alignment = 48;
        assert_eq!(
            supported.validate_required_limits(Some(&required)),
            Err(
                "required limit min_uniform_buffer_offset_alignment=48 must be a power of two"
                    .to_owned()
            ),
            "non-power-of-two alignment must still be rejected"
        );

        // Same for storage buffer offset.
        let mut supported2 = Limits::DEFAULT;
        supported2.min_storage_buffer_offset_alignment = 32;
        let mut required2 = supported2;
        required2.min_storage_buffer_offset_alignment = 96;
        assert_eq!(
            supported2.validate_required_limits(Some(&required2)),
            Err(
                "required limit min_storage_buffer_offset_alignment=96 must be a power of two"
                    .to_owned()
            ),
            "non-power-of-two storage alignment must still be rejected"
        );
    }

    #[test]
    fn per_stage_limits_default_values_match_cts_table() {
        // CTS kLimitInfos: maxStorageBuffersIn{Vertex,Fragment}Stage default=8,
        // maxStorageTexturesIn{Vertex,Fragment}Stage default=4.
        let limits = Limits::DEFAULT;
        assert_eq!(limits.max_storage_buffers_in_vertex_stage, 8);
        assert_eq!(limits.max_storage_buffers_in_fragment_stage, 8);
        assert_eq!(limits.max_storage_textures_in_vertex_stage, 4);
        assert_eq!(limits.max_storage_textures_in_fragment_stage, 4);
    }

    #[test]
    fn per_stage_limits_better_than_supported_rejected() {
        let supported = Limits::DEFAULT;

        let mut required = Limits::DEFAULT;
        required.max_storage_buffers_in_vertex_stage =
            supported.max_storage_buffers_in_vertex_stage + 1;
        assert!(
            supported.validate_required_limits(Some(&required)).is_err(),
            "max_storage_buffers_in_vertex_stage above supported must fail"
        );

        required = Limits::DEFAULT;
        required.max_storage_buffers_in_fragment_stage =
            supported.max_storage_buffers_in_fragment_stage + 1;
        assert!(
            supported.validate_required_limits(Some(&required)).is_err(),
            "max_storage_buffers_in_fragment_stage above supported must fail"
        );

        required = Limits::DEFAULT;
        required.max_storage_textures_in_vertex_stage =
            supported.max_storage_textures_in_vertex_stage + 1;
        assert!(
            supported.validate_required_limits(Some(&required)).is_err(),
            "max_storage_textures_in_vertex_stage above supported must fail"
        );

        required = Limits::DEFAULT;
        required.max_storage_textures_in_fragment_stage =
            supported.max_storage_textures_in_fragment_stage + 1;
        assert!(
            supported.validate_required_limits(Some(&required)).is_err(),
            "max_storage_textures_in_fragment_stage above supported must fail"
        );
    }

    #[test]
    fn per_stage_limits_requested_value_delivered() {
        // Requesting the supported (== default) value must yield that value exactly.
        let supported = Limits::DEFAULT;
        let required = Limits::DEFAULT; // all at supported value

        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("requesting supported per-stage limits must succeed");
        assert_eq!(
            effective.max_storage_buffers_in_vertex_stage,
            Limits::DEFAULT.max_storage_buffers_in_vertex_stage,
        );
        assert_eq!(
            effective.max_storage_buffers_in_fragment_stage,
            Limits::DEFAULT.max_storage_buffers_in_fragment_stage,
        );
        assert_eq!(
            effective.max_storage_textures_in_vertex_stage,
            Limits::DEFAULT.max_storage_textures_in_vertex_stage,
        );
        assert_eq!(
            effective.max_storage_textures_in_fragment_stage,
            Limits::DEFAULT.max_storage_textures_in_fragment_stage,
        );

        // Requesting worse-than-default is legal and must not be rejected.
        let mut required_worse = Limits::DEFAULT;
        required_worse.max_storage_buffers_in_vertex_stage =
            Limits::DEFAULT.max_storage_buffers_in_vertex_stage - 1;
        required_worse.max_storage_buffers_in_fragment_stage =
            Limits::DEFAULT.max_storage_buffers_in_fragment_stage - 1;
        let effective_worse = Limits::DEFAULT
            .validate_required_limits(Some(&required_worse))
            .expect("worse-than-default per-stage request must succeed");
        // effective clamps to DEFAULT (the minimum); it must not be below default.
        assert_eq!(
            effective_worse.max_storage_buffers_in_vertex_stage,
            Limits::DEFAULT.max_storage_buffers_in_vertex_stage,
        );
    }

    #[test]
    fn storage_textures_in_fragment_stage_auto_upgrades_and_pins_core_limits() {
        let mut supported = Limits::DEFAULT;
        supported.max_storage_textures_per_shader_stage = 8;
        supported.max_storage_textures_in_vertex_stage = 8;
        supported.max_storage_textures_in_fragment_stage = 8;

        let mut required = Limits::DEFAULT;
        required.max_storage_textures_in_fragment_stage = 6;

        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("fragment storage texture request within supported limits should validate");

        assert_eq!(effective.max_storage_textures_per_shader_stage, 6);
        assert_eq!(effective.max_storage_textures_in_fragment_stage, 6);
        assert_eq!(effective.max_storage_textures_in_vertex_stage, 6);
    }

    #[test]
    fn storage_buffers_per_shader_stage_request_pins_core_stage_limits() {
        let mut supported = Limits::DEFAULT;
        supported.max_storage_buffers_per_shader_stage = 10;
        supported.max_storage_buffers_in_vertex_stage = 10;
        supported.max_storage_buffers_in_fragment_stage = 10;

        let mut required = Limits::DEFAULT;
        required.max_storage_buffers_per_shader_stage = 10;

        let effective = supported
            .validate_required_limits(Some(&required))
            .expect("per-shader-stage storage buffer request should validate");

        assert_eq!(effective.max_storage_buffers_per_shader_stage, 10);
        assert_eq!(effective.max_storage_buffers_in_vertex_stage, 10);
        assert_eq!(effective.max_storage_buffers_in_fragment_stage, 10);
    }
}
