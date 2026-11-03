//! Device-selected Tint writer workarounds and host-independent gate logic.

/// First macOS version with the Apple unsigned division/modulo driver fix.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) const METAL_U32_DIV_MOD_FIXED_VERSION: (isize, isize, isize) = (26, 6, 1);

#[cfg(all(test, any(feature = "metal", feature = "vulkan")))]
pub(crate) static FORCE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Returns (Intel, AMD) flags using Dawn's case-sensitive vendor-name fallback order.
#[cfg(any(test, feature = "metal"))]
pub(crate) fn metal_vendor_flags(name: &str) -> (bool, bool) {
    [
        ("AMD", (false, true)),
        ("Apple", (false, false)),
        ("Radeon", (false, true)),
        ("Intel", (true, false)),
        ("Geforce", (false, false)),
        ("Quadro", (false, false)),
    ]
    .into_iter()
    .find_map(|(trademark, flags)| name.contains(trademark).then_some(flags))
    .unwrap_or((false, false))
}

/// ARM's Vulkan vendor ID, shared by Dawn-derived device gates.
#[cfg(any(test, feature = "vulkan"))]
const VENDOR_ARM: u32 = 0x13b5;
/// Qualcomm's PCI Vulkan vendor ID.
#[cfg(any(test, feature = "vulkan"))]
const VENDOR_QUALCOMM: u32 = 0x5143;
/// Qualcomm's ACPI Vulkan vendor ID.
#[cfg(any(test, feature = "vulkan"))]
const VENDOR_QUALCOMM_ACPI: u32 = 0x4d4f4351;

/// Selects Dawn's SPIR-V 1.4 path from advertised extensions and device identity.
#[cfg(any(test, feature = "vulkan"))]
pub(crate) fn use_spirv_1_4(
    spirv_extension: bool,
    float_controls_extension: bool,
    is_android: bool,
    vendor_id: u32,
    device_id: u32,
) -> bool {
    // Dawn gpu_info.json: Adreno 7xx has OpCopyLogical driver bugs.
    let adreno_7xx = (vendor_id == VENDOR_QUALCOMM
        && matches!(
            device_id & 0xff000000,
            0x07000000 | 0x43000000 | 0x36000000 | 0x37000000
        ))
        || (vendor_id == VENDOR_QUALCOMM_ACPI
            && matches!(device_id & 0xffffff00, 0x37314400 | 0x36334300));
    spirv_extension
        && float_controls_extension
        && !(is_android && vendor_id == VENDOR_ARM)
        && !adreno_7xx
}

/// Writer features and driver workarounds selected once when a HAL device is created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub struct HalShaderWorkarounds {
    /// SPIR-V: emit version 1.4 with both required Vulkan extensions enabled.
    pub use_spirv_1_4: bool,
    /// MSL: skip the demote transform on non-Apple GPUs.
    pub disable_demote_to_helper: bool,
    /// SPIR-V: use the enabled demote-to-helper feature.
    pub use_demote_to_helper_invocation: bool,
    /// SPIR-V: use enabled native packed integer dot product support.
    /// False by default, so Tint polyfills without device feature information.
    pub native_dot_4x8_packed: bool,
    /// SPIR-V: rely on enabled robustImageAccess2 instead of image clamps.
    pub disable_image_robustness: bool,
    /// SPIR-V: rely on robustBufferAccess2 for runtime-sized storage arrays.
    pub disable_storage_runtime_sized_array_index_clamping: bool,
    /// SPIR-V: initialize workgroup variables with OpConstantNull.
    pub use_zero_initialize_workgroup_memory: bool,
    /// SPIR-V: add MaximallyReconvergesKHR to entry points.
    pub use_maximal_reconvergence: bool,
    /// SPIR-V: add SubgroupUniformControlFlowKHR to compute and fragment entry points.
    pub use_subgroup_uniform_control_flow: bool,
    /// MSL: inject a non-constant zero into unsigned divide and modulo.
    pub fix_u32_div_mod: bool,
    /// MSL/SPIR-V: polyfill dynamically indexed boolean-vector stores.
    pub polyfill_bool_vec_dynamic_store: bool,
    /// SPIR-V: replace workgroup atomic stores with atomic exchanges.
    pub replace_workgroup_atomic_store_with_exchange: bool,
    /// SPIR-V: replace unsigned comparisons against zero.
    pub replace_unsigned_compare_zero: bool,
    /// MSL: Keep f16 constants out of module scope.
    pub disable_module_constant_f16: bool,
    /// MSL: Polyfill signed normalized 16-bit unpacking.
    pub polyfill_unpack_2x16_snorm: bool,
    /// MSL: Polyfill unsigned normalized 16-bit unpacking.
    pub polyfill_unpack_2x16_unorm: bool,
    /// MSL: Polyfill hyperbolic tangent of f16 values.
    pub polyfill_tanh_f16: bool,
    /// MSL: Polyfill floating-point clamp.
    pub polyfill_clamp_float: bool,
    /// MSL/SPIR-V: Collapse nested subgroup minimum and maximum operations.
    pub collapse_subgroup_min_max: bool,
    /// MSL: Represent workgroup booleans as u32.
    pub replace_workgroup_bool_with_u32: bool,
    /// MSL/SPIR-V: Scalarize vector maximum, minimum and clamp operations.
    pub scalarize_max_min_clamp: bool,
    /// SPIR-V: Polyfill floating-point negation.
    pub polyfill_float_negation: bool,
    /// SPIR-V: Polyfill floating-point absolute value.
    pub polyfill_float_abs: bool,
    /// SPIR-V: Polyfill length of scalar floating-point values.
    pub polyfill_length_scalar_float: bool,
    /// SPIR-V: Polyfill distance between scalar floating-point values.
    pub polyfill_distance_scalar_float: bool,
    /// SPIR-V: Lower switch statements to if/else statements.
    pub polyfill_case_switch: bool,
    /// SPIR-V: Polyfill depth comparisons on 2D textures.
    pub texture_sample_compare_2d_polyfill: bool,
    /// SPIR-V: Polyfill normalized 8-bit packing and unpacking.
    pub polyfill_pack_unpack_4x8_norm: bool,
    /// SPIR-V: Specialize handle parameters using direct variable access.
    pub dva_transform_handle: bool,
    /// SPIR-V: Lower f16 saturate to minimum and maximum operations.
    pub polyfill_saturate_as_min_max_f16: bool,
    /// SPIR-V: Pass matrix arguments to user functions by pointer.
    pub pass_matrix_by_pointer: bool,
}

impl HalShaderWorkarounds {
    #[cfg(any(test, feature = "metal"))]
    pub(crate) fn for_metal(
        is_macos: bool,
        is_apple: bool,
        is_intel: bool,
        os_at_least_26_6_1: bool,
        is_amd: bool,
        supports_apple9: bool,
    ) -> Self {
        Self {
            disable_demote_to_helper: !is_apple,
            fix_u32_div_mod: is_macos && is_apple && !os_at_least_26_6_1,
            polyfill_bool_vec_dynamic_store: is_macos && is_intel,
            disable_module_constant_f16: is_intel,
            polyfill_unpack_2x16_snorm: is_amd || supports_apple9,
            polyfill_unpack_2x16_unorm: supports_apple9,
            polyfill_tanh_f16: is_amd,
            polyfill_clamp_float: is_amd,
            collapse_subgroup_min_max: is_amd,
            replace_workgroup_bool_with_u32: is_amd || is_intel,
            ..Self::default()
        }
    }

    #[cfg(any(test, feature = "vulkan"))]
    pub(crate) fn for_vulkan(
        vendor_id: u32,
        driver_version: u32,
        device_id: u32,
        driver_id: Option<u32>,
        is_android: bool,
        is_windows: bool,
    ) -> Self {
        let samsung_android = is_android && vendor_id == 0x144d;
        let version = (
            driver_version >> 22,
            (driver_version >> 12) & 0x3ff,
            driver_version & 0xfff,
        );
        let facts =
            VulkanDriverPredicates::new(vendor_id, device_id, driver_id, is_android, is_windows);
        // Dawn's lexicographic comparison sorts three-element 25.3.0 below {25, 3, 0, 0}.
        let bad_amd_mesa = facts.is_amd_mesa && version <= (25, 3, 0);
        Self {
            polyfill_bool_vec_dynamic_store: is_windows && vendor_id == 0x8086,
            replace_workgroup_atomic_store_with_exchange: samsung_android,
            // Dawn's common/GPUInfo.cpp DriverVersion::operator<=> uses
            // lexicographical_compare_three_way: the decoded 3-element 25.0.0
            // is less than the 4-element gate {25, 0, 0, 0} (equal shorter prefix).
            replace_unsigned_compare_zero: samsung_android && version <= (25, 0, 0),
            polyfill_float_negation: bad_amd_mesa,
            polyfill_float_abs: bad_amd_mesa,
            polyfill_length_scalar_float: bad_amd_mesa,
            polyfill_distance_scalar_float: bad_amd_mesa,
            polyfill_case_switch: facts.may_be_imagination_proprietary,
            texture_sample_compare_2d_polyfill: facts.may_be_qualcomm_proprietary,
            scalarize_max_min_clamp: facts.may_be_qualcomm_proprietary,
            polyfill_pack_unpack_4x8_norm: facts.may_be_arm_proprietary,
            dva_transform_handle: facts.may_be_qualcomm_proprietary
                || facts.may_be_arm_proprietary
                || facts.is_swiftshader,
            polyfill_saturate_as_min_max_f16: facts.is_intel_mesa,
            collapse_subgroup_min_max: facts.is_windows_amd,
            pass_matrix_by_pointer: facts.is_android_qualcomm,
            ..Self::default()
        }
    }

    /// Parses force-enabled flag names without accessing the environment.
    #[cfg(any(test, feature = "metal", feature = "vulkan"))]
    pub(crate) fn parse_forced(value: Option<&str>) -> Self {
        let mut flags = Self::default();
        for name in value.unwrap_or_default().split(',').map(str::trim) {
            match name {
                "fix_u32_div_mod" => flags.fix_u32_div_mod = true,
                "polyfill_bool_vec_dynamic_store" => flags.polyfill_bool_vec_dynamic_store = true,
                "replace_workgroup_atomic_store_with_exchange" => {
                    flags.replace_workgroup_atomic_store_with_exchange = true
                }
                "replace_unsigned_compare_zero" => flags.replace_unsigned_compare_zero = true,
                "disable_module_constant_f16" => flags.disable_module_constant_f16 = true,
                "polyfill_unpack_2x16_snorm" => flags.polyfill_unpack_2x16_snorm = true,
                "polyfill_unpack_2x16_unorm" => flags.polyfill_unpack_2x16_unorm = true,
                "polyfill_tanh_f16" => flags.polyfill_tanh_f16 = true,
                "polyfill_clamp_float" => flags.polyfill_clamp_float = true,
                "collapse_subgroup_min_max" => flags.collapse_subgroup_min_max = true,
                "replace_workgroup_bool_with_u32" => flags.replace_workgroup_bool_with_u32 = true,
                "scalarize_max_min_clamp" => flags.scalarize_max_min_clamp = true,
                "polyfill_float_negation" => flags.polyfill_float_negation = true,
                "polyfill_float_abs" => flags.polyfill_float_abs = true,
                "polyfill_length_scalar_float" => flags.polyfill_length_scalar_float = true,
                "polyfill_distance_scalar_float" => flags.polyfill_distance_scalar_float = true,
                "polyfill_case_switch" => flags.polyfill_case_switch = true,
                "texture_sample_compare_2d_polyfill" => {
                    flags.texture_sample_compare_2d_polyfill = true
                }
                "polyfill_pack_unpack_4x8_norm" => flags.polyfill_pack_unpack_4x8_norm = true,
                "dva_transform_handle" => flags.dva_transform_handle = true,
                "polyfill_saturate_as_min_max_f16" => flags.polyfill_saturate_as_min_max_f16 = true,
                "pass_matrix_by_pointer" => flags.pass_matrix_by_pointer = true,
                _ => {}
            }
        }
        flags
    }

    #[cfg(any(test, feature = "metal", feature = "vulkan"))]
    pub(crate) fn with_forced(self, forced: Self) -> Self {
        Self {
            use_spirv_1_4: self.use_spirv_1_4,
            disable_demote_to_helper: self.disable_demote_to_helper,
            use_demote_to_helper_invocation: self.use_demote_to_helper_invocation,
            native_dot_4x8_packed: self.native_dot_4x8_packed,
            disable_image_robustness: self.disable_image_robustness,
            disable_storage_runtime_sized_array_index_clamping: self
                .disable_storage_runtime_sized_array_index_clamping,
            use_zero_initialize_workgroup_memory: self.use_zero_initialize_workgroup_memory,
            use_maximal_reconvergence: self.use_maximal_reconvergence,
            use_subgroup_uniform_control_flow: self.use_subgroup_uniform_control_flow,
            fix_u32_div_mod: self.fix_u32_div_mod || forced.fix_u32_div_mod,
            polyfill_bool_vec_dynamic_store: self.polyfill_bool_vec_dynamic_store
                || forced.polyfill_bool_vec_dynamic_store,
            replace_workgroup_atomic_store_with_exchange: self
                .replace_workgroup_atomic_store_with_exchange
                || forced.replace_workgroup_atomic_store_with_exchange,
            replace_unsigned_compare_zero: self.replace_unsigned_compare_zero
                || forced.replace_unsigned_compare_zero,
            disable_module_constant_f16: self.disable_module_constant_f16
                || forced.disable_module_constant_f16,
            polyfill_unpack_2x16_snorm: self.polyfill_unpack_2x16_snorm
                || forced.polyfill_unpack_2x16_snorm,
            polyfill_unpack_2x16_unorm: self.polyfill_unpack_2x16_unorm
                || forced.polyfill_unpack_2x16_unorm,
            polyfill_tanh_f16: self.polyfill_tanh_f16 || forced.polyfill_tanh_f16,
            polyfill_clamp_float: self.polyfill_clamp_float || forced.polyfill_clamp_float,
            collapse_subgroup_min_max: self.collapse_subgroup_min_max
                || forced.collapse_subgroup_min_max,
            replace_workgroup_bool_with_u32: self.replace_workgroup_bool_with_u32
                || forced.replace_workgroup_bool_with_u32,
            scalarize_max_min_clamp: self.scalarize_max_min_clamp || forced.scalarize_max_min_clamp,
            polyfill_float_negation: self.polyfill_float_negation || forced.polyfill_float_negation,
            polyfill_float_abs: self.polyfill_float_abs || forced.polyfill_float_abs,
            polyfill_length_scalar_float: self.polyfill_length_scalar_float
                || forced.polyfill_length_scalar_float,
            polyfill_distance_scalar_float: self.polyfill_distance_scalar_float
                || forced.polyfill_distance_scalar_float,
            polyfill_case_switch: self.polyfill_case_switch || forced.polyfill_case_switch,
            texture_sample_compare_2d_polyfill: self.texture_sample_compare_2d_polyfill
                || forced.texture_sample_compare_2d_polyfill,
            polyfill_pack_unpack_4x8_norm: self.polyfill_pack_unpack_4x8_norm
                || forced.polyfill_pack_unpack_4x8_norm,
            dva_transform_handle: self.dva_transform_handle || forced.dva_transform_handle,
            polyfill_saturate_as_min_max_f16: self.polyfill_saturate_as_min_max_f16
                || forced.polyfill_saturate_as_min_max_f16,
            pass_matrix_by_pointer: self.pass_matrix_by_pointer || forced.pass_matrix_by_pointer,
        }
    }
}

/// Dawn's Vulkan driver predicates, independent of Vulkan and the host platform.
#[cfg(any(test, feature = "vulkan"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VulkanDriverPredicates {
    pub(crate) is_amd_mesa: bool,
    pub(crate) is_intel_mesa: bool,
    pub(crate) may_be_qualcomm_proprietary: bool,
    pub(crate) may_be_arm_proprietary: bool,
    pub(crate) may_be_imagination_proprietary: bool,
    pub(crate) is_windows_amd: bool,
    pub(crate) is_android_qualcomm: bool,
    pub(crate) is_swiftshader: bool,
}

#[cfg(any(test, feature = "vulkan"))]
impl VulkanDriverPredicates {
    pub(crate) fn new(
        vendor_id: u32,
        device_id: u32,
        driver_id: Option<u32>,
        is_android: bool,
        is_windows: bool,
    ) -> Self {
        // VkDriverId values from vulkan_core.h. Mesa predicates intentionally
        // use only driverID, whereas proprietary predicates also require vendorID.
        const VK_DRIVER_ID_MESA_RADV: u32 = 3;
        const VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA: u32 = 6;
        const VK_DRIVER_ID_IMAGINATION_PROPRIETARY: u32 = 7;
        const VK_DRIVER_ID_QUALCOMM_PROPRIETARY: u32 = 8;
        const VK_DRIVER_ID_ARM_PROPRIETARY: u32 = 9;
        const VENDOR_AMD: u32 = 0x1002;
        const VENDOR_IMAGINATION: u32 = 0x1010;
        const VENDOR_GOOGLE: u32 = 0x1ae0;
        // Dawn src/dawn/gpu_info.json: Google SwiftShader.
        const DEVICE_SWIFTSHADER: u32 = 0xc0de;
        Self {
            is_amd_mesa: driver_id == Some(VK_DRIVER_ID_MESA_RADV),
            is_intel_mesa: driver_id == Some(VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA),
            may_be_qualcomm_proprietary: vendor_id == VENDOR_QUALCOMM
                && (driver_id.is_none() || driver_id == Some(VK_DRIVER_ID_QUALCOMM_PROPRIETARY)),
            may_be_arm_proprietary: vendor_id == VENDOR_ARM
                && (driver_id.is_none() || driver_id == Some(VK_DRIVER_ID_ARM_PROPRIETARY)),
            may_be_imagination_proprietary: vendor_id == VENDOR_IMAGINATION
                && (driver_id.is_none() || driver_id == Some(VK_DRIVER_ID_IMAGINATION_PROPRIETARY)),
            is_windows_amd: is_windows && vendor_id == VENDOR_AMD,
            is_android_qualcomm: is_android && vendor_id == VENDOR_QUALCOMM,
            is_swiftshader: vendor_id == VENDOR_GOOGLE && device_id == DEVICE_SWIFTSHADER,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_1_4_decision_matches_dawn_device_gates() {
        for (spirv, controls, android, vendor, device, expected) in [
            (true, true, false, 0x10de, 0x2d04, true),
            (false, true, false, 0x10de, 0x2d04, false),
            (true, false, false, 0x10de, 0x2d04, false),
            (false, false, false, 0x10de, 0x2d04, false),
            (true, true, true, 0x13b5, 0, false),
            (true, true, false, 0x13b5, 0, true),
            (true, true, true, 0x10de, 0x2d04, true),
            (true, true, false, 0x5143, 0x07000001, false),
            (true, true, true, 0x5143, 0x430000ff, false),
            (true, true, false, 0x5143, 0x3600ffff, false),
            (true, true, false, 0x5143, 0x37ffffff, false),
            (true, true, false, 0x4d4f4351, 0x37314401, false),
            (true, true, true, 0x4d4f4351, 0x363343ff, false),
            (true, true, true, 0x5143, 0x06000001, true),
            (true, true, true, 0x5143, 0x44000001, true),
            (true, true, false, 0x4d4f4351, 0x36334400, true),
            (true, true, true, 0x4d4f4351, 0x41333801, true),
            (true, true, true, 0x4d4f4351, 0x354646ff, true),
            (true, true, false, 0x10de, 0x07000001, true),
            (true, true, false, 0x5143, 0x37314401, false),
            (true, true, false, 0x4d4f4351, 0x07000001, true),
        ] {
            assert_eq!(use_spirv_1_4(spirv, controls, android, vendor, device), expected,
                "spirv={spirv}, controls={controls}, android={android}, vendor={vendor:#x}, device={device:#x}");
        }
        assert!(!HalShaderWorkarounds::default().use_spirv_1_4);
        let flags = HalShaderWorkarounds {
            use_spirv_1_4: true,
            ..Default::default()
        };
        assert!(flags.with_forced(Default::default()).use_spirv_1_4);
        assert!(
            !HalShaderWorkarounds::for_metal(false, true, false, false, false, false).use_spirv_1_4
        );
    }

    #[test]
    fn metal_vendor_flags_match_dawn_name_fallback() {
        for (name, expected) in [
            ("AMD Radeon Pro 5500M", (false, true)),
            ("Radeon Pro 5500M", (false, true)),
            ("Intel(R) UHD Graphics 630", (true, false)),
            ("Apple M2", (false, false)),
            ("Some Gamdev GPU", (false, false)),
            ("amd GPU", (false, false)),
            ("radeon GPU", (false, false)),
            ("intel GPU", (false, false)),
            ("Intel AMD", (false, true)),
            ("Intel Apple Radeon", (false, false)),
            ("Intel Radeon", (false, true)),
        ] {
            assert_eq!(metal_vendor_flags(name), expected, "{name}");
        }
    }

    #[test]
    fn force_each_vendor_option_and_preserve_gate_flags() {
        for (name, expected) in [
            (
                "disable_module_constant_f16",
                HalShaderWorkarounds {
                    disable_module_constant_f16: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_unpack_2x16_snorm",
                HalShaderWorkarounds {
                    polyfill_unpack_2x16_snorm: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_unpack_2x16_unorm",
                HalShaderWorkarounds {
                    polyfill_unpack_2x16_unorm: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_tanh_f16",
                HalShaderWorkarounds {
                    polyfill_tanh_f16: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_clamp_float",
                HalShaderWorkarounds {
                    polyfill_clamp_float: true,
                    ..Default::default()
                },
            ),
            (
                "collapse_subgroup_min_max",
                HalShaderWorkarounds {
                    collapse_subgroup_min_max: true,
                    ..Default::default()
                },
            ),
            (
                "replace_workgroup_bool_with_u32",
                HalShaderWorkarounds {
                    replace_workgroup_bool_with_u32: true,
                    ..Default::default()
                },
            ),
            (
                "scalarize_max_min_clamp",
                HalShaderWorkarounds {
                    scalarize_max_min_clamp: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_float_negation",
                HalShaderWorkarounds {
                    polyfill_float_negation: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_float_abs",
                HalShaderWorkarounds {
                    polyfill_float_abs: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_length_scalar_float",
                HalShaderWorkarounds {
                    polyfill_length_scalar_float: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_distance_scalar_float",
                HalShaderWorkarounds {
                    polyfill_distance_scalar_float: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_case_switch",
                HalShaderWorkarounds {
                    polyfill_case_switch: true,
                    ..Default::default()
                },
            ),
            (
                "texture_sample_compare_2d_polyfill",
                HalShaderWorkarounds {
                    texture_sample_compare_2d_polyfill: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_pack_unpack_4x8_norm",
                HalShaderWorkarounds {
                    polyfill_pack_unpack_4x8_norm: true,
                    ..Default::default()
                },
            ),
            (
                "dva_transform_handle",
                HalShaderWorkarounds {
                    dva_transform_handle: true,
                    ..Default::default()
                },
            ),
            (
                "polyfill_saturate_as_min_max_f16",
                HalShaderWorkarounds {
                    polyfill_saturate_as_min_max_f16: true,
                    ..Default::default()
                },
            ),
            (
                "pass_matrix_by_pointer",
                HalShaderWorkarounds {
                    pass_matrix_by_pointer: true,
                    ..Default::default()
                },
            ),
        ] {
            let parsed =
                HalShaderWorkarounds::parse_forced(Some(&format!(" {name},{name},unknown ")));
            assert_eq!(parsed, expected);
            assert_eq!(
                HalShaderWorkarounds::default().with_forced(parsed),
                expected
            );
            assert_eq!(expected.with_forced(Default::default()), expected);
            assert_eq!(expected.with_forced(parsed), expected);
        }
    }

    #[test]
    fn metal_vendor_workarounds_are_not_platform_gated() {
        for macos in [false, true] {
            for (apple, intel, amd) in [
                (true, false, false),
                (false, true, false),
                (false, false, true),
                (false, false, false),
            ] {
                for apple9 in [false, true] {
                    let f = HalShaderWorkarounds::for_metal(macos, apple, intel, true, amd, apple9);
                    assert_eq!(f.disable_module_constant_f16, intel);
                    assert_eq!(f.polyfill_unpack_2x16_snorm, amd || apple9);
                    assert_eq!(f.polyfill_unpack_2x16_unorm, apple9);
                    assert_eq!(f.polyfill_tanh_f16, amd);
                    assert_eq!(f.polyfill_clamp_float, amd);
                    assert_eq!(f.collapse_subgroup_min_max, amd);
                    assert_eq!(f.replace_workgroup_bool_with_u32, amd || intel);
                    assert!(!f.scalarize_max_min_clamp);
                    assert!(!f.dva_transform_handle);
                }
            }
        }
    }

    #[test]
    fn vulkan_driver_predicates_and_workarounds() {
        // Cross every vendor with every relevant and unknown driver ID, plus
        // unavailable driver properties. This catches both wrong-vendor matches
        // and accidental proprietary fallback for a known nonproprietary driver.
        for vendor in [
            0x1002, 0x8086, 0x5143, 0x4d4f4351, 0x144d, 0x13b5, 0x1010, 0x1ae0, 0x106b, 0x10de, 0,
        ] {
            for driver in [
                None,
                Some(3),
                Some(6),
                Some(7),
                Some(8),
                Some(9),
                Some(0),
                Some(999),
            ] {
                for device in [0xc0de, 0] {
                    for android in [false, true] {
                        for windows in [false, true] {
                            let facts = VulkanDriverPredicates::new(
                                vendor, device, driver, android, windows,
                            );
                            let qualcomm =
                                vendor == 0x5143 && (driver.is_none() || driver == Some(8));
                            let arm = vendor == 0x13b5 && (driver.is_none() || driver == Some(9));
                            let imagination =
                                vendor == 0x1010 && (driver.is_none() || driver == Some(7));
                            let swiftshader = vendor == 0x1ae0 && device == 0xc0de;
                            assert_eq!(facts.is_amd_mesa, driver == Some(3));
                            assert_eq!(facts.is_intel_mesa, driver == Some(6));
                            assert_eq!(facts.may_be_qualcomm_proprietary, qualcomm);
                            assert_eq!(facts.may_be_arm_proprietary, arm);
                            assert_eq!(facts.may_be_imagination_proprietary, imagination);
                            assert_eq!(facts.is_windows_amd, windows && vendor == 0x1002);
                            assert_eq!(facts.is_android_qualcomm, android && vendor == 0x5143);
                            if matches!(vendor, 0x4d4f4351 | 0x144d) {
                                assert!(!facts.may_be_qualcomm_proprietary);
                                assert!(!facts.is_android_qualcomm);
                            }
                            assert_eq!(facts.is_swiftshader, swiftshader);
                            for (version, bad) in [
                                ((25, 2, 4095), true),
                                ((25, 3, 0), true),
                                ((25, 3, 1), false),
                                ((25, 4, 0), false),
                                ((26, 0, 0), false),
                            ] {
                                let raw = (version.0 << 22) | (version.1 << 12) | version.2;
                                let f = HalShaderWorkarounds::for_vulkan(
                                    vendor, raw, device, driver, android, windows,
                                );
                                assert_eq!(f.polyfill_float_negation, driver == Some(3) && bad);
                                assert_eq!(f.polyfill_float_abs, driver == Some(3) && bad);
                                assert_eq!(
                                    f.polyfill_length_scalar_float,
                                    driver == Some(3) && bad
                                );
                                assert_eq!(
                                    f.polyfill_distance_scalar_float,
                                    driver == Some(3) && bad
                                );
                                assert_eq!(f.polyfill_case_switch, imagination);
                                assert_eq!(f.texture_sample_compare_2d_polyfill, qualcomm);
                                assert_eq!(f.scalarize_max_min_clamp, qualcomm);
                                assert_eq!(f.polyfill_pack_unpack_4x8_norm, arm);
                                assert_eq!(f.dva_transform_handle, qualcomm || arm || swiftshader);
                                assert_eq!(f.polyfill_saturate_as_min_max_f16, driver == Some(6));
                                assert_eq!(
                                    f.collapse_subgroup_min_max,
                                    windows && vendor == 0x1002
                                );
                                assert_eq!(f.pass_matrix_by_pointer, android && vendor == 0x5143);
                                assert!(!f.disable_module_constant_f16);
                                assert!(!f.replace_workgroup_bool_with_u32);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn vulkan_default_writer_flags_disable_native_dot_products() {
        assert!(!HalShaderWorkarounds::default().native_dot_4x8_packed);
        assert!(!HalShaderWorkarounds::default().use_maximal_reconvergence);
        assert!(!HalShaderWorkarounds::default().use_subgroup_uniform_control_flow);
    }

    #[test]
    fn metal_u32_div_mod_fixed_version_matches_dawn() {
        assert_eq!(METAL_U32_DIV_MOD_FIXED_VERSION, (26, 6, 1));
    }

    #[test]
    fn metal_gate_boundaries() {
        for fixed in [false, true] {
            for macos in [false, true] {
                // false also covers iOS Apple GPUs.
                for (apple, intel) in [(true, false), (false, true), (false, false)] {
                    let flags =
                        HalShaderWorkarounds::for_metal(macos, apple, intel, fixed, false, false);
                    assert_eq!(flags.disable_demote_to_helper, !apple);
                    assert_eq!(flags.fix_u32_div_mod, macos && apple && !fixed);
                    assert_eq!(flags.polyfill_bool_vec_dynamic_store, macos && intel);
                    assert!(!flags.replace_workgroup_atomic_store_with_exchange);
                    assert!(!flags.replace_unsigned_compare_zero);
                }
            }
        }
    }

    #[test]
    fn vulkan_gate_boundaries() {
        for vendor in [0x8086, 0x144d, 0x10de, 0] {
            for (android, windows) in [(false, false), (true, false), (false, true)] {
                for ((major, minor, patch), below_dawn_gate) in [
                    ((0, 0, 0), true),
                    ((24, 1023, 4095), true),
                    ((25, 0, 0), true),
                    ((25, 0, 1), false),
                    ((25, 1, 0), false),
                ] {
                    let flags = HalShaderWorkarounds::for_vulkan(
                        vendor,
                        (major << 22) | (minor << 12) | patch,
                        0,
                        None,
                        android,
                        windows,
                    );
                    assert!(!flags.fix_u32_div_mod);
                    assert_eq!(
                        flags.polyfill_bool_vec_dynamic_store,
                        windows && vendor == 0x8086
                    );
                    assert_eq!(
                        flags.replace_workgroup_atomic_store_with_exchange,
                        android && vendor == 0x144d
                    );
                    assert_eq!(
                        flags.replace_unsigned_compare_zero,
                        android && vendor == 0x144d && below_dawn_gate
                    );
                }
            }
        }
    }

    #[test]
    fn forced_parser_and_gate_union() {
        for value in [None, Some(""), Some("unknown,FIX_U32_DIV_MOD")] {
            assert_eq!(
                HalShaderWorkarounds::parse_forced(value),
                HalShaderWorkarounds::default()
            );
        }
        let names = [
            "fix_u32_div_mod",
            "polyfill_bool_vec_dynamic_store",
            "replace_workgroup_atomic_store_with_exchange",
            "replace_unsigned_compare_zero",
        ];
        for (index, name) in names.iter().enumerate() {
            let flags = HalShaderWorkarounds::parse_forced(Some(name));
            assert_eq!(
                [
                    flags.fix_u32_div_mod,
                    flags.polyfill_bool_vec_dynamic_store,
                    flags.replace_workgroup_atomic_store_with_exchange,
                    flags.replace_unsigned_compare_zero
                ],
                std::array::from_fn(|i| i == index)
            );
        }
        let flags = HalShaderWorkarounds::parse_forced(Some(&format!(
            " {},unknown,{},{} ",
            names[0], names[1], names[0]
        )));
        let gate = HalShaderWorkarounds {
            use_maximal_reconvergence: true,
            ..HalShaderWorkarounds::for_vulkan(0x144d, 24 << 22, 0, None, true, false)
        };
        let combined = gate.with_forced(flags);
        assert!(combined.use_maximal_reconvergence);
        assert!(!combined.use_subgroup_uniform_control_flow);
        assert!(
            combined.fix_u32_div_mod
                && combined.polyfill_bool_vec_dynamic_store
                && combined.replace_workgroup_atomic_store_with_exchange
                && combined.replace_unsigned_compare_zero
        );
        assert_eq!(
            combined.with_forced(HalShaderWorkarounds::default()),
            combined
        );
    }
}
