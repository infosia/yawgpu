//! Device-selected Tint writer workarounds and host-independent gate logic.

/// First macOS version with the Apple unsigned division/modulo driver fix.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) const METAL_U32_DIV_MOD_FIXED_VERSION: (isize, isize, isize) = (26, 6, 1);

#[cfg(all(test, any(feature = "metal", feature = "vulkan")))]
pub(crate) static FORCE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Writer features and driver workarounds selected once when a HAL device is created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub struct HalShaderWorkarounds {
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
}

impl HalShaderWorkarounds {
    #[cfg(any(test, feature = "metal"))]
    pub(crate) fn for_metal(
        is_macos: bool,
        is_apple: bool,
        is_intel: bool,
        os_at_least_26_6_1: bool,
    ) -> Self {
        Self {
            disable_demote_to_helper: !is_apple,
            fix_u32_div_mod: is_macos && is_apple && !os_at_least_26_6_1,
            polyfill_bool_vec_dynamic_store: is_macos && is_intel,
            ..Self::default()
        }
    }

    #[cfg(any(test, feature = "vulkan"))]
    pub(crate) fn for_vulkan(
        vendor_id: u32,
        driver_version: u32,
        is_android: bool,
        is_windows: bool,
    ) -> Self {
        let samsung_android = is_android && vendor_id == 0x144d;
        let version = (
            driver_version >> 22,
            (driver_version >> 12) & 0x3ff,
            driver_version & 0xfff,
        );
        // Dawn's common/GPUInfo.cpp DriverVersion::operator<=> uses
        // lexicographical_compare_three_way: the decoded 3-element 25.0.0
        // is less than the 4-element gate {25, 0, 0, 0} (equal shorter prefix).
        Self {
            polyfill_bool_vec_dynamic_store: is_windows && vendor_id == 0x8086,
            replace_workgroup_atomic_store_with_exchange: samsung_android,
            replace_unsigned_compare_zero: samsung_android && version <= (25, 0, 0),
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
                _ => {}
            }
        }
        flags
    }

    #[cfg(any(test, feature = "metal", feature = "vulkan"))]
    pub(crate) fn with_forced(self, forced: Self) -> Self {
        Self {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    let flags = HalShaderWorkarounds::for_metal(macos, apple, intel, fixed);
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
            ..HalShaderWorkarounds::for_vulkan(0x144d, 24 << 22, true, false)
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
