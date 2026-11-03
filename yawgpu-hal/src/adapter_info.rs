//! Adapter identity and host-independent backend decisions.

/// Hardware classification reported by an adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalAdapterType {
    /// Dedicated graphics processor, including virtual GPUs.
    Discrete,
    /// Graphics processor with unified or integrated memory.
    Integrated,
    /// Software CPU adapter.
    Cpu,
    /// Unrecognized hardware classification.
    Unknown,
}

/// Driver and hardware identity reported by the backend.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct HalAdapterInfo {
    /// PCI or backend-defined vendor identifier.
    pub vendor_id: u32,
    /// PCI or backend-defined device identifier, or zero when unavailable.
    pub device_id: u32,
    /// Hardware classification.
    pub adapter_type: HalAdapterType,
    /// Human-readable driver description.
    pub driver_description: String,
    /// Dawn's public vendor name.
    pub vendor_name: String,
    /// Dawn's public architecture name.
    pub architecture_name: String,
}

impl HalAdapterInfo {
    pub(crate) fn new(
        vendor_id: u32,
        device_id: u32,
        adapter_type: HalAdapterType,
        driver_description: String,
    ) -> Self {
        Self {
            vendor_id,
            device_id,
            adapter_type,
            driver_description,
            vendor_name: crate::gpu_info::vendor_name(vendor_id).to_owned(),
            architecture_name: crate::gpu_info::architecture_name(vendor_id, device_id).to_owned(),
        }
    }
}

#[cfg(any(test, feature = "vulkan"))]
pub(crate) fn vulkan_driver_version(vendor: u32, raw: u32, windows: bool) -> String {
    if vendor == 0x10de {
        format!(
            "{}.{}.{}.{}",
            (raw >> 22) & 0x3ff,
            (raw >> 14) & 0xff,
            (raw >> 6) & 0xff,
            raw & 0x3f
        )
    } else if vendor == 0x8086 && windows {
        format!("{}.{}", (raw >> 14) & 0xffff, raw & 0x3fff)
    } else {
        format!("{}.{}.{}", raw >> 22, (raw >> 12) & 0x3ff, raw & 0xfff)
    }
}

#[cfg(any(test, feature = "vulkan"))]
pub(crate) fn vulkan_driver_description(version: &str, driver: Option<(&str, &str)>) -> String {
    let Some((name, info)) = driver else {
        return format!("Vulkan driver version {version}");
    };
    let mut result = name.to_owned();
    if !info.is_empty() {
        result.push_str(": ");
        result.push_str(info);
    }
    if !result.contains(version) {
        result.push(' ');
        result.push_str(version);
    }
    result
}

#[cfg(any(test, feature = "metal"))]
pub(crate) fn metal_vendor_id(name: &str) -> u32 {
    [
        ("AMD", 0x1002),
        ("Apple", 0x106b),
        ("Radeon", 0x1002),
        ("Intel", 0x8086),
        ("Geforce", 0x10de),
        ("Quadro", 0x10de),
    ]
    .into_iter()
    .find_map(|(trademark, id)| name.contains(trademark).then_some(id))
    .unwrap_or(0)
}

#[cfg(any(test, feature = "metal"))]
pub(crate) fn metal_adapter_type(ios: bool, unified: bool) -> HalAdapterType {
    if ios || unified {
        HalAdapterType::Integrated
    } else {
        HalAdapterType::Discrete
    }
}

#[cfg(any(test, feature = "metal"))]
pub(crate) fn metal_architecture(
    metal3_available: bool,
    metal3: bool,
    common3: bool,
    common2: bool,
    common1: bool,
) -> &'static str {
    if metal3_available {
        return if metal3 { "metal-3" } else { "" };
    }
    [
        (common3, "common-3"),
        (common2, "common-2"),
        (common1, "common-1"),
    ]
    .into_iter()
    .find_map(|(supported, name)| supported.then_some(name))
    .unwrap_or("")
}

#[cfg(any(test, feature = "gles"))]
pub(crate) fn gles_info(vendor: &str, renderer: &str, version: &str) -> HalAdapterInfo {
    let vendor_id = [
        ("ATI", 0x1002),
        ("ARM", 0x13b5),
        ("Imagination", 0x1010),
        ("Intel", 0x8086),
        ("NVIDIA", 0x10de),
        ("Qualcomm", 0x5143),
        ("Samsung", 0x144d),
    ]
    .into_iter()
    .find_map(|(name, id)| vendor.contains(name).then_some(id))
    .unwrap_or(0);
    let adapter_type = if renderer.contains("SwiftShader") {
        HalAdapterType::Cpu
    } else {
        HalAdapterType::Unknown
    };
    HalAdapterInfo::new(
        vendor_id,
        angle_device_id(renderer),
        adapter_type,
        format!("OpenGL version {version}"),
    )
}

#[cfg(any(test, feature = "gles"))]
fn angle_device_id(renderer: &str) -> u32 {
    if !renderer.starts_with("ANGLE") {
        return 0;
    }
    // Dawn checks the lowercase prefix before the uppercase one.
    let Some(pos) = renderer.find("(0x").or_else(|| renderer.find("(0X")) else {
        return 0;
    };
    let hex: String = renderer[pos + 3..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .collect();
    // Unlike Dawn's throwing std::stoul, overflow beyond u64 returns zero here.
    u64::from_str_radix(&hex, 16)
        .map(|id| id as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_version_layouts_and_description() {
        assert_eq!(
            vulkan_driver_version(0x10de, (555 << 22) | (12 << 14) | (34 << 6) | 5, true),
            "555.12.34.5"
        );
        assert_eq!(
            vulkan_driver_version(0x8086, (101 << 14) | 1234, true),
            "101.1234"
        );
        assert_eq!(
            vulkan_driver_version(0x8086, (1 << 30) | (101 << 14) | 1234, true),
            "101.1234"
        );
        let raw = (1 << 22) | (2 << 12) | 3;
        for (vendor, windows) in [(0x8086, false), (0x1002, true), (0, false)] {
            assert_eq!(vulkan_driver_version(vendor, raw, windows), "1.2.3");
        }
        assert_eq!(
            vulkan_driver_description("1.2.3", None),
            "Vulkan driver version 1.2.3"
        );
        assert_eq!(
            vulkan_driver_description("1.2.3", Some(("driver", "details"))),
            "driver: details 1.2.3"
        );
        assert_eq!(
            vulkan_driver_description("1.2.3", Some(("driver", ""))),
            "driver 1.2.3"
        );
        assert_eq!(
            vulkan_driver_description("1.2.3", Some(("driver", "1.2.3"))),
            "driver: 1.2.3"
        );
        assert_eq!(
            vulkan_driver_description("1.2.3", Some(("driver 1.2.3", ""))),
            "driver 1.2.3"
        );
    }

    #[test]
    fn metal_identity_decisions() {
        for (name, id) in [
            ("AMD Apple", 0x1002),
            ("Apple Radeon", 0x106b),
            ("Radeon Intel", 0x1002),
            ("Intel Geforce", 0x8086),
            ("Geforce Quadro", 0x10de),
            ("Quadro", 0x10de),
            ("GeForce", 0),
            ("unknown", 0),
        ] {
            assert_eq!(metal_vendor_id(name), id);
        }
        assert_eq!(metal_adapter_type(false, false), HalAdapterType::Discrete);
        for (ios, unified) in [(true, false), (true, true), (false, true)] {
            assert_eq!(metal_adapter_type(ios, unified), HalAdapterType::Integrated);
        }
        for (families, expected) in [
            ([true, true, true, true, true], "metal-3"),
            ([true, false, true, true, true], ""),
            ([true, false, false, false, false], ""),
            ([false, true, true, true, true], "common-3"),
            ([false, false, true, true, true], "common-3"),
            ([false, false, false, true, true], "common-2"),
            ([false, false, false, false, true], "common-1"),
            ([false; 5], ""),
        ] {
            assert_eq!(
                metal_architecture(
                    families[0],
                    families[1],
                    families[2],
                    families[3],
                    families[4]
                ),
                expected
            );
        }
    }

    #[test]
    fn gles_identity_and_angle_parsing() {
        for (vendor, id) in [
            ("ATI ARM", 0x1002),
            ("ARM Imagination", 0x13b5),
            ("Imagination Intel", 0x1010),
            ("Intel NVIDIA", 0x8086),
            ("NVIDIA Qualcomm", 0x10de),
            ("Qualcomm Samsung", 0x5143),
            ("Samsung", 0x144d),
            ("unknown", 0),
        ] {
            assert_eq!(gles_info(vendor, "renderer", "3.1").vendor_id, id);
        }
        for (renderer, id) in [
            ("ANGLE GPU (0x1a2B)", 0x1a2b),
            ("ANGLE (0XABCD)", 0xabcd),
            ("ANGLE (0x)", 0),
            ("ANGLE (0xno)", 0),
            ("ANGLE GPU", 0),
            ("GPU (0x123)", 0),
            ("ANGLE (0X12) (0x34)", 0x34),
            ("ANGLE (0x100000001)", 1),
            ("ANGLE (0x10000000000000000)", 0),
        ] {
            assert_eq!(angle_device_id(renderer), id);
        }
        let info = gles_info("NVIDIA", "ANGLE SwiftShader (0x1234)", "3.2");
        assert_eq!(info.adapter_type, HalAdapterType::Cpu);
        assert_eq!(info.driver_description, "OpenGL version 3.2");
        assert_eq!(
            gles_info("", "hardware", "").adapter_type,
            HalAdapterType::Unknown
        );
    }
}
