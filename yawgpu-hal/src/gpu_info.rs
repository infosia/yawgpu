//! Dawn's public vendor and architecture identity lookup.

#[path = "gpu_info_data.rs"]
mod data;

pub(crate) fn vendor_name(vendor_id: u32) -> &'static str {
    data::VENDORS
        .iter()
        .find_map(|&(id, name)| (id == vendor_id).then_some(name))
        .unwrap_or("")
}

pub(crate) fn architecture_name(vendor_id: u32, device_id: u32) -> &'static str {
    data::ARCHITECTURES
        .iter()
        .find_map(|&(id, mask, name, devices)| {
            (id == vendor_id && devices.contains(&(device_id & mask))).then_some(name)
        })
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn js_enum_case(name: &str) -> String {
        let mut words = name.split(' ');
        let mut result = words.next().unwrap().to_lowercase();
        for word in words {
            if !result.ends_with(|c: char| c.is_ascii_digit()) {
                result.push('-');
            }
            result.push_str(&word.to_lowercase());
        }
        result
    }

    #[test]
    fn names_match_dawn_samples_and_unknown_ids() {
        assert_eq!(vendor_name(0x10de), "nvidia");
        assert_eq!(vendor_name(0x106b), "apple");
        assert_eq!(vendor_name(0x1010), "img-tec");
        assert_eq!(vendor_name(0x5143), "qualcomm");
        assert_eq!(vendor_name(0), "");
        assert_eq!(architecture_name(0x5143, 0x07001234), "adreno-7xx");
        assert_eq!(architecture_name(0x1002, 0x6731), "terascale-2");
        assert_eq!(architecture_name(0x1002, 0x1636), "gcn-5");
        assert_eq!(architecture_name(0x10de, 0x2684), "lovelace");
        // The JSON's commented example has an internal Beta Rev 3 set.
        assert_eq!(architecture_name(0xdead, 0x1b30), "");
        assert_eq!(architecture_name(0, 0), "");
        assert_eq!(architecture_name(0x10de, 0xffffffff), "");
        assert_eq!(js_enum_case("Gen 9 LP"), "gen-9lp");
    }

    #[test]
    fn generated_tables_equal_pinned_json() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../third_party/dawn/src/dawn/gpu_info.json");
        if !path.exists() {
            return;
        }
        let source: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let number = |value: &serde_json::Value| {
            u32::from_str_radix(value.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
        };
        let mut vendors = Vec::new();
        let mut architectures = Vec::new();
        for (name, vendor) in source["vendors"].as_object().unwrap() {
            if name.starts_with('_') {
                continue;
            }
            let id = number(&vendor["id"]);
            vendors.push((
                id,
                js_enum_case(vendor["name_override"].as_str().unwrap_or(name)),
            ));
            for set in vendor["devices"].as_array().into_iter().flatten() {
                if set["internal"].as_bool().unwrap_or(false) {
                    continue;
                }
                let mask = if set["mask"].is_null() {
                    u32::MAX
                } else {
                    number(&set["mask"])
                };
                for (arch, ids) in set["architecture"].as_object().into_iter().flatten() {
                    if arch.starts_with('_') {
                        continue;
                    }
                    architectures.push((
                        id,
                        mask,
                        js_enum_case(arch),
                        ids.as_array()
                            .unwrap()
                            .iter()
                            .map(number)
                            .collect::<Vec<_>>(),
                    ));
                }
            }
        }
        assert_eq!(
            vendors,
            data::VENDORS
                .iter()
                .map(|&(id, name)| (id, name.to_owned()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            architectures,
            data::ARCHITECTURES
                .iter()
                .map(|&(id, mask, name, ids)| (id, mask, name.to_owned(), ids.to_vec()))
                .collect::<Vec<_>>()
        );
        // Check internal sets, including the JSON's commented example. The
        // pinned JSON currently has no internal sets in its real vendors.
        for (_, vendor) in source["vendors"].as_object().unwrap() {
            let id = number(&vendor["id"]);
            for set in vendor["devices"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|s| s["internal"].as_bool() == Some(true))
            {
                for (arch, ids) in set["architecture"].as_object().into_iter().flatten() {
                    if arch.starts_with('_') {
                        continue;
                    }
                    for device in ids.as_array().unwrap() {
                        let device = number(device);
                        if !architectures
                            .iter()
                            .any(|(v, m, _, ids)| *v == id && ids.contains(&(device & m)))
                        {
                            assert_eq!(architecture_name(id, device), "");
                        }
                    }
                }
            }
        }
    }
}
