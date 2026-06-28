use super::*;
use crate::{YaWGPUShaderSourceMSL, YAWGPU_STYPE_SHADER_SOURCE_MSL};

/// Converts shader module descriptor into the corresponding yawgpu representation.
///
/// # Safety
///
/// `descriptor.nextInChain` must be either null or a valid linked list of
/// `WGPUChainedStruct` nodes. Recognized shader-source nodes must point to
/// valid `WGPUShaderSourceWGSL` or `YaWGPUShaderSourceMSL` storage. WGSL and
/// MSL string data must be valid for their declared lengths. When an MSL node
/// has a non-zero `entryPointCount`, `entryPoints` must point to an array with
/// at least that many valid `YaWGPUMslEntryPoint` elements.
#[must_use]
pub unsafe fn map_shader_module_descriptor(
    descriptor: &native::WGPUShaderModuleDescriptor,
) -> core::ShaderModuleSource {
    let mut source = None;
    let mut chain = descriptor.nextInChain;

    while let Some(node) = chain.as_ref() {
        if node.sType == native::WGPUSType_ShaderSourceWGSL {
            if source.is_some() {
                return core::ShaderModuleSource::Invalid(
                    "shader module descriptor must contain exactly one shader source".to_owned(),
                );
            }
            let Some(wgsl) = chain.cast::<native::WGPUShaderSourceWGSL>().as_ref() else {
                return core::ShaderModuleSource::Invalid(
                    "WGSL shader source chain node must be valid".to_owned(),
                );
            };
            let code = string_view_to_str(wgsl.code).map_or_else(String::new, ToOwned::to_owned);
            source = Some(core::ShaderModuleSource::Wgsl(code));
        }
        if node.sType == YAWGPU_STYPE_SHADER_SOURCE_MSL {
            if source.is_some() {
                return core::ShaderModuleSource::Invalid(
                    "shader module descriptor must contain exactly one shader source".to_owned(),
                );
            }
            let Some(msl) = chain.cast::<YaWGPUShaderSourceMSL>().as_ref() else {
                return core::ShaderModuleSource::Invalid(
                    "MSL shader source chain node must be valid".to_owned(),
                );
            };
            source = Some(map_msl_shader_source(msl));
        }

        chain = node.next;
    }

    source.unwrap_or_else(|| {
        core::ShaderModuleSource::Invalid(
            "shader module descriptor must contain exactly one shader source".to_owned(),
        )
    })
}

unsafe fn map_msl_shader_source(msl: &YaWGPUShaderSourceMSL) -> core::ShaderModuleSource {
    #[cfg(not(feature = "shader-passthrough"))]
    {
        let _ = msl;
        core::ShaderModuleSource::Invalid("shader passthrough not enabled".to_owned())
    }
    #[cfg(feature = "shader-passthrough")]
    {
        let code = string_view_to_str(msl.code).map_or_else(String::new, ToOwned::to_owned);
        let entries = if msl.entryPointCount == 0 {
            &[][..]
        } else if msl.entryPoints.is_null() {
            return core::ShaderModuleSource::Invalid(
                "MSL shader source entryPoints must be valid".to_owned(),
            );
        } else {
            std::slice::from_raw_parts(msl.entryPoints, msl.entryPointCount)
        };
        let mut mapped_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            let Some(stage) = map_msl_entry_point_stage(entry.stage) else {
                return core::ShaderModuleSource::Invalid(
                    "MSL entry point stage must be exactly one shader stage".to_owned(),
                );
            };
            mapped_entries.push(core::MslEntryPoint {
                name: string_view_to_str(entry.name).map_or_else(String::new, ToOwned::to_owned),
                stage,
                workgroup_size: entry.workgroupSize,
            });
        }
        core::ShaderModuleSource::MslPassthrough {
            source: code,
            entries: mapped_entries,
        }
    }
}

#[cfg(feature = "shader-passthrough")]
fn map_msl_entry_point_stage(stage: native::WGPUShaderStage) -> Option<core::ShaderStage> {
    match stage {
        native::WGPUShaderStage_Vertex => Some(core::ShaderStage::Vertex),
        native::WGPUShaderStage_Fragment => Some(core::ShaderStage::Fragment),
        native::WGPUShaderStage_Compute => Some(core::ShaderStage::Compute),
        _ => None,
    }
}
