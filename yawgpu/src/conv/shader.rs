use super::*;

/// Converts a shader module descriptor chain to a core shader source.
///
/// # Safety
///
/// `descriptor.nextInChain` must be either null or a valid linked list of
/// `WGPUChainedStruct` nodes. Recognized shader-source nodes must point to
/// valid `WGPUShaderSourceWGSL` or `WGPUShaderSourceSPIRV` storage. WGSL
/// string data and SPIR-V word data must be valid for their declared lengths.
/// Converts shader module descriptor into the corresponding yawgpu representation.
#[must_use]
pub unsafe fn map_shader_module_descriptor(
    descriptor: &native::WGPUShaderModuleDescriptor,
) -> core::ShaderModuleSource {
    let mut source = None;
    let mut chain = descriptor.nextInChain;

    while let Some(node) = chain.as_ref() {
        match node.sType {
            native::WGPUSType_ShaderSourceWGSL => {
                if source.is_some() {
                    return core::ShaderModuleSource::Invalid(
                        "shader module descriptor must contain exactly one shader source"
                            .to_owned(),
                    );
                }
                let Some(wgsl) = chain.cast::<native::WGPUShaderSourceWGSL>().as_ref() else {
                    return core::ShaderModuleSource::Invalid(
                        "WGSL shader source chain node must be valid".to_owned(),
                    );
                };
                let code =
                    string_view_to_str(wgsl.code).map_or_else(String::new, ToOwned::to_owned);
                source = Some(core::ShaderModuleSource::Wgsl(code));
            }
            native::WGPUSType_ShaderSourceSPIRV => {
                if source.is_some() {
                    return core::ShaderModuleSource::Invalid(
                        "shader module descriptor must contain exactly one shader source"
                            .to_owned(),
                    );
                }
                let Some(spirv) = chain.cast::<native::WGPUShaderSourceSPIRV>().as_ref() else {
                    return core::ShaderModuleSource::Invalid(
                        "SPIR-V shader source chain node must be valid".to_owned(),
                    );
                };
                if spirv.codeSize > 0 && spirv.code.is_null() {
                    return core::ShaderModuleSource::Invalid(
                        "SPIR-V shader source code must not be null when codeSize is non-zero"
                            .to_owned(),
                    );
                }
                let words = if spirv.codeSize == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(spirv.code, spirv.codeSize as usize).to_vec()
                };
                source = Some(core::ShaderModuleSource::Spirv(words));
            }
            _ => {}
        }

        chain = node.next;
    }

    source.unwrap_or_else(|| {
        core::ShaderModuleSource::Invalid(
            "shader module descriptor must contain exactly one shader source".to_owned(),
        )
    })
}
