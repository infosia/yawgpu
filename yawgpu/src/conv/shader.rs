use super::*;

/// Converts shader module descriptor into the corresponding yawgpu representation.
///
/// # Safety
///
/// `descriptor.nextInChain` must be either null or a valid linked list of
/// `WGPUChainedStruct` nodes. Recognized shader-source nodes must point to
/// valid `WGPUShaderSourceWGSL` storage. WGSL string data must be valid for
/// its declared length.
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

        chain = node.next;
    }

    source.unwrap_or_else(|| {
        core::ShaderModuleSource::Invalid(
            "shader module descriptor must contain exactly one shader source".to_owned(),
        )
    })
}
