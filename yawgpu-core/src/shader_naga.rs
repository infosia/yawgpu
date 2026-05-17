#![allow(dead_code)]
// P5.0 intentionally lands reflection helpers before pipeline creation uses
// them. Later Phase-5 slices consume these crate-private APIs.

pub(crate) struct ValidatedWgslModule {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPoint {
    pub name: String,
    pub stage: ReflectedShaderStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTypeScalarClass {
    Float,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReflectedTypeClass {
    pub scalar: ReflectedTypeScalarClass,
    pub components: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedIoLocation {
    pub location: u32,
    pub ty: ReflectedTypeClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPointIo {
    pub entry_point: String,
    pub inputs: Vec<ReflectedIoLocation>,
    pub outputs: Vec<ReflectedIoLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedOverrideKey {
    pub name: Option<String>,
    pub id: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedWorkgroupSize {
    pub entry_point: String,
    pub literal_size: [u32; 3],
    /// Per-dimension override keys for `@workgroup_size(x, y, z)`.
    ///
    /// Naga already stores the literal fallback in `literal_size`; when a
    /// dimension is override-driven, this key lets pipeline validation apply
    /// pipeline constants before enforcing compute limits.
    pub override_keys: [Option<ReflectedOverrideKey>; 3],
    pub workgroup_storage_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedBufferType {
    Uniform,
    Storage,
    ReadOnlyStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTextureSampleUsage {
    Sample,
    Load,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedStorageTextureAccess {
    pub read: bool,
    pub write: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReflectedResourceBindingKind {
    Buffer(ReflectedBufferType),
    Sampler,
    Texture {
        sampled: bool,
        sample_usage: ReflectedTextureSampleUsage,
    },
    StorageTexture {
        format: String,
        access: ReflectedStorageTextureAccess,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedResourceBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: ReflectedResourceBindingKind,
    pub statically_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedFragmentBuiltins {
    pub entry_point: String,
    pub frag_depth: bool,
    pub sample_mask: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedOverride {
    pub name: Option<String>,
    pub id: Option<u16>,
    pub ty: ReflectedTypeClass,
    pub has_default: bool,
}

pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ValidatedWgslModule, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|error| error.to_string())?;
    let capabilities = naga::valid::Capabilities::SHADER_FLOAT16;
    // Enabled capabilities:
    // - SHADER_FLOAT16: Phase-5 overridable-constant validation needs WGSL
    //   `enable f16; override x: f16;` shaders from Dawn.
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities);
    let info = validator
        .validate(&module)
        .map_err(|error| error.to_string())?;
    Ok(ValidatedWgslModule { module, info })
}

impl ValidatedWgslModule {
    pub(crate) fn entry_points(&self) -> Vec<ReflectedEntryPoint> {
        self.module
            .entry_points
            .iter()
            .filter_map(|entry| {
                Some(ReflectedEntryPoint {
                    name: entry.name.clone(),
                    stage: map_shader_stage(entry.stage)?,
                })
            })
            .collect()
    }

    pub(crate) fn compute_workgroup_size(
        &self,
        entry_point: &str,
    ) -> Result<Option<ReflectedWorkgroupSize>, String> {
        let Some(entry) =
            self.module.entry_points.iter().find(|entry| {
                entry.name == entry_point && entry.stage == naga::ShaderStage::Compute
            })
        else {
            return Ok(None);
        };

        let mut override_keys: [Option<ReflectedOverrideKey>; 3] = [None, None, None];
        if let Some(overrides) = entry.workgroup_size_overrides {
            for (index, expression) in overrides.into_iter().enumerate() {
                override_keys[index] = expression
                    .and_then(|expression| override_key_from_expression(&self.module, expression));
            }
        }

        Ok(Some(ReflectedWorkgroupSize {
            entry_point: entry.name.clone(),
            literal_size: entry.workgroup_size,
            override_keys,
            workgroup_storage_size: self.workgroup_storage_size()?,
        }))
    }

    pub(crate) fn entry_point_io(&self) -> Vec<ReflectedEntryPointIo> {
        self.module
            .entry_points
            .iter()
            .filter_map(|entry| {
                let stage = map_shader_stage(entry.stage)?;
                Some(ReflectedEntryPointIo {
                    entry_point: entry.name.clone(),
                    inputs: collect_function_inputs(&self.module, &entry.function, stage),
                    outputs: collect_function_outputs(&self.module, &entry.function, stage),
                })
            })
            .collect()
    }

    pub(crate) fn resource_bindings(&self) -> Vec<ReflectedResourceBinding> {
        self.module
            .global_variables
            .iter()
            .filter_map(|(handle, global)| {
                let binding = global.binding?;
                let kind = resource_binding_kind(&self.module, global, handle)?;
                Some(ReflectedResourceBinding {
                    group: binding.group,
                    binding: binding.binding,
                    kind,
                    statically_used: self
                        .module
                        .entry_points
                        .iter()
                        .enumerate()
                        .any(|(index, _)| !self.info.get_entry_point(index)[handle].is_empty()),
                })
            })
            .collect()
    }

    pub(crate) fn fragment_builtins(&self) -> Vec<ReflectedFragmentBuiltins> {
        self.module
            .entry_points
            .iter()
            .filter(|entry| entry.stage == naga::ShaderStage::Fragment)
            .map(|entry| {
                let mut builtins = ReflectedFragmentBuiltins {
                    entry_point: entry.name.clone(),
                    frag_depth: false,
                    sample_mask: false,
                };
                collect_output_builtins(&self.module, &entry.function, &mut builtins);
                builtins
            })
            .collect()
    }

    pub(crate) fn overrides(&self) -> Vec<ReflectedOverride> {
        self.module
            .overrides
            .iter()
            .filter_map(|(_, override_)| {
                Some(ReflectedOverride {
                    name: override_.name.clone(),
                    id: override_.id,
                    ty: type_class(&self.module, override_.ty)?,
                    has_default: override_.init.is_some(),
                })
            })
            .collect()
    }

    fn workgroup_storage_size(&self) -> Result<u32, String> {
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(self.module.to_ctx())
            .map_err(|error| error.to_string())?;
        Ok(self
            .module
            .global_variables
            .iter()
            .filter(|(_, global)| global.space == naga::AddressSpace::WorkGroup)
            .map(|(_, global)| layouter[global.ty].size)
            .sum())
    }
}

fn map_shader_stage(stage: naga::ShaderStage) -> Option<ReflectedShaderStage> {
    match stage {
        naga::ShaderStage::Vertex => Some(ReflectedShaderStage::Vertex),
        naga::ShaderStage::Fragment => Some(ReflectedShaderStage::Fragment),
        naga::ShaderStage::Compute => Some(ReflectedShaderStage::Compute),
        _ => None,
    }
}

fn override_key_from_expression(
    module: &naga::Module,
    expression: naga::Handle<naga::Expression>,
) -> Option<ReflectedOverrideKey> {
    match module.global_expressions.try_get(expression).ok()? {
        naga::Expression::Override(handle) => {
            let override_ = module.overrides.try_get(*handle).ok()?;
            Some(ReflectedOverrideKey {
                name: override_.name.clone(),
                id: override_.id,
            })
        }
        _ => None,
    }
}

fn collect_function_inputs(
    module: &naga::Module,
    function: &naga::Function,
    stage: ReflectedShaderStage,
) -> Vec<ReflectedIoLocation> {
    if stage == ReflectedShaderStage::Compute {
        return Vec::new();
    }

    let mut locations = Vec::new();
    for argument in &function.arguments {
        collect_binding_locations(
            module,
            argument.ty,
            argument.binding.as_ref(),
            &mut locations,
        );
    }
    locations
}

fn collect_function_outputs(
    module: &naga::Module,
    function: &naga::Function,
    stage: ReflectedShaderStage,
) -> Vec<ReflectedIoLocation> {
    if stage == ReflectedShaderStage::Compute {
        return Vec::new();
    }

    let mut locations = Vec::new();
    if let Some(result) = &function.result {
        collect_binding_locations(module, result.ty, result.binding.as_ref(), &mut locations);
    }
    locations
}

fn collect_binding_locations(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
    locations: &mut Vec<ReflectedIoLocation>,
) {
    if let Some(naga::Binding::Location { location, .. }) = binding {
        if let Some(ty) = type_class(module, ty) {
            locations.push(ReflectedIoLocation {
                location: *location,
                ty,
            });
        }
        return;
    }

    let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        return;
    };
    for member in members {
        if let Some(naga::Binding::Location { location, .. }) = member.binding.as_ref() {
            if let Some(ty) = type_class(module, member.ty) {
                locations.push(ReflectedIoLocation {
                    location: *location,
                    ty,
                });
            }
        }
    }
}

fn collect_output_builtins(
    module: &naga::Module,
    function: &naga::Function,
    builtins: &mut ReflectedFragmentBuiltins,
) {
    let Some(result) = &function.result else {
        return;
    };
    collect_binding_builtins(module, result.ty, result.binding.as_ref(), builtins);
}

fn collect_binding_builtins(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&naga::Binding>,
    builtins: &mut ReflectedFragmentBuiltins,
) {
    if let Some(naga::Binding::BuiltIn(builtin)) = binding {
        mark_fragment_builtin(*builtin, builtins);
        return;
    }

    let naga::TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        return;
    };
    for member in members {
        if let Some(naga::Binding::BuiltIn(builtin)) = member.binding.as_ref() {
            mark_fragment_builtin(*builtin, builtins);
        }
    }
}

fn mark_fragment_builtin(builtin: naga::BuiltIn, builtins: &mut ReflectedFragmentBuiltins) {
    match builtin {
        naga::BuiltIn::FragDepth => builtins.frag_depth = true,
        naga::BuiltIn::SampleMask => builtins.sample_mask = true,
        _ => {}
    }
}

fn type_class(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Option<ReflectedTypeClass> {
    match &module.types.get_handle(ty).ok()?.inner {
        naga::TypeInner::Scalar(scalar) => scalar_class(*scalar).map(|scalar| ReflectedTypeClass {
            scalar,
            components: 1,
        }),
        naga::TypeInner::Vector { size, scalar } => {
            scalar_class(*scalar).map(|scalar| ReflectedTypeClass {
                scalar,
                components: vector_components(*size),
            })
        }
        _ => None,
    }
}

fn scalar_class(scalar: naga::Scalar) -> Option<ReflectedTypeScalarClass> {
    match scalar.kind {
        naga::ScalarKind::Float => Some(ReflectedTypeScalarClass::Float),
        naga::ScalarKind::Sint => Some(ReflectedTypeScalarClass::Sint),
        naga::ScalarKind::Uint => Some(ReflectedTypeScalarClass::Uint),
        _ => None,
    }
}

fn vector_components(size: naga::VectorSize) -> u8 {
    match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    }
}

fn resource_binding_kind(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    handle: naga::Handle<naga::GlobalVariable>,
) -> Option<ReflectedResourceBindingKind> {
    match global.space {
        naga::AddressSpace::Uniform => Some(ReflectedResourceBindingKind::Buffer(
            ReflectedBufferType::Uniform,
        )),
        naga::AddressSpace::Storage { access } => {
            let ty = if access.contains(naga::StorageAccess::STORE) {
                ReflectedBufferType::Storage
            } else {
                ReflectedBufferType::ReadOnlyStorage
            };
            Some(ReflectedResourceBindingKind::Buffer(ty))
        }
        naga::AddressSpace::Handle => match &module.types.get_handle(global.ty).ok()?.inner {
            naga::TypeInner::Sampler { .. } => Some(ReflectedResourceBindingKind::Sampler),
            naga::TypeInner::Image { class, .. } => match class {
                naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. } => {
                    Some(ReflectedResourceBindingKind::Texture {
                        sampled: matches!(class, naga::ImageClass::Sampled { .. }),
                        sample_usage: sampled_texture_usage(module, handle),
                    })
                }
                naga::ImageClass::Storage { format, access } => {
                    Some(ReflectedResourceBindingKind::StorageTexture {
                        format: format!("{format:?}"),
                        access: ReflectedStorageTextureAccess {
                            read: access.contains(naga::StorageAccess::LOAD),
                            write: access.contains(naga::StorageAccess::STORE),
                        },
                    })
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

fn sampled_texture_usage(
    module: &naga::Module,
    handle: naga::Handle<naga::GlobalVariable>,
) -> ReflectedTextureSampleUsage {
    if module.entry_points.iter().any(|entry| {
        entry.function.expressions.iter().any(|(_, expression)| {
            matches!(
                expression,
                naga::Expression::ImageSample { image, .. }
                    if expression_global(&entry.function, *image) == Some(handle)
            )
        })
    }) {
        ReflectedTextureSampleUsage::Sample
    } else {
        ReflectedTextureSampleUsage::Load
    }
}

fn expression_global(
    function: &naga::Function,
    expression: naga::Handle<naga::Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    match function.expressions.try_get(expression).ok()? {
        naga::Expression::GlobalVariable(handle) => Some(*handle),
        naga::Expression::Access { base, .. }
        | naga::Expression::AccessIndex { base, .. }
        | naga::Expression::Load { pointer: base } => expression_global(function, *base),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_and_validate_wgsl, ReflectedBufferType, ReflectedResourceBindingKind,
        ReflectedShaderStage, ReflectedTextureSampleUsage, ReflectedTypeScalarClass,
    };

    #[test]
    fn parses_and_validates_trivial_wgsl() {
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        assert!(parse_and_validate_wgsl(source).is_ok());
    }

    #[test]
    fn rejects_invalid_wgsl() {
        assert!(parse_and_validate_wgsl("not wgsl @@@").is_err());
    }

    #[test]
    fn reflects_entry_points() {
        let module = parse_and_validate_wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() {}
             @compute @workgroup_size(1) fn cs() {}",
        )
        .unwrap();

        let entry_points = module.entry_points();
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "vs" && entry.stage == ReflectedShaderStage::Vertex));
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "fs" && entry.stage == ReflectedShaderStage::Fragment));
        assert!(entry_points
            .iter()
            .any(|entry| entry.name == "cs" && entry.stage == ReflectedShaderStage::Compute));
    }

    #[test]
    fn reflects_compute_workgroup_size_and_storage() {
        let module = parse_and_validate_wgsl(
            "var<workgroup> scratch: array<u32, 4>;
             @compute @workgroup_size(8, 4, 1) fn cs() {
                 scratch[0] = 1u;
             }",
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [8, 4, 1]);
        assert_eq!(reflected.override_keys, [None, None, None]);
        assert_eq!(reflected.workgroup_storage_size, 16);
    }

    #[test]
    fn reflects_override_driven_workgroup_size() {
        let module = parse_and_validate_wgsl(
            "override wg_x: u32 = 8u;
             @compute @workgroup_size(wg_x, 1, 1) fn cs() {}",
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [1, 1, 1]);
        assert_eq!(
            reflected.override_keys[0].as_ref().unwrap().name.as_deref(),
            Some("wg_x")
        );
    }

    #[test]
    fn reflects_vertex_fragment_io() {
        let module = parse_and_validate_wgsl(
            "struct VsOut {
                 @builtin(position) pos: vec4<f32>,
                 @location(1) color: vec4<f32>,
             }
             @vertex fn vs(@location(0) a: vec3<f32>) -> VsOut {
                 return VsOut(vec4<f32>(0.0), vec4<f32>(a, 1.0));
             }
             @fragment fn fs(@location(1) color: vec4<f32>) -> @location(0) vec4<f32> {
                 return color;
             }",
        )
        .unwrap();

        let io = module.entry_point_io();
        let vs = io.iter().find(|entry| entry.entry_point == "vs").unwrap();
        assert_eq!(vs.inputs[0].location, 0);
        assert_eq!(vs.inputs[0].ty.scalar, ReflectedTypeScalarClass::Float);
        assert_eq!(vs.inputs[0].ty.components, 3);
        assert_eq!(vs.outputs[0].location, 1);

        let fs = io.iter().find(|entry| entry.entry_point == "fs").unwrap();
        assert_eq!(fs.inputs[0].location, 1);
        assert_eq!(fs.outputs[0].location, 0);
    }

    #[test]
    fn reflects_resource_bindings_and_static_use() {
        let module = parse_and_validate_wgsl(
            "struct U { value: vec4<f32> }
             @group(0) @binding(0) var<uniform> u: U;
             @group(0) @binding(1) var samp: sampler;
             @group(0) @binding(2) var tex: texture_2d<f32>;
             @group(0) @binding(3) var unused_tex: texture_2d<f32>;
             @fragment fn fs() -> @location(0) vec4<f32> {
                 return textureSample(tex, samp, vec2<f32>(0.5)) + u.value;
             }",
        )
        .unwrap();

        let bindings = module.resource_bindings();
        let uniform = bindings
            .iter()
            .find(|binding| binding.binding == 0)
            .unwrap();
        assert_eq!(
            uniform.kind,
            ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Uniform)
        );
        assert!(uniform.statically_used);

        let texture = bindings
            .iter()
            .find(|binding| binding.binding == 2)
            .unwrap();
        assert_eq!(
            texture.kind,
            ReflectedResourceBindingKind::Texture {
                sampled: true,
                sample_usage: ReflectedTextureSampleUsage::Sample
            }
        );
        assert!(texture.statically_used);

        let unused = bindings
            .iter()
            .find(|binding| binding.binding == 3)
            .unwrap();
        assert!(!unused.statically_used);
    }

    #[test]
    fn reflects_fragment_builtin_outputs() {
        let module = parse_and_validate_wgsl(
            "struct Out {
                 @builtin(frag_depth) depth: f32,
                 @builtin(sample_mask) mask: u32,
             }
             @fragment fn fs() -> Out {
                 return Out(0.5, 1u);
             }",
        )
        .unwrap();

        let builtins = module.fragment_builtins();
        assert!(builtins[0].frag_depth);
        assert!(builtins[0].sample_mask);
    }

    #[test]
    fn reflects_overrides_and_accepts_f16_override() {
        let module = parse_and_validate_wgsl(
            "enable f16;
             override half_value: f16;
             @id(7) override int_value: i32 = 3;
             @compute @workgroup_size(1) fn cs() {}",
        )
        .unwrap();

        let overrides = module.overrides();
        let half = overrides
            .iter()
            .find(|override_| override_.name.as_deref() == Some("half_value"))
            .unwrap();
        assert_eq!(half.ty.scalar, ReflectedTypeScalarClass::Float);
        assert!(!half.has_default);

        let int = overrides
            .iter()
            .find(|override_| override_.id == Some(7))
            .unwrap();
        assert_eq!(int.ty.scalar, ReflectedTypeScalarClass::Sint);
        assert!(int.has_default);
    }
}
