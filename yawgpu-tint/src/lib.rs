//! Rust bindings for Dawn's Tint shader compiler, driven through a small C++
//! shim (`shim/tint_shim.cpp`).
//!
//! When built without `YAWGPU_DAWN_DIR` set, the Tint backend is not linked
//! ([`HAVE_TINT`] is `false`) and the public API returns an unavailable error.
#![warn(missing_docs)]

/// Whether this build links the Tint backend.
pub const HAVE_TINT: bool = cfg!(have_tint);

#[cfg(have_tint)]
mod imp {
    use std::ffi::{CStr, CString};
    use std::marker::PhantomData;
    use std::os::raw::{c_char, c_void};
    use std::ptr;
    use std::slice;

    #[repr(C)]
    struct RawProgram(c_void);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawEntryPoint {
        name: *const c_char,
        stage: u8,
        has_workgroup_size: bool,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
        frag_depth_used: bool,
        sample_mask_used: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawResourceBinding {
        group: u32,
        binding: u32,
        resource_type: u8,
        dim: u8,
        sampled_kind: u8,
        sampler_type: u8,
        texel_format: u8,
        size: u64,
        has_array_size: bool,
        array_size: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawOverride {
        name: *const c_char,
        id: u16,
        type_class: u8,
        has_default: bool,
        default_value: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawBindingRemap {
        group: u32,
        binding: u32,
        dst_group: u32,
        dst_binding: u32,
    }

    #[repr(C)]
    struct RawBindings {
        uniform: *const RawBindingRemap,
        n_uniform: usize,
        storage: *const RawBindingRemap,
        n_storage: usize,
        texture: *const RawBindingRemap,
        n_texture: usize,
        storage_texture: *const RawBindingRemap,
        n_storage_texture: usize,
        sampler: *const RawBindingRemap,
        n_sampler: usize,
    }

    #[repr(C)]
    struct RawOverrideValue {
        name: *const c_char,
        value: f64,
    }

    #[repr(C)]
    struct RawMslOutput {
        msl: *mut c_char,
        needs_storage_buffer_sizes: bool,
    }

    extern "C" {
        fn yawgpu_tint_initialize();
        fn yawgpu_tint_program_create(
            wgsl: *const c_char,
            wgsl_len: usize,
            shader_f16: bool,
            err: *mut *mut c_char,
        ) -> *mut RawProgram;
        fn yawgpu_tint_program_destroy(program: *mut RawProgram);
        fn yawgpu_tint_entry_point_count(program: *const RawProgram) -> usize;
        fn yawgpu_tint_entry_point_get(
            program: *const RawProgram,
            i: usize,
            out: *mut RawEntryPoint,
        ) -> bool;
        fn yawgpu_tint_resource_binding_count(
            program: *const RawProgram,
            ep: *const c_char,
        ) -> usize;
        fn yawgpu_tint_resource_binding_get(
            program: *const RawProgram,
            ep: *const c_char,
            i: usize,
            out: *mut RawResourceBinding,
        ) -> bool;
        fn yawgpu_tint_override_count(program: *const RawProgram) -> usize;
        fn yawgpu_tint_override_get(
            program: *const RawProgram,
            i: usize,
            out: *mut RawOverride,
        ) -> bool;
        fn yawgpu_tint_generate_msl(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings,
            ov: *const RawOverrideValue,
            n_ov: usize,
            disable_robustness: bool,
            out: *mut RawMslOutput,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_generate_spirv(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings,
            ov: *const RawOverrideValue,
            n_ov: usize,
            disable_robustness: bool,
            words_out: *mut *mut u32,
            n_words_out: *mut usize,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_generate_glsl(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings,
            ov: *const RawOverrideValue,
            n_ov: usize,
            glsl_out: *mut *mut c_char,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_string_free(s: *mut c_char);
        fn yawgpu_tint_u32_free(s: *mut u32);
    }

    fn take_error(err: *mut c_char) -> String {
        if err.is_null() {
            return "tint: unknown error".to_owned();
        }
        // SAFETY: error strings are allocated by the shim and NUL-terminated.
        unsafe {
            let msg = CStr::from_ptr(err).to_string_lossy().into_owned();
            yawgpu_tint_string_free(err);
            msg
        }
    }

    fn cstring(name: &str, what: &str) -> Result<CString, String> {
        CString::new(name).map_err(|_| format!("{what} contains an interior NUL"))
    }

    fn raw_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: reflection strings are borrowed from the live program handle.
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Initializes the Tint runtime.
    pub fn initialize() {
        // SAFETY: the shim guards Tint initialization.
        unsafe { yawgpu_tint_initialize() }
    }

    /// A parsed and validated Tint program.
    #[derive(Debug)]
    pub struct Program {
        raw: *mut RawProgram,
    }

    impl Program {
        /// Parses WGSL into a Tint program.
        pub fn parse(wgsl: &str, shader_f16: bool) -> Result<Self, String> {
            initialize();
            let mut err = ptr::null_mut();
            // SAFETY: `wgsl` is valid for the call and the shim copies it using the length.
            let raw = unsafe {
                yawgpu_tint_program_create(
                    wgsl.as_ptr().cast::<c_char>(),
                    wgsl.len(),
                    shader_f16,
                    &mut err,
                )
            };
            if raw.is_null() {
                Err(take_error(err))
            } else {
                Ok(Self { raw })
            }
        }

        /// Returns reflected entry points.
        pub fn entry_points(&self) -> Result<Vec<EntryPoint>, String> {
            // SAFETY: `self.raw` is owned by this Program.
            let count = unsafe { yawgpu_tint_entry_point_count(self.raw) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawEntryPoint {
                    name: ptr::null(),
                    stage: 0,
                    has_workgroup_size: false,
                    wg_x: 0,
                    wg_y: 0,
                    wg_z: 0,
                    frag_depth_used: false,
                    sample_mask_used: false,
                };
                // SAFETY: `raw` points to valid writable memory.
                let ok = unsafe { yawgpu_tint_entry_point_get(self.raw, i, &mut raw) };
                if !ok {
                    return Err("tint: failed to read entry point reflection".to_owned());
                }
                out.push(EntryPoint::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Returns resource bindings used by `entry_point`.
        pub fn resource_bindings(&self, entry_point: &str) -> Result<Vec<ResourceBinding>, String> {
            let ep = cstring(entry_point, "entry point")?;
            // SAFETY: pointers are valid for the duration of the call.
            let count = unsafe { yawgpu_tint_resource_binding_count(self.raw, ep.as_ptr()) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawResourceBinding {
                    group: 0,
                    binding: 0,
                    resource_type: 0,
                    dim: 0,
                    sampled_kind: 0,
                    sampler_type: 0,
                    texel_format: 0,
                    size: 0,
                    has_array_size: false,
                    array_size: 0,
                };
                // SAFETY: pointers are valid for the duration of the call.
                let ok =
                    unsafe { yawgpu_tint_resource_binding_get(self.raw, ep.as_ptr(), i, &mut raw) };
                if !ok {
                    return Err("tint: failed to read resource binding reflection".to_owned());
                }
                out.push(ResourceBinding::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Returns module override declarations.
        pub fn overrides(&self) -> Result<Vec<Override>, String> {
            // SAFETY: `self.raw` is owned by this Program.
            let count = unsafe { yawgpu_tint_override_count(self.raw) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawOverride {
                    name: ptr::null(),
                    id: 0,
                    type_class: 0,
                    has_default: false,
                    default_value: 0.0,
                };
                // SAFETY: `raw` points to valid writable memory.
                let ok = unsafe { yawgpu_tint_override_get(self.raw, i, &mut raw) };
                if !ok {
                    return Err("tint: failed to read override reflection".to_owned());
                }
                out.push(Override::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Generates MSL for `entry_point`.
        pub fn generate_msl(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
            robust: bool,
        ) -> Result<MslOutput, String> {
            let ep = cstring(entry_point, "entry point")?;
            let raw_bindings_owned = bindings.as_raw();
            let raw_bindings = raw_bindings_owned.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut out = RawMslOutput {
                msl: ptr::null_mut(),
                needs_storage_buffer_sizes: false,
            };
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_msl(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    !robust,
                    &mut out,
                    &mut err,
                )
            };
            if !ok {
                return Err(take_error(err));
            }
            if out.msl.is_null() {
                return Err("tint: MSL generator returned NULL output".to_owned());
            }
            // SAFETY: `out.msl` is owned by Rust after success.
            let msl = unsafe {
                let s = CStr::from_ptr(out.msl).to_string_lossy().into_owned();
                yawgpu_tint_string_free(out.msl);
                s
            };
            Ok(MslOutput {
                source: msl,
                needs_storage_buffer_sizes: out.needs_storage_buffer_sizes,
            })
        }

        /// Generates SPIR-V words for `entry_point`.
        pub fn generate_spirv(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
            robust: bool,
        ) -> Result<Vec<u32>, String> {
            let ep = cstring(entry_point, "entry point")?;
            let raw_bindings_owned = bindings.as_raw();
            let raw_bindings = raw_bindings_owned.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut words = ptr::null_mut();
            let mut len = 0usize;
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_spirv(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    !robust,
                    &mut words,
                    &mut len,
                    &mut err,
                )
            };
            if !ok {
                return Err(take_error(err));
            }
            // A successful generate of valid SPIR-V always yields a non-empty
            // module (≥5 header words), but guard the null/empty case anyway —
            // `slice::from_raw_parts` requires a non-null pointer even for len 0.
            if words.is_null() {
                return Ok(Vec::new());
            }
            // SAFETY: the shim returns `len` initialized words allocated by malloc.
            let out = unsafe {
                let data = slice::from_raw_parts(words, len).to_vec();
                yawgpu_tint_u32_free(words);
                data
            };
            Ok(out)
        }

        /// Generates GLSL ES 3.1 for `entry_point`.
        pub fn generate_glsl(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
        ) -> Result<String, String> {
            let ep = cstring(entry_point, "entry point")?;
            let raw_bindings_owned = bindings.as_raw();
            let raw_bindings = raw_bindings_owned.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut glsl = ptr::null_mut();
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_glsl(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    &mut glsl,
                    &mut err,
                )
            };
            if !ok {
                return Err(take_error(err));
            }
            if glsl.is_null() {
                return Err("tint: GLSL generator returned NULL output".to_owned());
            }
            // SAFETY: `glsl` is owned by Rust after success.
            let out = unsafe {
                let s = CStr::from_ptr(glsl).to_string_lossy().into_owned();
                yawgpu_tint_string_free(glsl);
                s
            };
            Ok(out)
        }
    }

    impl Drop for Program {
        fn drop(&mut self) {
            // SAFETY: `raw` was returned by the shim and is owned by this Program.
            unsafe { yawgpu_tint_program_destroy(self.raw) }
        }
    }

    /// MSL generator output.
    pub struct MslOutput {
        /// Generated MSL source.
        pub source: String,
        /// Whether the generated MSL needs a storage-buffer-size table.
        pub needs_storage_buffer_sizes: bool,
    }

    /// A reflected entry point.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EntryPoint {
        /// Entry point name.
        pub name: String,
        /// Pipeline stage.
        pub stage: PipelineStage,
        /// Workgroup size when Tint can resolve it during reflection.
        pub workgroup_size: Option<[u32; 3]>,
        /// Whether the fragment entry point writes frag_depth.
        pub frag_depth_used: bool,
        /// Whether the entry point uses sample_mask.
        pub sample_mask_used: bool,
    }

    impl EntryPoint {
        fn try_from_raw(raw: RawEntryPoint) -> Result<Self, String> {
            Ok(Self {
                name: raw_string(raw.name),
                stage: PipelineStage::try_from_raw(raw.stage)?,
                workgroup_size: raw
                    .has_workgroup_size
                    .then_some([raw.wg_x, raw.wg_y, raw.wg_z]),
                frag_depth_used: raw.frag_depth_used,
                sample_mask_used: raw.sample_mask_used,
            })
        }
    }

    /// Pipeline stage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PipelineStage {
        /// Vertex shader stage.
        Vertex,
        /// Fragment shader stage.
        Fragment,
        /// Compute shader stage.
        Compute,
    }

    impl PipelineStage {
        fn try_from_raw(raw: u8) -> Result<Self, String> {
            match raw {
                0 => Ok(Self::Vertex),
                1 => Ok(Self::Fragment),
                2 => Ok(Self::Compute),
                _ => Err(format!("tint: unknown pipeline stage {raw}")),
            }
        }
    }

    /// A reflected resource binding.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ResourceBinding {
        /// Bind group.
        pub group: u32,
        /// Binding number within the bind group.
        pub binding: u32,
        /// Resource class.
        pub resource_type: ResourceType,
        /// Texture dimension.
        pub dim: TextureDimension,
        /// Sampled texture component kind.
        pub sampled_kind: SampledKind,
        /// Sampler kind.
        pub sampler_type: SamplerType,
        /// Storage texture texel format.
        pub texel_format: TexelFormat,
        /// Static byte size reported by Tint, or zero when not applicable.
        pub size: u64,
        /// Binding array size when present.
        pub array_size: Option<u32>,
    }

    impl ResourceBinding {
        fn try_from_raw(raw: RawResourceBinding) -> Result<Self, String> {
            Ok(Self {
                group: raw.group,
                binding: raw.binding,
                resource_type: ResourceType::try_from_raw(raw.resource_type)?,
                dim: TextureDimension::try_from_raw(raw.dim)?,
                sampled_kind: SampledKind::try_from_raw(raw.sampled_kind)?,
                sampler_type: SamplerType::try_from_raw(raw.sampler_type)?,
                texel_format: TexelFormat::try_from_raw(raw.texel_format)?,
                size: raw.size,
                array_size: raw.has_array_size.then_some(raw.array_size),
            })
        }
    }

    macro_rules! raw_enum {
        (
            $(#[$meta:meta])*
            pub enum $name:ident { $($(#[$vmeta:meta])* $variant:ident = $value:literal,)* }
        ) => {
            $(#[$meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum $name {
                $($(#[$vmeta])* $variant,)*
            }

            impl $name {
                fn try_from_raw(raw: u8) -> Result<Self, String> {
                    match raw {
                        $($value => Ok(Self::$variant),)*
                        _ => Err(format!("tint: unknown {} value {}", stringify!($name), raw)),
                    }
                }
            }
        };
    }

    raw_enum! {
        /// Tint resource binding class.
        pub enum ResourceType {
            /// Uniform buffer.
            UniformBuffer = 0,
            /// Writable storage buffer.
            StorageBuffer = 1,
            /// Read-only storage buffer.
            ReadOnlyStorageBuffer = 2,
            /// Sampler.
            Sampler = 3,
            /// Sampled texture.
            SampledTexture = 4,
            /// Multisampled texture.
            MultisampledTexture = 5,
            /// Write-only storage texture.
            WriteOnlyStorageTexture = 6,
            /// Read-only storage texture.
            ReadOnlyStorageTexture = 7,
            /// Read-write storage texture.
            ReadWriteStorageTexture = 8,
            /// Depth texture.
            DepthTexture = 9,
            /// Depth multisampled texture.
            DepthMultisampledTexture = 10,
            /// External texture.
            ExternalTexture = 11,
            /// Read-only texel buffer.
            ReadOnlyTexelBuffer = 12,
            /// Read-write texel buffer.
            ReadWriteTexelBuffer = 13,
            /// Input attachment.
            InputAttachment = 14,
        }
    }

    raw_enum! {
        /// Texture dimension.
        pub enum TextureDimension {
            /// One-dimensional texture.
            D1 = 0,
            /// Two-dimensional texture.
            D2 = 1,
            /// Two-dimensional array texture.
            D2Array = 2,
            /// Three-dimensional texture.
            D3 = 3,
            /// Cube texture.
            Cube = 4,
            /// Cube array texture.
            CubeArray = 5,
            /// No texture dimension.
            None = 6,
        }
    }

    raw_enum! {
        /// Sampled texture component kind.
        pub enum SampledKind {
            /// Float component.
            Float = 0,
            /// Unsigned integer component.
            UInt = 1,
            /// Signed integer component.
            SInt = 2,
            /// Filterable float component.
            Filterable = 3,
            /// Unfilterable float component.
            Unfilterable = 4,
            /// Unknown filterability.
            UnknownFilterable = 5,
        }
    }

    raw_enum! {
        /// Sampler binding kind.
        pub enum SamplerType {
            /// Comparison sampler.
            Comparison = 0,
            /// Filtering sampler.
            Filtering = 1,
            /// Non-filtering sampler.
            NonFiltering = 2,
            /// Unknown filtering mode.
            UnknownFiltering = 3,
        }
    }

    raw_enum! {
        /// Storage texture texel format.
        pub enum TexelFormat {
            /// r8snorm.
            R8Snorm = 0,
            /// r8uint.
            R8Uint = 1,
            /// r8sint.
            R8Sint = 2,
            /// rg8unorm.
            Rg8Unorm = 3,
            /// rg8snorm.
            Rg8Snorm = 4,
            /// rg8uint.
            Rg8Uint = 5,
            /// rg8sint.
            Rg8Sint = 6,
            /// r16unorm.
            R16Unorm = 7,
            /// r16snorm.
            R16Snorm = 8,
            /// r16uint.
            R16Uint = 9,
            /// r16sint.
            R16Sint = 10,
            /// r16float.
            R16Float = 11,
            /// rg16unorm.
            Rg16Unorm = 12,
            /// rg16snorm.
            Rg16Snorm = 13,
            /// rg16uint.
            Rg16Uint = 14,
            /// rg16sint.
            Rg16Sint = 15,
            /// rg16float.
            Rg16Float = 16,
            /// bgra8unorm.
            Bgra8Unorm = 17,
            /// rgba8unorm.
            Rgba8Unorm = 18,
            /// rgba8snorm.
            Rgba8Snorm = 19,
            /// rgba8uint.
            Rgba8Uint = 20,
            /// rgba8sint.
            Rgba8Sint = 21,
            /// rgba16unorm.
            Rgba16Unorm = 22,
            /// rgba16snorm.
            Rgba16Snorm = 23,
            /// rgba16uint.
            Rgba16Uint = 24,
            /// rgba16sint.
            Rgba16Sint = 25,
            /// rgba16float.
            Rgba16Float = 26,
            /// r32uint.
            R32Uint = 27,
            /// r32sint.
            R32Sint = 28,
            /// r32float.
            R32Float = 29,
            /// rg32uint.
            Rg32Uint = 30,
            /// rg32sint.
            Rg32Sint = 31,
            /// rg32float.
            Rg32Float = 32,
            /// rgba32uint.
            Rgba32Uint = 33,
            /// rgba32sint.
            Rgba32Sint = 34,
            /// rgba32float.
            Rgba32Float = 35,
            /// r8unorm.
            R8Unorm = 36,
            /// rgb10a2uint.
            Rgb10A2Uint = 37,
            /// rgb10a2unorm.
            Rgb10A2Unorm = 38,
            /// rg11b10ufloat.
            Rg11B10Ufloat = 39,
            /// No texel format.
            None = 40,
        }
    }

    /// A reflected pipeline override.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Override {
        /// Override name.
        pub name: String,
        /// Numeric override identifier.
        pub id: u16,
        /// Override scalar type.
        pub type_class: OverrideType,
        /// Whether the override has a default initializer.
        pub has_default: bool,
        /// Reflected default value when Tint exposes it.
        pub default_value: f64,
    }

    impl Override {
        fn try_from_raw(raw: RawOverride) -> Result<Self, String> {
            Ok(Self {
                name: raw_string(raw.name),
                id: raw.id,
                type_class: OverrideType::try_from_raw(raw.type_class)?,
                has_default: raw.has_default,
                default_value: raw.default_value,
            })
        }
    }

    raw_enum! {
        /// Override scalar type.
        pub enum OverrideType {
            /// Boolean override.
            Bool = 0,
            /// f32 override.
            Float32 = 1,
            /// u32 override.
            Uint32 = 2,
            /// i32 override.
            Int32 = 3,
            /// f16 override.
            Float16 = 4,
        }
    }

    /// A single binding remap from source group/binding to destination group/binding.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BindingRemap {
        /// Source bind group.
        pub group: u32,
        /// Source binding.
        pub binding: u32,
        /// Destination bind group.
        pub dst_group: u32,
        /// Destination binding.
        pub dst_binding: u32,
    }

    impl BindingRemap {
        fn as_raw(self) -> RawBindingRemap {
            RawBindingRemap {
                group: self.group,
                binding: self.binding,
                dst_group: self.dst_group,
                dst_binding: self.dst_binding,
            }
        }
    }

    /// Resource binding remap sets grouped by resource class.
    #[derive(Debug, Clone, Default)]
    pub struct Bindings {
        /// Uniform buffer binding remaps.
        pub uniform: Vec<BindingRemap>,
        /// Storage buffer binding remaps.
        pub storage: Vec<BindingRemap>,
        /// Sampled/depth/external texture binding remaps.
        pub texture: Vec<BindingRemap>,
        /// Storage texture binding remaps.
        pub storage_texture: Vec<BindingRemap>,
        /// Sampler binding remaps.
        pub sampler: Vec<BindingRemap>,
    }

    struct RawBindingsOwned<'a> {
        uniform: Vec<RawBindingRemap>,
        storage: Vec<RawBindingRemap>,
        texture: Vec<RawBindingRemap>,
        storage_texture: Vec<RawBindingRemap>,
        sampler: Vec<RawBindingRemap>,
        _marker: PhantomData<&'a Bindings>,
    }

    impl Bindings {
        fn as_raw(&self) -> RawBindingsOwned<'_> {
            RawBindingsOwned {
                uniform: self
                    .uniform
                    .iter()
                    .copied()
                    .map(BindingRemap::as_raw)
                    .collect(),
                storage: self
                    .storage
                    .iter()
                    .copied()
                    .map(BindingRemap::as_raw)
                    .collect(),
                texture: self
                    .texture
                    .iter()
                    .copied()
                    .map(BindingRemap::as_raw)
                    .collect(),
                storage_texture: self
                    .storage_texture
                    .iter()
                    .copied()
                    .map(BindingRemap::as_raw)
                    .collect(),
                sampler: self
                    .sampler
                    .iter()
                    .copied()
                    .map(BindingRemap::as_raw)
                    .collect(),
                _marker: PhantomData,
            }
        }
    }

    impl RawBindingsOwned<'_> {
        fn as_raw(&self) -> RawBindings {
            RawBindings {
                uniform: self.uniform.as_ptr(),
                n_uniform: self.uniform.len(),
                storage: self.storage.as_ptr(),
                n_storage: self.storage.len(),
                texture: self.texture.as_ptr(),
                n_texture: self.texture.len(),
                storage_texture: self.storage_texture.as_ptr(),
                n_storage_texture: self.storage_texture.len(),
                sampler: self.sampler.as_ptr(),
                n_sampler: self.sampler.len(),
            }
        }
    }

    /// Pipeline override substitution value.
    #[derive(Debug, Clone, PartialEq)]
    pub struct OverrideValue {
        /// Override name or numeric ID encoded as decimal text.
        pub name: String,
        /// Value to substitute, converted by Tint to the declared override type.
        pub value: f64,
    }

    struct RawOverrideValues {
        names: Vec<CString>,
        values: Vec<RawOverrideValue>,
    }

    impl RawOverrideValues {
        fn new(overrides: &[OverrideValue]) -> Result<Self, String> {
            let names: Vec<CString> = overrides
                .iter()
                .map(|ov| cstring(&ov.name, "override name"))
                .collect::<Result<_, _>>()?;
            let values = names
                .iter()
                .zip(overrides)
                .map(|(name, ov)| RawOverrideValue {
                    name: name.as_ptr(),
                    value: ov.value,
                })
                .collect();
            Ok(Self { names, values })
        }

        fn as_ptr(&self) -> *const RawOverrideValue {
            let _keep_names_alive = &self.names;
            self.values.as_ptr()
        }

        fn len(&self) -> usize {
            self.values.len()
        }
    }

    /// Compiles WGSL to MSL using the default parse and binding behavior.
    pub fn wgsl_to_msl(wgsl: &str, entry_point: &str) -> Result<String, String> {
        let program = Program::parse(wgsl, false)?;
        Ok(program
            .generate_msl(entry_point, &Bindings::default(), &[], true)?
            .source)
    }
}

#[cfg(not(have_tint))]
mod imp {
    const UNAVAILABLE: &str = "yawgpu-tint was built without Tint (YAWGPU_DAWN_DIR unset)";

    /// Initializes the Tint runtime.
    pub fn initialize() {}

    /// A parsed and validated Tint program.
    #[derive(Debug)]
    pub struct Program;

    impl Program {
        /// Parses WGSL into a Tint program.
        pub fn parse(_wgsl: &str, _shader_f16: bool) -> Result<Self, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Returns reflected entry points.
        pub fn entry_points(&self) -> Result<Vec<EntryPoint>, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Returns resource bindings used by `entry_point`.
        pub fn resource_bindings(
            &self,
            _entry_point: &str,
        ) -> Result<Vec<ResourceBinding>, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Returns module override declarations.
        pub fn overrides(&self) -> Result<Vec<Override>, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Generates MSL for `entry_point`.
        pub fn generate_msl(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
            _robust: bool,
        ) -> Result<MslOutput, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Generates SPIR-V words for `entry_point`.
        pub fn generate_spirv(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
            _robust: bool,
        ) -> Result<Vec<u32>, String> {
            Err(UNAVAILABLE.to_owned())
        }

        /// Generates GLSL ES 3.1 for `entry_point`.
        pub fn generate_glsl(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
        ) -> Result<String, String> {
            Err(UNAVAILABLE.to_owned())
        }
    }

    /// MSL generator output.
    pub struct MslOutput {
        /// Generated MSL source.
        pub source: String,
        /// Whether the generated MSL needs a storage-buffer-size table.
        pub needs_storage_buffer_sizes: bool,
    }

    /// A reflected entry point.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EntryPoint {
        /// Entry point name.
        pub name: String,
        /// Pipeline stage.
        pub stage: PipelineStage,
        /// Workgroup size when Tint can resolve it during reflection.
        pub workgroup_size: Option<[u32; 3]>,
        /// Whether the fragment entry point writes frag_depth.
        pub frag_depth_used: bool,
        /// Whether the entry point uses sample_mask.
        pub sample_mask_used: bool,
    }

    /// Pipeline stage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PipelineStage {
        /// Vertex shader stage.
        Vertex,
        /// Fragment shader stage.
        Fragment,
        /// Compute shader stage.
        Compute,
    }

    /// A reflected resource binding.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ResourceBinding {
        /// Bind group.
        pub group: u32,
        /// Binding number within the bind group.
        pub binding: u32,
        /// Resource class.
        pub resource_type: ResourceType,
        /// Texture dimension.
        pub dim: TextureDimension,
        /// Sampled texture component kind.
        pub sampled_kind: SampledKind,
        /// Sampler kind.
        pub sampler_type: SamplerType,
        /// Storage texture texel format.
        pub texel_format: TexelFormat,
        /// Static byte size reported by Tint, or zero when not applicable.
        pub size: u64,
        /// Binding array size when present.
        pub array_size: Option<u32>,
    }

    /// Tint resource binding class.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ResourceType {
        /// Uniform buffer.
        UniformBuffer,
        /// Writable storage buffer.
        StorageBuffer,
        /// Read-only storage buffer.
        ReadOnlyStorageBuffer,
        /// Sampler.
        Sampler,
        /// Sampled texture.
        SampledTexture,
        /// Multisampled texture.
        MultisampledTexture,
        /// Write-only storage texture.
        WriteOnlyStorageTexture,
        /// Read-only storage texture.
        ReadOnlyStorageTexture,
        /// Read-write storage texture.
        ReadWriteStorageTexture,
        /// Depth texture.
        DepthTexture,
        /// Depth multisampled texture.
        DepthMultisampledTexture,
        /// External texture.
        ExternalTexture,
        /// Read-only texel buffer.
        ReadOnlyTexelBuffer,
        /// Read-write texel buffer.
        ReadWriteTexelBuffer,
        /// Input attachment.
        InputAttachment,
    }

    /// Texture dimension.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TextureDimension {
        /// One-dimensional texture.
        D1,
        /// Two-dimensional texture.
        D2,
        /// Two-dimensional array texture.
        D2Array,
        /// Three-dimensional texture.
        D3,
        /// Cube texture.
        Cube,
        /// Cube array texture.
        CubeArray,
        /// No texture dimension.
        None,
    }

    /// Sampled texture component kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SampledKind {
        /// Float component.
        Float,
        /// Unsigned integer component.
        UInt,
        /// Signed integer component.
        SInt,
        /// Filterable float component.
        Filterable,
        /// Unfilterable float component.
        Unfilterable,
        /// Unknown filterability.
        UnknownFilterable,
    }

    /// Sampler binding kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SamplerType {
        /// Comparison sampler.
        Comparison,
        /// Filtering sampler.
        Filtering,
        /// Non-filtering sampler.
        NonFiltering,
        /// Unknown filtering mode.
        UnknownFiltering,
    }

    /// Storage texture texel format.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TexelFormat {
        /// r8snorm.
        R8Snorm,
        /// r8uint.
        R8Uint,
        /// r8sint.
        R8Sint,
        /// rg8unorm.
        Rg8Unorm,
        /// rg8snorm.
        Rg8Snorm,
        /// rg8uint.
        Rg8Uint,
        /// rg8sint.
        Rg8Sint,
        /// r16unorm.
        R16Unorm,
        /// r16snorm.
        R16Snorm,
        /// r16uint.
        R16Uint,
        /// r16sint.
        R16Sint,
        /// r16float.
        R16Float,
        /// rg16unorm.
        Rg16Unorm,
        /// rg16snorm.
        Rg16Snorm,
        /// rg16uint.
        Rg16Uint,
        /// rg16sint.
        Rg16Sint,
        /// rg16float.
        Rg16Float,
        /// bgra8unorm.
        Bgra8Unorm,
        /// rgba8unorm.
        Rgba8Unorm,
        /// rgba8snorm.
        Rgba8Snorm,
        /// rgba8uint.
        Rgba8Uint,
        /// rgba8sint.
        Rgba8Sint,
        /// rgba16unorm.
        Rgba16Unorm,
        /// rgba16snorm.
        Rgba16Snorm,
        /// rgba16uint.
        Rgba16Uint,
        /// rgba16sint.
        Rgba16Sint,
        /// rgba16float.
        Rgba16Float,
        /// r32uint.
        R32Uint,
        /// r32sint.
        R32Sint,
        /// r32float.
        R32Float,
        /// rg32uint.
        Rg32Uint,
        /// rg32sint.
        Rg32Sint,
        /// rg32float.
        Rg32Float,
        /// rgba32uint.
        Rgba32Uint,
        /// rgba32sint.
        Rgba32Sint,
        /// rgba32float.
        Rgba32Float,
        /// r8unorm.
        R8Unorm,
        /// rgb10a2uint.
        Rgb10A2Uint,
        /// rgb10a2unorm.
        Rgb10A2Unorm,
        /// rg11b10ufloat.
        Rg11B10Ufloat,
        /// No texel format.
        None,
    }

    /// A reflected pipeline override.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Override {
        /// Override name.
        pub name: String,
        /// Numeric override identifier.
        pub id: u16,
        /// Override scalar type.
        pub type_class: OverrideType,
        /// Whether the override has a default initializer.
        pub has_default: bool,
        /// Reflected default value when Tint exposes it.
        pub default_value: f64,
    }

    /// Override scalar type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum OverrideType {
        /// Boolean override.
        Bool,
        /// f32 override.
        Float32,
        /// u32 override.
        Uint32,
        /// i32 override.
        Int32,
        /// f16 override.
        Float16,
    }

    /// A single binding remap from source group/binding to destination group/binding.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BindingRemap {
        /// Source bind group.
        pub group: u32,
        /// Source binding.
        pub binding: u32,
        /// Destination bind group.
        pub dst_group: u32,
        /// Destination binding.
        pub dst_binding: u32,
    }

    /// Resource binding remap sets grouped by resource class.
    #[derive(Debug, Clone, Default)]
    pub struct Bindings {
        /// Uniform buffer binding remaps.
        pub uniform: Vec<BindingRemap>,
        /// Storage buffer binding remaps.
        pub storage: Vec<BindingRemap>,
        /// Sampled/depth/external texture binding remaps.
        pub texture: Vec<BindingRemap>,
        /// Storage texture binding remaps.
        pub storage_texture: Vec<BindingRemap>,
        /// Sampler binding remaps.
        pub sampler: Vec<BindingRemap>,
    }

    /// Pipeline override substitution value.
    #[derive(Debug, Clone, PartialEq)]
    pub struct OverrideValue {
        /// Override name or numeric ID encoded as decimal text.
        pub name: String,
        /// Value to substitute, converted by Tint to the declared override type.
        pub value: f64,
    }

    /// Compiles WGSL to MSL using the default parse and binding behavior.
    pub fn wgsl_to_msl(_wgsl: &str, _entry_point: &str) -> Result<String, String> {
        Err(UNAVAILABLE.to_owned())
    }
}

pub use imp::*;

#[cfg(all(test, have_tint))]
mod tests {
    use super::*;

    fn compute_wgsl() -> &'static str {
        "@compute @workgroup_size(8, 1, 1) fn cs() {}"
    }

    fn render_wgsl() -> &'static str {
        r#"
struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0));
  var out: VsOut;
  out.pos = vec4f(p[vi], 0.0, 1.0);
  out.uv = p[vi];
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  return vec4f(in.uv, 0.0, 1.0);
}
"#
    }

    fn spirv_has_binding_decoration(words: &[u32], binding: u32) -> bool {
        words.windows(4).any(|w| {
            let opcode = w[0] & 0xffff;
            let word_count = w[0] >> 16;
            opcode == 71 && word_count == 4 && w[2] == 33 && w[3] == binding
        })
    }

    #[test]
    fn compute_generates_msl_spirv_glsl() {
        let program = Program::parse(compute_wgsl(), false).unwrap();
        let bindings = Bindings::default();
        let msl = program.generate_msl("cs", &bindings, &[], true).unwrap();
        assert!(msl.source.contains("kernel"), "MSL:\n{}", msl.source);
        let spirv = program.generate_spirv("cs", &bindings, &[], true).unwrap();
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));
        let glsl = program.generate_glsl("cs", &bindings, &[]).unwrap();
        assert!(glsl.contains("#version 310 es"), "GLSL:\n{glsl}");
    }

    #[test]
    fn render_stages_generate_msl_and_spirv() {
        let program = Program::parse(render_wgsl(), false).unwrap();
        let bindings = Bindings::default();
        for ep in ["vs", "fs"] {
            let msl = program.generate_msl(ep, &bindings, &[], true).unwrap();
            assert!(!msl.source.is_empty());
            let spirv = program.generate_spirv(ep, &bindings, &[], true).unwrap();
            assert_eq!(spirv.first().copied(), Some(0x0723_0203));
        }
    }

    #[test]
    fn reflects_bindings_and_workgroup_size() {
        let wgsl = r#"
struct U { value: vec4f }
struct S { value: array<vec4f, 4> }
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(2) var<storage, read_write> s: S;
@group(2) @binding(3) var t: texture_2d<f32>;
@group(2) @binding(4) var samp: sampler;

@compute @workgroup_size(8, 4, 1)
fn cs() {
  let c = textureSampleLevel(t, samp, vec2f(0.5), 0.0);
  s.value[0] = u.value + c;
}
"#;
        let program = Program::parse(wgsl, false).unwrap();
        let entries = program.entry_points().unwrap();
        assert_eq!(entries[0].workgroup_size, Some([8, 4, 1]));
        let bindings = program.resource_bindings("cs").unwrap();
        assert!(bindings.iter().any(|b| {
            b.group == 0 && b.binding == 0 && b.resource_type == ResourceType::UniformBuffer
        }));
        assert!(bindings.iter().any(|b| {
            b.group == 1 && b.binding == 2 && b.resource_type == ResourceType::StorageBuffer
        }));
        assert!(bindings.iter().any(|b| {
            b.group == 2 && b.binding == 3 && b.resource_type == ResourceType::SampledTexture
        }));
        assert!(bindings
            .iter()
            .any(|b| b.group == 2 && b.binding == 4 && b.resource_type == ResourceType::Sampler));
    }

    #[test]
    fn reflects_and_substitutes_overrides() {
        let wgsl = r#"
override x: u32 = 4;
@compute @workgroup_size(x, 1, 1)
fn cs() {}
"#;
        let program = Program::parse(wgsl, false).unwrap();
        let overrides = program.overrides().unwrap();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].name, "x");
        assert_eq!(overrides[0].type_class, OverrideType::Uint32);
        assert!(overrides[0].has_default);
        assert_eq!(overrides[0].default_value, 4.0);

        let bindings = Bindings::default();
        let a = program
            .generate_msl(
                "cs",
                &bindings,
                &[OverrideValue {
                    name: "x".to_owned(),
                    value: 2.0,
                }],
                true,
            )
            .unwrap()
            .source;
        let b = program
            .generate_msl(
                "cs",
                &bindings,
                &[OverrideValue {
                    name: "x".to_owned(),
                    value: 5.0,
                }],
                true,
            )
            .unwrap()
            .source;
        assert_ne!(a, b);
    }

    #[test]
    fn reflects_override_default_values() {
        let wgsl = r#"
override f: f32 = 1.5;
override b: bool = true;
override u: u32 = 7u;
override i: i32 = -3i;
@compute @workgroup_size(1) fn cs() {}
"#;
        let program = Program::parse(wgsl, false).unwrap();
        let overrides = program.overrides().unwrap();
        let default = |name: &str| {
            overrides
                .iter()
                .find(|override_| override_.name == name)
                .map(|override_| override_.default_value)
                .unwrap()
        };
        assert_eq!(default("f"), 1.5);
        assert_eq!(default("b"), 1.0);
        assert_eq!(default("u"), 7.0);
        assert_eq!(default("i"), -3.0);
    }

    #[test]
    fn binding_remap_changes_msl_and_spirv() {
        let wgsl = r#"
struct U { value: vec4f }
@group(0) @binding(0) var<uniform> u: U;
@compute @workgroup_size(1)
fn cs() { _ = u.value; }
"#;
        let program = Program::parse(wgsl, false).unwrap();
        let default_bindings = Bindings::default();
        let remapped = Bindings {
            uniform: vec![BindingRemap {
                group: 0,
                binding: 0,
                dst_group: 0,
                dst_binding: 7,
            }],
            ..Bindings::default()
        };
        let default_msl = program
            .generate_msl("cs", &default_bindings, &[], true)
            .unwrap()
            .source;
        let remapped_msl = program
            .generate_msl("cs", &remapped, &[], true)
            .unwrap()
            .source;
        assert!(remapped_msl.contains("[[buffer(7)]]"), "{remapped_msl}");
        assert_ne!(default_msl, remapped_msl);

        let default_spv = program
            .generate_spirv("cs", &default_bindings, &[], true)
            .unwrap();
        let remapped_spv = program.generate_spirv("cs", &remapped, &[], true).unwrap();
        assert_ne!(default_spv, remapped_spv);
        assert!(spirv_has_binding_decoration(&remapped_spv, 7));
    }

    #[test]
    fn shader_f16_parses_and_generates() {
        let wgsl = r#"
enable f16;
@compute @workgroup_size(1)
fn cs() {
  let x: f16 = 1.0h;
  _ = x;
}
"#;
        let program = Program::parse(wgsl, true).unwrap();
        let msl = program
            .generate_msl("cs", &Bindings::default(), &[], true)
            .unwrap();
        assert!(msl.source.contains("kernel"));
    }

    #[test]
    fn invalid_wgsl_reports_error() {
        let err = Program::parse("this is not wgsl", false).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn legacy_smoke_wrapper_still_works() {
        let msl = wgsl_to_msl(compute_wgsl(), "cs").unwrap();
        assert!(msl.contains("kernel"));
    }
}
