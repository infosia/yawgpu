#![allow(dead_code)]

#[path = "cts/validation/common.rs"]
mod common;

#[path = "cts/validation/capability_checks/features/common.rs"]
mod feature_common;

#[path = "cts/validation/capability_checks/features/clip_distances.rs"]
mod clip_distances;
#[path = "cts/validation/capability_checks/features/query_types.rs"]
mod query_types;
#[path = "cts/validation/capability_checks/features/subgroup_size_control.rs"]
mod subgroup_size_control;
#[path = "cts/validation/capability_checks/features/texture_component_swizzle.rs"]
mod texture_component_swizzle;
#[path = "cts/validation/capability_checks/features/texture_formats.rs"]
mod texture_formats;
#[path = "cts/validation/capability_checks/features/texture_formats_tier1.rs"]
mod texture_formats_tier1;
#[path = "cts/validation/capability_checks/features/texture_formats_tier2.rs"]
mod texture_formats_tier2;

#[path = "cts/validation/texture/bgra8unorm_storage.rs"]
mod bgra8unorm_storage;
#[path = "cts/validation/texture/float32_filterable.rs"]
mod float32_filterable;
#[path = "cts/validation/texture/rg11b10ufloat_renderable.rs"]
mod rg11b10ufloat_renderable;
