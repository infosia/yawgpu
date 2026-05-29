//! CTS port of `webgpu/api/validation/dispatch.spec.ts`.

#[test]
#[ignore = "yawgpu does not expose a shader-language-feature query for linear_indexing and core does not validate total linear-indexing invocation range at command finish; CTS expects a finish error for oversized direct dispatch"]
fn dispatch_linear_indexing_range() {
    // Naga / core acceptance of the experimental `linear_indexing` builtins
    // is not a portable validation surface in yawgpu yet.
}

#[test]
#[ignore = "CTS dispatchIndirect linear_indexing range coverage is an operation/readback test; yawgpu's Noop validation layer cannot assert skipped execution from indirect workgroup counts"]
fn dispatch_indirect_linear_indexing_range() {
    // This CTS case observes output buffer contents after queue execution.
}
