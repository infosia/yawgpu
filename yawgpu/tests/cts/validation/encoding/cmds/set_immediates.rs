//! CTS `webgpu/api/validation/encoding/cmds/setImmediates.spec.ts` is N/A.
//!
//! The CTS cases cover `setImmediates` immediate-data writes. yawgpu's pinned
//! `webgpu.h` declares pass/bundle `SetImmediates` entry points and generated
//! Rust bindings include their proc-pointer types, but bindgen function
//! declarations are intentionally ignored and yawgpu has no hand-written
//! exported `wgpu*SetImmediates` functions or core command implementation.
//! The 3 `g.test()` cases (`alignment`, `overflow`, `out_of_bounds`) are
//! therefore not portable until yawgpu implements that FFI/core surface.
