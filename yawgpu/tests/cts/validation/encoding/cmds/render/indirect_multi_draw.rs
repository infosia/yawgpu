//! N/A for `$CTS/src/webgpu/api/validation/encoding/cmds/render/indirect_multi_draw.spec.ts`.
//!
//! The CTS cases target the Chromium experimental JavaScript commands
//! `multiDrawIndirect` and `multiDrawIndexedIndirect`. yawgpu's generated
//! `webgpu.h` bindings and hand-written C FFI expose
//! `wgpuRenderPassEncoderDrawIndirect` and
//! `wgpuRenderPassEncoderDrawIndexedIndirect`, but no `multiDraw*` command
//! or vendor extension surface. The six `g.test()` cases in this spec are
//! therefore non-portable for the current C ABI and are reported as N/A.
