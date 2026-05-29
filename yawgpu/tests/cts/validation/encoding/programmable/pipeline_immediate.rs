// N/A: CTS pipeline_immediate.spec.ts requires the immediate-data command
// family (`wgpuComputePassEncoderSetImmediates`,
// `wgpuRenderPassEncoderSetImmediates`, and
// `wgpuRenderBundleEncoderSetImmediates`). The pinned webgpu.h declares these
// exports, but yawgpu only exposes generated proc-pointer typedefs for them and
// has no hand-written FFI exports or yawgpu-core command/state implementation.
// Until those commands exist, the four g.test() cases in this spec cannot be
// expressed as Rust API tests.
