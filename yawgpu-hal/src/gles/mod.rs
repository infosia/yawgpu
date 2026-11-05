//! GLES backend (Tier 2 / experimental).
//!
//! This module owns the Windows ANGLE / EGL bring-up path. Resource
//! implementations not covered by the current phase remain unavailable stubs.

// The static-enum dispatch pattern (CLAUDE.md "no `dyn Trait`") gives
// `GlesInstanceInner` / `GlesSurfaceInner` exactly one variant on
// non-Windows (the `Wgl` arm is `#[cfg(windows)]`-gated), which makes
// every `if let GlesInstanceInner::Egl(_) = ...` / `let ... else` in
// this module trivially irrefutable on those targets. The patterns
// themselves are correct and stay refutable on Windows; suppress the
// lint at module scope rather than annotating each call site.
#![allow(irrefutable_let_patterns)]

mod adapter;
mod buffer;
mod device;
mod egl;
mod exit_guard;
mod format;
mod instance;
mod pipeline;
mod queue;
mod sampler;
mod surface;
mod texture;
#[cfg(windows)]
mod wgl;

pub use adapter::GlesAdapter;
pub use buffer::GlesBuffer;
pub use device::GlesDevice;
pub use instance::{BackendChoice, GlesInstance};
pub use pipeline::{GlesComputePipeline, GlesRenderPipeline};
pub use queue::GlesQueue;
pub use sampler::GlesSampler;
pub use surface::GlesSurface;
pub use texture::GlesTexture;

const BACKEND: &str = "gles";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "manual real-backend test"]
    fn hal_device_shader_workarounds_gles_returns_all_false() {
        // GlesDevice owns a live GL context, so it has no device-free constructor.
        let device = GlesInstance::new().and_then(|instance| {
            instance
                .enumerate_adapters()
                .into_iter()
                .next()
                .ok_or(crate::HalError::BackendUnavailable { backend: BACKEND })?
                .create_device()
        });
        let device = match device {
            Ok(device) => crate::HalDevice::Gles(device),
            Err(error) => {
                eprintln!("skipping: GLES device unavailable: {error:?}");
                return;
            }
        };
        assert_eq!(
            device.shader_workarounds(),
            crate::HalShaderWorkarounds::default()
        );
    }
}
