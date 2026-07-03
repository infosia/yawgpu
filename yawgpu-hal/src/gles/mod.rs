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

pub(super) fn rebuild_hal_error(error: &crate::HalError) -> crate::HalError {
    // TODO: Consider deriving Clone for HalError upstream once all variants are
    // confirmed cheap or intentionally cloneable.
    match error {
        crate::HalError::BackendUnavailable { backend } => {
            crate::HalError::BackendUnavailable { backend }
        }
        crate::HalError::DeviceCreationFailed { backend } => {
            crate::HalError::DeviceCreationFailed { backend }
        }
        crate::HalError::QueueSubmissionFailed { backend, message } => {
            crate::HalError::QueueSubmissionFailed {
                backend,
                message: message.clone(),
            }
        }
        crate::HalError::BufferOperationFailed { backend, message } => {
            crate::HalError::BufferOperationFailed { backend, message }
        }
        crate::HalError::ShaderCompilationFailed { backend, message } => {
            crate::HalError::ShaderCompilationFailed {
                backend,
                message: message.clone(),
            }
        }
        crate::HalError::SwapchainCreationFailed { backend, message } => {
            crate::HalError::SwapchainCreationFailed { backend, message }
        }
        crate::HalError::AcquireFailed { backend, message } => {
            crate::HalError::AcquireFailed { backend, message }
        }
        crate::HalError::PresentFailed { backend, message } => {
            crate::HalError::PresentFailed { backend, message }
        }
        crate::HalError::OutOfMemory { backend, resource } => {
            crate::HalError::OutOfMemory { backend, resource }
        }
    }
}
