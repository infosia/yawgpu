//! GLES backend (Tier 2 / experimental).
//!
//! This module owns the Windows ANGLE / EGL bring-up path. Resource
//! implementations not covered by the current phase remain unavailable stubs.

mod adapter;
mod buffer;
mod device;
mod egl;
mod instance;
mod pipeline;
mod queue;
mod sampler;
mod surface;
mod texture;

pub use adapter::GlesAdapter;
pub use buffer::GlesBuffer;
pub use device::GlesDevice;
pub use instance::GlesInstance;
pub use pipeline::{GlesComputePipeline, GlesRenderPipeline};
pub use queue::GlesQueue;
pub use sampler::GlesSampler;
pub use surface::GlesSurface;
pub use texture::GlesTexture;

const BACKEND: &str = "gles";

fn unavailable<T>() -> Result<T, crate::HalError> {
    Err(crate::HalError::BackendUnavailable { backend: BACKEND })
}
