/// Stores GLES compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesComputePipeline;

impl std::fmt::Debug for GlesComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesComputePipeline").finish()
    }
}

/// Stores GLES render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesRenderPipeline;

impl std::fmt::Debug for GlesRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesRenderPipeline").finish()
    }
}
