/// Stores GLES sampler data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesSampler;

impl std::fmt::Debug for GlesSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesSampler").finish()
    }
}
