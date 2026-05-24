/// Stores GLES texture data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesTexture;

impl std::fmt::Debug for GlesTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesTexture").finish()
    }
}
