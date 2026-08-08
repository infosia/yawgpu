use std::hash::{BuildHasherDefault, Hasher};
use std::mem::size_of;

/// Cheap hasher for maps whose keys are internal resource identities.
///
/// Resource identities are process-local addresses, not attacker-controlled
/// input, so HashDoS-resistant SipHash adds cost without providing a useful
/// security property. Rotating preserves every identity bit while moving the
/// commonly-zero alignment bits away from the low end.
#[derive(Debug, Default)]
pub(crate) struct IdentityHasher(usize);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0usize;
        for (index, byte) in bytes.iter().copied().enumerate().take(size_of::<usize>()) {
            value |= usize::from(byte) << (index * 8);
        }
        self.0 = value.rotate_right(4);
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = value.rotate_right(4);
    }
}

pub(crate) type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    #[test]
    fn identity_hasher_is_deterministic_for_usize_keys() {
        let mut direct = IdentityHasher::default();
        direct.write_usize(0x12340);

        let mut through_hash = IdentityHasher::default();
        0x12340usize.hash(&mut through_hash);

        assert_eq!(direct.finish(), through_hash.finish());
        assert_ne!(direct.finish(), 0);
    }
}
