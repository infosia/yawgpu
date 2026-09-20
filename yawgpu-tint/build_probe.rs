//! Pure, filesystem-only helpers shared by `yawgpu-tint`'s build script and its
//! `tests/build_probe.rs` test target.
//!
//! Cargo compiles `build.rs` without `--test`, so a `#[cfg(test)] mod tests`
//! living there is never built into a libtest harness and never runs. Block 98's
//! R7 therefore keeps the decision logic that is worth testing in this standalone
//! module, which both `build.rs` and a real test target include with `#[path]`.
//! Nothing here may depend on a build-dependency (`cmake` is not linkable from a
//! test target) — `std` only.

use std::path::{Path, PathBuf};

pub(crate) fn android_abi_for_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "aarch64" => Some("arm64-v8a"),
        "arm" => Some("armeabi-v7a"),
        "x86_64" => Some("x86_64"),
        "x86" => Some("x86"),
        "riscv64" => Some("riscv64"),
        _ => None,
    }
}

/// The files whose existence decides whether a candidate Dawn root is usable:
/// the checkout's own `CMakeLists.txt` (submodule initialized) and abseil's
/// (`tools/fetch_dawn_dependencies.py` has run).
///
/// This is the single source for both the existence probe in
/// [`dawn_checkout_usable`] and the `cargo:rerun-if-changed` keys emitted in
/// `main`, so the decision inputs and the rerun keys cannot drift apart.
pub(crate) fn dawn_probe_paths(root: &Path) -> [PathBuf; 2] {
    [
        root.join("CMakeLists.txt"),
        root.join("third_party")
            .join("abseil-cpp")
            .join("CMakeLists.txt"),
    ]
}

/// Whether `root` holds a Dawn checkout complete enough to build Tint from:
/// every [`dawn_probe_paths`] entry must be an existing file.
pub(crate) fn dawn_checkout_usable(root: &Path) -> bool {
    dawn_probe_paths(root).iter().all(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::{android_abi_for_arch, dawn_checkout_usable, dawn_probe_paths};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A unique directory under the system temp dir, removed on drop so a failing
    /// assertion does not leave the tree behind. `build.rs` has no
    /// dev-dependencies, so this stands in for `tempfile`.
    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before the unix epoch")
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "yawgpu-tint-build-{tag}-{}-{nanos}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).expect("create temp root");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        /// Creates `relative` (and its parent directories) as an empty file.
        fn touch(&self, relative: &Path) {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create parent dir");
            }
            fs::write(&path, b"").expect("create probe file");
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn android_abi_for_arch_maps_supported_targets() {
        assert_eq!(android_abi_for_arch("aarch64"), Some("arm64-v8a"));
        assert_eq!(android_abi_for_arch("arm"), Some("armeabi-v7a"));
        assert_eq!(android_abi_for_arch("x86_64"), Some("x86_64"));
        assert_eq!(android_abi_for_arch("x86"), Some("x86"));
        assert_eq!(android_abi_for_arch("riscv64"), Some("riscv64"));
        assert_eq!(android_abi_for_arch("wasm32"), None);
    }

    #[test]
    fn dawn_probe_paths_names_the_two_decision_inputs() {
        let root = Path::new("/some/root");
        assert_eq!(
            dawn_probe_paths(root),
            [
                PathBuf::from("/some/root/CMakeLists.txt"),
                PathBuf::from("/some/root/third_party/abseil-cpp/CMakeLists.txt"),
            ]
        );
    }

    #[test]
    fn dawn_checkout_usable_rejects_missing_root() {
        let root = TempRoot::new("missing");
        assert!(!dawn_checkout_usable(&root.path().join("no-such-dawn")));
    }

    #[test]
    fn dawn_checkout_usable_rejects_empty_root() {
        let root = TempRoot::new("empty");
        assert!(!dawn_checkout_usable(root.path()));
    }

    #[test]
    fn dawn_checkout_usable_rejects_unfetched_submodule() {
        let root = TempRoot::new("unfetched");
        // Submodule initialized, `fetch_dawn_dependencies.py` not run yet.
        root.touch(Path::new("CMakeLists.txt"));
        assert!(!dawn_checkout_usable(root.path()));
    }

    #[test]
    fn dawn_checkout_usable_accepts_complete_checkout() {
        let root = TempRoot::new("complete");
        root.touch(Path::new("CMakeLists.txt"));
        root.touch(Path::new("third_party/abseil-cpp/CMakeLists.txt"));
        assert!(dawn_checkout_usable(root.path()));
    }

    /// The transition R2/R4 turn on: the same root flips to usable as the setup
    /// completes, and back to unusable when the checkout is torn down again.
    #[test]
    fn dawn_checkout_usable_tracks_setup_transitions() {
        let root = TempRoot::new("transition");
        assert!(!dawn_checkout_usable(root.path()));

        root.touch(Path::new("CMakeLists.txt"));
        assert!(!dawn_checkout_usable(root.path()));

        let abseil = Path::new("third_party/abseil-cpp/CMakeLists.txt");
        root.touch(abseil);
        assert!(dawn_checkout_usable(root.path()));

        fs::remove_file(root.path().join(abseil)).expect("remove probe file");
        assert!(!dawn_checkout_usable(root.path()));
    }
}
