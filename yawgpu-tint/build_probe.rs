//! Pure selection/parsing and filesystem helpers shared by `yawgpu-tint`'s
//! build script and its `tests/build_probe.rs` test target.
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

pub(crate) fn apple_deployment_target_var(target_os: &str) -> Option<&'static str> {
    match target_os {
        "macos" => Some("MACOSX_DEPLOYMENT_TARGET"),
        "ios" => Some("IPHONEOS_DEPLOYMENT_TARGET"),
        _ => None,
    }
}

/// Reads a single `VARIABLE=VERSION` line from rustc's deployment-target output.
pub(crate) fn parse_apple_deployment_target<'a>(
    variable: &str,
    output: &'a str,
) -> Option<&'a str> {
    let (name, version) = output.trim().split_once('=')?;
    let mut components = version.split('.');
    if name != variable
        || components.clone().count() > 3
        || !components.all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    Some(version)
}

/// Uses an explicit environment value verbatim, otherwise parses rustc's default.
pub(crate) fn select_apple_deployment_target<'a>(
    variable: &str,
    env_value: Option<&'a str>,
    rustc_output: Option<&'a str>,
) -> Option<&'a str> {
    env_value.or_else(|| parse_apple_deployment_target(variable, rustc_output?))
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
    use super::{
        android_abi_for_arch, apple_deployment_target_var, dawn_checkout_usable, dawn_probe_paths,
        parse_apple_deployment_target, select_apple_deployment_target,
    };
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
    fn apple_deployment_target_var_selects_only_supported_operating_systems() {
        assert_eq!(
            apple_deployment_target_var("macos"),
            Some("MACOSX_DEPLOYMENT_TARGET")
        );
        // Device and simulator targets both have CARGO_CFG_TARGET_OS=ios.
        assert_eq!(
            apple_deployment_target_var("ios"),
            Some("IPHONEOS_DEPLOYMENT_TARGET")
        );
        for os in [
            "android", "linux", "windows", "tvos", "watchos", "visionos", "",
        ] {
            assert_eq!(apple_deployment_target_var(os), None, "{os}");
        }
    }

    #[test]
    fn parse_apple_deployment_target_accepts_rustc_versions() {
        for (variable, output, expected) in [
            (
                "MACOSX_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET=11.0\n",
                "11.0",
            ),
            (
                "IPHONEOS_DEPLOYMENT_TARGET",
                "IPHONEOS_DEPLOYMENT_TARGET=14.0\n",
                "14.0",
            ),
            (
                "IPHONEOS_DEPLOYMENT_TARGET",
                "IPHONEOS_DEPLOYMENT_TARGET=10.0\r\n",
                "10.0",
            ),
            (
                "MACOSX_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET=13.2.1",
                "13.2.1",
            ),
            (
                "MACOSX_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET=13",
                "13",
            ),
        ] {
            assert_eq!(
                parse_apple_deployment_target(variable, output),
                Some(expected)
            );
        }
    }

    #[test]
    fn parse_apple_deployment_target_rejects_empty_malformed_and_wrong_variable_output() {
        for output in [
            "",
            "\n",
            "11.0",
            "MACOSX_DEPLOYMENT_TARGET",
            "MACOSX_DEPLOYMENT_TARGET=",
            "MACOSX_DEPLOYMENT_TARGET=latest",
            "MACOSX_DEPLOYMENT_TARGET=11..0",
            "MACOSX_DEPLOYMENT_TARGET=.11",
            "MACOSX_DEPLOYMENT_TARGET=11.",
            "MACOSX_DEPLOYMENT_TARGET=11.0.0.1",
            "MACOSX_DEPLOYMENT_TARGET=-11.0",
            "MACOSX_DEPLOYMENT_TARGET=11.0=12.0",
            "MACOSX_DEPLOYMENT_TARGET= 11.0",
            "MACOSX_DEPLOYMENT_TARGET=11.0\nextra",
            "IPHONEOS_DEPLOYMENT_TARGET=14.0",
            "MACOSX_DEPLOYMENT_TARGET=11.0\nMACOSX_DEPLOYMENT_TARGET=12.0",
        ] {
            assert_eq!(
                parse_apple_deployment_target("MACOSX_DEPLOYMENT_TARGET", output),
                None,
                "{output:?}"
            );
        }
        assert_eq!(
            parse_apple_deployment_target(
                "IPHONEOS_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET=11.0"
            ),
            None
        );
    }

    #[test]
    fn select_apple_deployment_target_preserves_explicit_values() {
        for variable in ["MACOSX_DEPLOYMENT_TARGET", "IPHONEOS_DEPLOYMENT_TARGET"] {
            for value in ["13.0", "13.0.0", "", "as given"] {
                for output in [
                    Some("malformed"),
                    Some("MACOSX_DEPLOYMENT_TARGET=11.0"),
                    Some("IPHONEOS_DEPLOYMENT_TARGET=14.0"),
                    None,
                ] {
                    assert_eq!(
                        select_apple_deployment_target(variable, Some(value), output),
                        Some(value)
                    );
                }
            }
        }
    }

    #[test]
    fn select_apple_deployment_target_uses_rustc_only_without_an_override() {
        for (variable, output, expected) in [
            (
                "MACOSX_DEPLOYMENT_TARGET",
                "MACOSX_DEPLOYMENT_TARGET=11.0",
                "11.0",
            ),
            (
                "IPHONEOS_DEPLOYMENT_TARGET",
                "IPHONEOS_DEPLOYMENT_TARGET=14.0",
                "14.0",
            ),
        ] {
            assert_eq!(
                select_apple_deployment_target(variable, None, Some(output)),
                Some(expected)
            );
            assert_eq!(
                select_apple_deployment_target(variable, None, Some("malformed")),
                None
            );
            assert_eq!(select_apple_deployment_target(variable, None, None), None);
        }
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
