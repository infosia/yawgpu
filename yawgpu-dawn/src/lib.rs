//! Opt-in compatibility layer for Dawn's native WebGPU ABI.
#![cfg(feature = "dawn-compat")]

use yawgpu as _;

/// Opaque Dawn table. Its C++ layout is defined by the package's dawn_proc_table.h.
#[repr(C)]
pub struct DawnProcTable {
    _opaque: [u8; 0],
}

extern "C" {
    fn yawgpuDawnGetProcTable() -> *const DawnProcTable;
    fn yawgpuDawnSetDefaultBackend(backend: u32);
}

/// Returns the address of the immutable, process-lifetime table for the pinned Dawn revision.
#[must_use]
pub fn dawn_proc_table() -> *const DawnProcTable {
    // SAFETY: The C++ accessor takes no arguments and points to a constant static table.
    unsafe { yawgpuDawnGetProcTable() }
}

/// Sets the process-wide default for subsequent proc-table instance creation.
/// Pass a `YAWGPU_INSTANCE_BACKEND_*` value. An explicit selector takes precedence.
pub fn set_default_backend(backend: u32) {
    // SAFETY: The setter accepts any u32 and updates an atomic value.
    unsafe { yawgpuDawnSetDefaultBackend(backend) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yawgpu::{native, YaWGPUInstanceBackendSelect};

    #[link(name = "dawn_smoke", kind = "static")]
    extern "C" {
        fn smokeAdapterBackend(
            desc: *const native::WGPUInstanceDescriptor,
        ) -> native::WGPUBackendType;
    }

    #[test]
    fn default_backend_applies_only_without_a_selector() {
        let mut desc = native::WGPUInstanceDescriptor {
            nextInChain: std::ptr::null_mut(),
            requiredFeatureCount: 0,
            requiredFeatures: std::ptr::null(),
            requiredLimits: std::ptr::null(),
        };
        set_default_backend(yawgpu::YAWGPU_INSTANCE_BACKEND_NOOP);
        // SAFETY: Each descriptor and its chain are valid throughout the synchronous call.
        unsafe {
            assert_eq!(smokeAdapterBackend(&desc), native::WGPUBackendType_Null);
            assert!(desc.nextInChain.is_null());

            // The copied descriptor must still reject a missing requiredFeatures array.
            desc.requiredFeatureCount = 1;
            assert_eq!(
                smokeAdapterBackend(&desc),
                native::WGPUBackendType_Undefined
            );
            desc.requiredFeatureCount = 0;

            set_default_backend(yawgpu::YAWGPU_INSTANCE_BACKEND_GLES);
            assert_ne!(
                smokeAdapterBackend(std::ptr::null()),
                native::WGPUBackendType_Null
            );

            let mut selector = YaWGPUInstanceBackendSelect {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: yawgpu::YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
                },
                backend: yawgpu::YAWGPU_INSTANCE_BACKEND_NOOP,
            };
            let mut prefix = native::WGPUChainedStruct {
                next: &mut selector.chain,
                sType: 0,
            };
            desc.nextInChain = &mut prefix;
            assert_eq!(smokeAdapterBackend(&desc), native::WGPUBackendType_Null);
            assert_eq!(desc.nextInChain, std::ptr::addr_of_mut!(prefix));
            assert_eq!(prefix.next, std::ptr::addr_of_mut!(selector.chain));
        }
    }

    #[test]
    fn dawn_proc_table_has_a_stable_non_null_address() {
        let table = dawn_proc_table();
        assert!(!table.is_null());
        assert_eq!(table, dawn_proc_table());
    }
}
