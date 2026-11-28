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
}

/// Returns the address of the immutable, process-lifetime table for the pinned Dawn revision.
#[must_use]
pub fn dawn_proc_table() -> *const DawnProcTable {
    // SAFETY: The C++ accessor takes no arguments and points to a constant static table.
    unsafe { yawgpuDawnGetProcTable() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dawn_proc_table_has_a_stable_non_null_address() {
        let table = dawn_proc_table();
        assert!(!table.is_null());
        assert_eq!(table, dawn_proc_table());
    }
}
