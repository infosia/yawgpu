//! Shared composition of the combined immediates block (Block 94).
//!
//! Both executing Tier-1 backends deliver one combined per-stage block per
//! draw/dispatch -- Metal via `set{Vertex,Fragment}Bytes`/`setBytes` at the
//! Tint-chosen buffer slot, Vulkan via `vkCmdPushConstants` at offset 0 --
//! with the same layout, mirroring Dawn's `ImmediatesLayout.h`
//! (`RenderImmediates`/`ComputeImmediates`): user immediate bytes first,
//! pipeline-internal constants (the 8-byte fragment depth-range pair) appended
//! directly after the layout-reserved user region. Metal's block carries
//! further internal members after that; see `metal/encode.rs`.

/// Composes the combined immediates block for one shader stage: user
/// immediate bytes at `[0, user_size)` and the two `f32` depth-range values
/// (`[min_depth, max_depth]`) written at `depth_range_offset` when present.
/// The rest of the block stays zero.
///
/// `user_data` is the pass's full immediate scratch snapshot (up to the
/// device's `maxImmediateSize` bytes); only the `user_size` prefix the
/// pipeline's layout reserves is copied, so the rest of the snapshot cannot
/// reach internal members, and any reserved suffix the snapshot does not
/// cover stays zero. Pure function -- no backend calls -- so it is directly
/// unit-testable without a real device.
pub(crate) fn compose_immediates_block(
    user_data: &[u8],
    user_size: u32,
    block_size: u32,
    depth_range_offset: Option<u32>,
    depth_range: [f32; 2],
) -> Vec<u8> {
    let block_size = block_size as usize;
    let mut block = vec![0u8; block_size];
    let copy_len = (user_size as usize).min(block_size).min(user_data.len());
    block[..copy_len].copy_from_slice(&user_data[..copy_len]);
    if let Some(offset) = depth_range_offset {
        let offset = offset as usize;
        if offset + 8 <= block.len() {
            block[offset..offset + 4].copy_from_slice(&depth_range[0].to_ne_bytes());
            block[offset + 4..offset + 8].copy_from_slice(&depth_range[1].to_ne_bytes());
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    /// User-only block (no internal depth-range pair): the composed block is
    /// exactly `block_size` bytes copied from the front of `user_data`.
    #[test]
    fn compose_immediates_block_copies_user_prefix_when_no_depth_range() {
        let user_data: Vec<u8> = (0..64u8).collect();

        let block = compose_immediates_block(&user_data, 16, 16, None, [0.0, 1.0]);

        assert_eq!(block, user_data[0..16]);
    }

    /// User + depth range: user bytes occupy `[0, depth_range_offset)`, the
    /// depth-range pair lands at `[offset, offset + 8)`, and no user bytes
    /// beyond the offset leak into the block (Dawn `RenderImmediates`
    /// layout: user prefix, then `ClampFragDepthArgs`).
    #[test]
    fn compose_immediates_block_appends_depth_range_after_user_prefix() {
        let user_data: Vec<u8> = (0..64u8).collect();

        let block = compose_immediates_block(&user_data, 16, 24, Some(16), [0.25, 0.75]);

        assert_eq!(block.len(), 24);
        assert_eq!(&block[0..16], &user_data[0..16]);
        assert_eq!(&block[16..20], &0.25f32.to_ne_bytes());
        assert_eq!(&block[20..24], &0.75f32.to_ne_bytes());
    }

    /// Snapshot bytes past `user_size` stay out of the block even when no
    /// depth-range pair follows the user region.
    #[test]
    fn compose_immediates_block_stops_copy_at_user_size() {
        let user_data = [7u8; 64];

        let block = compose_immediates_block(&user_data, 8, 32, None, [0.0, 1.0]);

        assert_eq!(block.len(), 32);
        assert_eq!(&block[..8], &[7u8; 8]);
        assert_eq!(&block[8..], &[0u8; 24]);
    }

    /// A depth-range-only pipeline (no user immediates reserved,
    /// `depth_range_offset == Some(0)`) composes exactly the bare 8-byte
    /// pair -- the pre-Block-94 behaviour the standalone Metal frag-depth
    /// clamp / Vulkan pixel-center-polyfill deliveries used to produce.
    #[test]
    fn compose_immediates_block_depth_range_only_pipeline_yields_bare_pair() {
        let block = compose_immediates_block(&[], 0, 8, Some(0), [0.1, 0.9]);

        assert_eq!(block.len(), 8);
        assert_eq!(&block[0..4], &0.1f32.to_ne_bytes());
        assert_eq!(&block[4..8], &0.9f32.to_ne_bytes());
    }

    /// A snapshot shorter than the reserved user region leaves the reserved
    /// suffix zero (pass scratch is zero at pass begin, so a short snapshot
    /// only happens for out-of-band callers; the composed block still never
    /// reads out of bounds).
    #[test]
    fn compose_immediates_block_zero_fills_reserved_suffix_beyond_snapshot() {
        let block = compose_immediates_block(&[7, 7], 8, 8, None, [0.0, 1.0]);

        assert_eq!(block, [7, 7, 0, 0, 0, 0, 0, 0]);
    }
}
