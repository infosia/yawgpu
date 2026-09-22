use super::*;

/// A rectangle of mip levels and array layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SubresourceRange {
    /// First mip level.
    pub(super) base_mip_level: u32,
    /// Number of mip levels.
    pub(super) mip_level_count: u32,
    /// First array layer.
    pub(super) base_array_layer: u32,
    /// Number of array layers.
    pub(super) array_layer_count: u32,
}

/// A coalesced rectangle sharing the same old layout state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LayoutRun {
    /// Subresources requiring a barrier.
    pub(super) range: SubresourceRange,
    /// Layout state before the transition.
    pub(super) old_state: u8,
}

/// Layout state indexed by mip level, then array layer (Block 105 R1).
#[derive(Debug)]
pub(super) struct SubresourceLayouts {
    mip_level_count: u32,
    array_layers: u32,
    states: Mutex<Vec<u8>>,
}

impl SubresourceLayouts {
    /// Initializes all subresources to UNDEFINED.
    pub(super) fn new(mip_level_count: u32, array_layers: u32) -> Self {
        Self {
            mip_level_count,
            array_layers,
            states: Mutex::new(
                (0..mip_level_count)
                    .flat_map(|_| (0..array_layers).map(|_| IMAGE_LAYOUT_UNDEFINED))
                    .collect(),
            ),
        }
    }

    /// Returns the rectangle covering the whole image.
    pub(super) fn whole(&self) -> SubresourceRange {
        SubresourceRange {
            base_mip_level: 0,
            mip_level_count: self.mip_level_count,
            base_array_layer: 0,
            array_layer_count: self.array_layers,
        }
    }

    /// Validates nonempty bounds without overflowing endpoint arithmetic.
    fn validate(&self, range: SubresourceRange) -> Result<(), HalError> {
        if range.mip_level_count == 0
            || range.array_layer_count == 0
            || range.base_mip_level >= self.mip_level_count
            || range.mip_level_count > self.mip_level_count - range.base_mip_level
            || range.base_array_layer >= self.array_layers
            || range.array_layer_count > self.array_layers - range.base_array_layer
        {
            return Err(texture_error("subresource range exceeds texture"));
        }
        Ok(())
    }

    /// Computes a checked flat index for a validated subresource.
    fn index(&self, mip: u32, layer: u32) -> Result<usize, HalError> {
        usize::try_from(u64::from(mip) * u64::from(self.array_layers) + u64::from(layer))
            .map_err(|_| texture_error("subresource range exceeds texture"))
    }

    /// Records a layout and returns barrier runs in ascending mip/layer order.
    pub(super) fn transition(
        &self,
        range: SubresourceRange,
        new_state: u8,
    ) -> Result<Vec<LayoutRun>, HalError> {
        self.validate(range)?;
        let mut states = self
            .states
            .lock()
            .map_err(|_| texture_error("subresource layout lock poisoned"))?;
        let mut result = Vec::new();
        let mut previous: Vec<LayoutRun> = Vec::new();
        for mip in range.base_mip_level..range.base_mip_level + range.mip_level_count {
            let mut runs: Vec<LayoutRun> = Vec::new();
            for layer in range.base_array_layer..range.base_array_layer + range.array_layer_count {
                let state = states
                    .get_mut(self.index(mip, layer)?)
                    .ok_or_else(|| texture_error("subresource range exceeds texture"))?;
                let old_state = *state;
                *state = new_state;
                if !needs_barrier(old_state, new_state) {
                    continue;
                }
                if let Some(last) = runs.last_mut() {
                    if last.old_state == old_state
                        && last.range.base_array_layer + last.range.array_layer_count == layer
                    {
                        last.range.array_layer_count += 1;
                        continue;
                    }
                }
                runs.push(LayoutRun {
                    range: SubresourceRange {
                        base_mip_level: mip,
                        mip_level_count: 1,
                        base_array_layer: layer,
                        array_layer_count: 1,
                    },
                    old_state,
                });
            }
            let identical = previous.len() == runs.len()
                && previous.iter().zip(&runs).all(|(a, b)| {
                    a.old_state == b.old_state
                        && a.range.base_array_layer == b.range.base_array_layer
                        && a.range.array_layer_count == b.range.array_layer_count
                });
            if identical {
                for run in &mut previous {
                    run.range.mip_level_count += 1;
                }
            } else {
                result.append(&mut previous);
                previous = runs;
            }
        }
        result.append(&mut previous);
        Ok(result)
    }

    /// Records an implicit render-pass layout transition without barrier runs.
    pub(super) fn set(&self, range: SubresourceRange, state: u8) -> Result<(), HalError> {
        self.validate(range)?;
        let mut states = self
            .states
            .lock()
            .map_err(|_| texture_error("subresource layout lock poisoned"))?;
        for mip in range.base_mip_level..range.base_mip_level + range.mip_level_count {
            for layer in range.base_array_layer..range.base_array_layer + range.array_layer_count {
                *states
                    .get_mut(self.index(mip, layer)?)
                    .ok_or_else(|| texture_error("subresource range exceeds texture"))? = state;
            }
        }
        Ok(())
    }

    /// Returns one subresource's state for tests and debugging.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn state(&self, mip_level: u32, array_layer: u32) -> Result<u8, HalError> {
        self.validate(SubresourceRange {
            base_mip_level: mip_level,
            mip_level_count: 1,
            base_array_layer: array_layer,
            array_layer_count: 1,
        })?;
        self.states
            .lock()
            .map_err(|_| texture_error("subresource layout lock poisoned"))?
            .get(self.index(mip_level, array_layer)?)
            .copied()
            .ok_or_else(|| texture_error("subresource range exceeds texture"))
    }
}

/// Same-layout barriers are required only for transfer writes and GENERAL.
pub(super) fn needs_barrier(old_state: u8, new_state: u8) -> bool {
    old_state != new_state || matches!(new_state, IMAGE_LAYOUT_TRANSFER_DST | IMAGE_LAYOUT_GENERAL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(mip: u32, mips: u32, layer: u32, layers: u32) -> SubresourceRange {
        SubresourceRange {
            base_mip_level: mip,
            mip_level_count: mips,
            base_array_layer: layer,
            array_layer_count: layers,
        }
    }

    #[test]
    fn new_and_whole_cover_all_undefined_subresources() {
        let layouts = SubresourceLayouts::new(2, 4);
        assert_eq!(layouts.whole(), range(0, 2, 0, 4));
        for mip in 0..2 {
            for layer in 0..4 {
                assert_eq!(layouts.state(mip, layer).unwrap(), IMAGE_LAYOUT_UNDEFINED);
            }
        }
    }

    #[test]
    fn transition_changes_only_requested_subresources() {
        let layouts = SubresourceLayouts::new(2, 4);
        let selected = range(1, 1, 1, 2);
        assert_eq!(
            layouts
                .transition(selected, IMAGE_LAYOUT_TRANSFER_DST)
                .unwrap(),
            vec![LayoutRun {
                range: selected,
                old_state: IMAGE_LAYOUT_UNDEFINED
            }]
        );
        for mip in 0..2 {
            for layer in 0..4 {
                assert_eq!(
                    layouts.state(mip, layer).unwrap(),
                    if mip == 1 && (1..3).contains(&layer) {
                        IMAGE_LAYOUT_TRANSFER_DST
                    } else {
                        IMAGE_LAYOUT_UNDEFINED
                    }
                );
            }
        }
    }

    #[test]
    fn mixed_states_coalesce_layers_in_mip_order() {
        let layouts = SubresourceLayouts::new(2, 4);
        layouts
            .set(range(0, 1, 0, 2), IMAGE_LAYOUT_TRANSFER_DST)
            .unwrap();
        layouts
            .set(range(0, 1, 2, 2), IMAGE_LAYOUT_SHADER_READ_ONLY)
            .unwrap();
        assert_eq!(
            layouts
                .transition(layouts.whole(), IMAGE_LAYOUT_TRANSFER_SRC)
                .unwrap(),
            vec![
                LayoutRun {
                    range: range(0, 1, 0, 2),
                    old_state: IMAGE_LAYOUT_TRANSFER_DST
                },
                LayoutRun {
                    range: range(0, 1, 2, 2),
                    old_state: IMAGE_LAYOUT_SHADER_READ_ONLY
                },
                LayoutRun {
                    range: range(1, 1, 0, 4),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
            ]
        );
    }

    #[test]
    fn uniform_whole_image_coalesces_across_mips() {
        let layouts = SubresourceLayouts::new(3, 4);
        assert_eq!(
            layouts
                .transition(layouts.whole(), IMAGE_LAYOUT_TRANSFER_DST)
                .unwrap(),
            vec![LayoutRun {
                range: layouts.whole(),
                old_state: IMAGE_LAYOUT_UNDEFINED
            }]
        );
    }

    #[test]
    fn identical_split_mips_coalesce_each_layer_run() {
        let layouts = SubresourceLayouts::new(3, 4);
        layouts
            .set(range(0, 3, 0, 2), IMAGE_LAYOUT_TRANSFER_DST)
            .unwrap();
        assert_eq!(
            layouts
                .transition(layouts.whole(), IMAGE_LAYOUT_GENERAL)
                .unwrap(),
            vec![
                LayoutRun {
                    range: range(0, 3, 0, 2),
                    old_state: IMAGE_LAYOUT_TRANSFER_DST
                },
                LayoutRun {
                    range: range(0, 3, 2, 2),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
            ]
        );
    }

    #[test]
    fn same_state_barriers_follow_write_hazards() {
        for state in 0..=7 {
            let layouts = SubresourceLayouts::new(2, 3);
            layouts.set(layouts.whole(), state).unwrap();
            let runs = layouts.transition(layouts.whole(), state).unwrap();
            assert_eq!(
                runs.len(),
                usize::from(matches!(
                    state,
                    IMAGE_LAYOUT_TRANSFER_DST | IMAGE_LAYOUT_GENERAL
                ))
            );
        }
    }

    #[test]
    fn read_only_holes_and_nonadjacent_mips_keep_runs_separate() {
        let layouts = SubresourceLayouts::new(3, 3);
        layouts
            .set(range(0, 3, 1, 1), IMAGE_LAYOUT_TRANSFER_SRC)
            .unwrap();
        layouts
            .set(range(1, 1, 0, 3), IMAGE_LAYOUT_TRANSFER_SRC)
            .unwrap();
        assert_eq!(
            layouts
                .transition(layouts.whole(), IMAGE_LAYOUT_TRANSFER_SRC)
                .unwrap(),
            vec![
                LayoutRun {
                    range: range(0, 1, 0, 1),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
                LayoutRun {
                    range: range(0, 1, 2, 1),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
                LayoutRun {
                    range: range(2, 1, 0, 1),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
                LayoutRun {
                    range: range(2, 1, 2, 1),
                    old_state: IMAGE_LAYOUT_UNDEFINED
                },
            ]
        );
    }

    #[test]
    fn set_records_state_without_barrier_runs() {
        let layouts = SubresourceLayouts::new(2, 4);
        layouts
            .set(range(1, 1, 2, 1), IMAGE_LAYOUT_PRESENT)
            .unwrap();
        assert_eq!(layouts.state(1, 2).unwrap(), IMAGE_LAYOUT_PRESENT);
        assert_eq!(layouts.state(1, 1).unwrap(), IMAGE_LAYOUT_UNDEFINED);
    }

    #[test]
    fn invalid_and_zero_ranges_return_errors_without_mutation() {
        let layouts = SubresourceLayouts::new(2, 4);
        for invalid in [
            range(0, 0, 0, 1),
            range(0, 1, 0, 0),
            range(2, 1, 0, 1),
            range(0, 1, 4, 1),
            range(1, 2, 0, 1),
            range(0, 1, 3, 2),
            range(u32::MAX, 2, 0, 1),
            range(0, 1, u32::MAX, 2),
        ] {
            assert!(layouts
                .transition(invalid, IMAGE_LAYOUT_GENERAL)
                .unwrap_err()
                .to_string()
                .contains("subresource range exceeds texture"));
            assert!(layouts.set(invalid, IMAGE_LAYOUT_GENERAL).is_err());
        }
        assert!(layouts.state(2, 0).is_err());
        assert!(layouts.state(0, 4).is_err());
        assert_eq!(layouts.state(0, 0).unwrap(), IMAGE_LAYOUT_UNDEFINED);
        assert!(SubresourceLayouts::new(0, 1)
            .transition(range(0, 1, 0, 1), 1)
            .is_err());
    }

    #[test]
    fn three_dimensional_mips_use_one_array_layer() {
        let layouts = SubresourceLayouts::new(4, 1);
        layouts
            .set(range(2, 1, 0, 1), IMAGE_LAYOUT_GENERAL)
            .unwrap();
        assert_eq!(layouts.state(2, 0).unwrap(), IMAGE_LAYOUT_GENERAL);
        assert_eq!(layouts.state(3, 0).unwrap(), IMAGE_LAYOUT_UNDEFINED);
        assert!(layouts.state(2, 1).is_err());
    }

    #[test]
    fn needs_barrier_covers_every_layout_pair() {
        let states = [
            (IMAGE_LAYOUT_UNDEFINED, false),
            (IMAGE_LAYOUT_TRANSFER_DST, true),
            (IMAGE_LAYOUT_TRANSFER_SRC, false),
            (IMAGE_LAYOUT_COLOR_ATTACHMENT, false),
            (IMAGE_LAYOUT_PRESENT, false),
            (IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT, false),
            (IMAGE_LAYOUT_GENERAL, true),
            (IMAGE_LAYOUT_SHADER_READ_ONLY, false),
        ];
        for (old, _) in states {
            for (new, same_state_barrier) in states {
                assert_eq!(
                    needs_barrier(old, new),
                    if old == new { same_state_barrier } else { true }
                );
            }
        }
    }
}
