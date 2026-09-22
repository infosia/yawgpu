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

/// Whether any storage binding references this image, regardless of view ranges.
pub(super) fn image_has_storage_binding(
    image: vk::Image,
    storage_images: impl IntoIterator<Item = vk::Image>,
) -> bool {
    storage_images.into_iter().any(|storage| storage == image)
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
        let mut error = None;
        let runs = coalesce_subresources(range, |mip, layer| {
            let state = self.index(mip, layer).and_then(|index| {
                states
                    .get_mut(index)
                    .ok_or_else(|| texture_error("subresource range exceeds texture"))
            });
            match state {
                Ok(state) => {
                    let old_state = *state;
                    *state = new_state;
                    needs_barrier(old_state, new_state).then_some(old_state)
                }
                Err(cause) => {
                    error = Some(cause);
                    None
                }
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        Ok(runs
            .into_iter()
            .map(|(range, old_state)| LayoutRun { range, old_state })
            .collect())
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

    /// Returns one subresource's state for tests.
    #[cfg(test)]
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

/// Coalesces classified subresources in ascending mip/layer order, excluding `None`.
/// Adjacent layers merge first, then adjacent mips with identical layer-run lists.
pub(super) fn coalesce_subresources<T: Copy + PartialEq>(
    range: SubresourceRange,
    mut classify: impl FnMut(u32, u32) -> Option<T>,
) -> Vec<(SubresourceRange, T)> {
    let mut result = Vec::new();
    let mut previous: Vec<(SubresourceRange, T)> = Vec::new();
    for mip in
        (0..range.mip_level_count).filter_map(|offset| range.base_mip_level.checked_add(offset))
    {
        let mut runs: Vec<(SubresourceRange, T)> = Vec::new();
        for layer in (0..range.array_layer_count)
            .filter_map(|offset| range.base_array_layer.checked_add(offset))
        {
            let Some(value) = classify(mip, layer) else {
                continue;
            };
            if let Some((last, last_value)) = runs.last_mut() {
                if *last_value == value
                    && last.base_array_layer.checked_add(last.array_layer_count) == Some(layer)
                {
                    last.array_layer_count += 1;
                    continue;
                }
            }
            runs.push((
                SubresourceRange {
                    base_mip_level: mip,
                    mip_level_count: 1,
                    base_array_layer: layer,
                    array_layer_count: 1,
                },
                value,
            ));
        }
        let identical = previous.len() == runs.len()
            && previous.iter().zip(&runs).all(|((a, av), (b, bv))| {
                av == bv
                    && a.base_array_layer == b.base_array_layer
                    && a.array_layer_count == b.array_layer_count
            });
        if identical {
            for (run, _) in &mut previous {
                run.mip_level_count += 1;
            }
        } else {
            result.append(&mut previous);
            previous = runs;
        }
    }
    result.append(&mut previous);
    result
}

/// Same-layout barriers are required only for transfer writes and GENERAL.
pub(super) fn needs_barrier(old_state: u8, new_state: u8) -> bool {
    old_state != new_state || matches!(new_state, IMAGE_LAYOUT_TRANSFER_DST | IMAGE_LAYOUT_GENERAL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn image_has_storage_binding_finds_same_image_among_storage_images() {
        let image = vk::Image::from_raw(17);
        assert!(image_has_storage_binding(
            image,
            [vk::Image::from_raw(18), image, vk::Image::from_raw(19)]
        ));
    }

    #[test]
    fn image_has_storage_binding_f1_disjoint_views_of_same_image_agree() {
        let image = vk::Image::from_raw(17);
        // F1: A spans mips 0..2, B samples mip 0, and storage reads mip 1.
        // Both sampled views use the same image-wide decision; their ranges
        // deliberately do not participate, including B's disjoint mip range.
        let sampled_views = [(image, range(0, 2, 0, 1)), (image, range(0, 1, 0, 1))];
        let storage_view = (image, range(1, 1, 0, 1));
        for (sampled_image, _) in sampled_views {
            assert!(image_has_storage_binding(sampled_image, [storage_view.0]));
        }
    }

    #[test]
    fn image_has_storage_binding_rejects_different_image() {
        assert!(!image_has_storage_binding(
            vk::Image::from_raw(17),
            [vk::Image::from_raw(18)]
        ));
    }

    #[test]
    fn image_has_storage_binding_rejects_empty_storage_set() {
        assert!(!image_has_storage_binding(vk::Image::from_raw(17), []));
    }

    /// A uniform classification forms one rectangle, including nonzero origins.
    #[test]
    fn coalescer_uniform_range_is_one_rectangle() {
        let selected = range(2, 3, 4, 5);
        assert_eq!(
            coalesce_subresources(selected, |_, _| Some(7)),
            vec![(selected, 7)]
        );
    }

    /// Different states in one mip remain distinct adjacent layer runs.
    #[test]
    fn coalescer_splits_two_states_in_one_mip() {
        assert_eq!(
            coalesce_subresources(range(0, 1, 0, 4), |_, layer| Some(layer / 2)),
            vec![(range(0, 1, 0, 2), 0), (range(0, 1, 2, 2), 1)]
        );
    }

    /// Identical split layer lists merge across adjacent mips.
    #[test]
    fn coalescer_merges_identical_mip_run_lists() {
        assert_eq!(
            coalesce_subresources(range(0, 3, 0, 4), |_, layer| Some(layer / 2)),
            vec![(range(0, 3, 0, 2), 0), (range(0, 3, 2, 2), 1)]
        );
    }

    /// Excluded layers split equal-valued runs without joining across the hole.
    #[test]
    fn coalescer_none_hole_splits_layers() {
        assert_eq!(
            coalesce_subresources(range(0, 2, 0, 3), |_, layer| (layer != 1).then_some(())),
            vec![(range(0, 2, 0, 1), ()), (range(0, 2, 2, 1), ())]
        );
    }

    /// A skipped or differently classified mip prevents nonadjacent runs merging.
    #[test]
    fn coalescer_nonadjacent_identical_mips_stay_separate() {
        for middle in [None, Some(1)] {
            let mut expected = vec![(range(0, 1, 0, 3), 0)];
            if let Some(value) = middle {
                expected.push((range(1, 1, 0, 3), value));
            }
            expected.push((range(2, 1, 0, 3), 0));
            assert_eq!(
                coalesce_subresources(range(0, 3, 0, 3), |mip, _| {
                    if mip == 1 {
                        middle
                    } else {
                        Some(0)
                    }
                }),
                expected
            );
        }
    }

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
