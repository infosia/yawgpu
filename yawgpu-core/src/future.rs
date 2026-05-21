use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pass::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::*;
use crate::error::*;
use crate::extent::*;
use crate::format::*;
use crate::instance::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FutureId(u64);

impl FutureId {
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }

    #[must_use]
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Default)]
pub struct FutureRegistry {
    pub(crate) inner: Mutex<FutureRegistryInner>,
}

#[derive(Debug)]
pub(crate) struct FutureRegistryInner {
    pub(crate) next_id: u64,
    pub(crate) futures: BTreeMap<FutureId, FutureEntry>,
}

impl Default for FutureRegistryInner {
    fn default() -> Self {
        Self {
            next_id: 1,
            futures: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FutureState {
    Pending,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FutureCallbackMode {
    WaitAnyOnly,
    AllowProcessEvents,
    AllowSpontaneous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaitAnyStatus {
    Success,
    TimedOut,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WaitAnyResult {
    pub status: WaitAnyStatus,
    pub completed: Vec<FutureId>,
    pub callbacks_to_fire: Vec<FutureId>,
}

#[derive(Debug)]
pub(crate) struct FutureEntry {
    pub(crate) mode: FutureCallbackMode,
    pub(crate) state: FutureState,
    pub(crate) callback_fired: bool,
}

impl FutureRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn register(&self, mode: FutureCallbackMode) -> FutureId {
        let mut inner = self.inner.lock();
        let id = FutureId(inner.next_id);
        inner.next_id = inner.next_id.saturating_add(1);
        inner.futures.insert(
            id,
            FutureEntry {
                mode,
                state: FutureState::Pending,
                callback_fired: false,
            },
        );
        id
    }

    pub fn complete(&self, id: FutureId) {
        if let Some(entry) = self.inner.lock().futures.get_mut(&id) {
            entry.state = FutureState::Complete;
        }
    }

    #[must_use]
    pub fn process_events(&self) -> Vec<FutureId> {
        let mut inner = self.inner.lock();
        inner
            .futures
            .iter_mut()
            .filter_map(|(id, entry)| {
                let can_fire = entry.state == FutureState::Complete
                    && !entry.callback_fired
                    && matches!(
                        entry.mode,
                        FutureCallbackMode::AllowProcessEvents
                            | FutureCallbackMode::AllowSpontaneous
                    );
                if can_fire {
                    entry.callback_fired = true;
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    #[must_use]
    pub fn wait_any(&self, ids: &[FutureId]) -> WaitAnyResult {
        if ids.is_empty() {
            return WaitAnyResult {
                status: WaitAnyStatus::TimedOut,
                completed: Vec::new(),
                callbacks_to_fire: Vec::new(),
            };
        }

        let mut inner = self.inner.lock();
        let mut completed = Vec::new();
        let mut callbacks_to_fire = Vec::new();

        for id in ids {
            let Some(entry) = inner.futures.get_mut(id) else {
                continue;
            };
            if entry.state == FutureState::Complete {
                completed.push(*id);
                if !entry.callback_fired {
                    entry.callback_fired = true;
                    callbacks_to_fire.push(*id);
                }
            }
        }

        let status = if completed.is_empty() {
            WaitAnyStatus::TimedOut
        } else {
            WaitAnyStatus::Success
        };

        WaitAnyResult {
            status,
            completed,
            callbacks_to_fire,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn future_id_get_and_from_raw_round_trip() {
        assert_eq!(FutureId::from_raw(42).get(), 42);
        assert_eq!(FutureId::from_raw(0).get(), 0);
        assert_eq!(FutureId::from_raw(u64::MAX).get(), u64::MAX);
    }

    #[test]
    fn future_registry_process_events_respects_callback_mode() {
        let registry = FutureRegistry::new();
        let first = registry.register(FutureCallbackMode::WaitAnyOnly);
        let second = registry.register(FutureCallbackMode::AllowProcessEvents);
        registry.complete(first);
        registry.complete(second);

        assert_eq!(registry.process_events(), vec![second]);
        assert!(registry.process_events().is_empty());

        let result = registry.wait_any(&[first, second]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert_eq!(result.callbacks_to_fire, vec![first]);

        let result = registry.wait_any(&[first, second]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert!(result.callbacks_to_fire.is_empty());
    }
}
