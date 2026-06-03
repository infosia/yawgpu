use std::cell::UnsafeCell;
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::HalBuffer;

use crate::error::*;
use crate::limits::*;

/// Describes buffer descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDescriptor {
    /// Usage.
    pub usage: BufferUsage,
    /// Size.
    pub size: u64,
    /// Mapped at creation.
    pub mapped_at_creation: bool,
}

/// Enumerates map mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapMode {
    /// Read variant.
    Read,
    /// Write variant.
    Write,
}

impl MapMode {
    /// Constructs this object from bits.
    pub fn from_bits(bits: u32) -> Result<Self, &'static str> {
        /// Constant value for read.
        pub(crate) const READ: u32 = 1;
        /// Constant value for write.
        pub(crate) const WRITE: u32 = 2;
        /// Constant value for allowed.
        pub(crate) const ALLOWED: u32 = READ | WRITE;

        if bits & !ALLOWED != 0 {
            return Err("map mode has unsupported bits");
        }
        match bits {
            READ => Ok(Self::Read),
            WRITE => Ok(Self::Write),
            _ => Err("map mode must be exactly Read or Write"),
        }
    }
}

/// Enumerates map async status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapAsyncStatus {
    /// Success variant.
    Success,
    /// Callback cancelled variant.
    CallbackCancelled,
    /// Error variant.
    Error,
    /// Aborted variant.
    Aborted,
}

/// Enumerates queue work done status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueWorkDoneStatus {
    /// Success variant.
    Success,
    /// Callback cancelled variant.
    CallbackCancelled,
    /// Error variant.
    Error,
}

/// Enumerates buffer usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u64);

impl BufferUsage {
    /// Constant value for none.
    pub const NONE: Self = Self(0);
    /// Constant value for map read.
    pub const MAP_READ: Self = Self(1);
    /// Constant value for map write.
    pub const MAP_WRITE: Self = Self(2);
    /// Constant value for copy src.
    pub const COPY_SRC: Self = Self(4);
    /// Constant value for copy dst.
    pub const COPY_DST: Self = Self(8);
    /// Constant value for index.
    pub const INDEX: Self = Self(16);
    /// Constant value for vertex.
    pub const VERTEX: Self = Self(32);
    /// Constant value for uniform.
    pub const UNIFORM: Self = Self(64);
    /// Constant value for storage.
    pub const STORAGE: Self = Self(128);
    /// Constant value for indirect.
    pub const INDIRECT: Self = Self(256);
    /// Constant value for query resolve.
    pub const QUERY_RESOLVE: Self = Self(512);
    const ALL: Self = Self(
        Self::MAP_READ.0
            | Self::MAP_WRITE.0
            | Self::COPY_SRC.0
            | Self::COPY_DST.0
            | Self::INDEX.0
            | Self::VERTEX.0
            | Self::UNIFORM.0
            | Self::STORAGE.0
            | Self::INDIRECT.0
            | Self::QUERY_RESOLVE.0,
    );

    /// Constructs this object from bits retain.
    #[must_use]
    pub fn from_bits_retain(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw usage bitmask.
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Returns whether every bit set in `other` is also set in `self`.
    #[must_use]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Converts a validated WebGPU `BufferUsage` bitfield into the HAL-level
/// struct so the backend layer can see which usage bits the caller declared.
pub(crate) fn hal_buffer_usage(usage: BufferUsage) -> yawgpu_hal::HalBufferUsage {
    yawgpu_hal::HalBufferUsage {
        map_read: usage.contains(BufferUsage::MAP_READ),
        map_write: usage.contains(BufferUsage::MAP_WRITE),
        copy_src: usage.contains(BufferUsage::COPY_SRC),
        copy_dst: usage.contains(BufferUsage::COPY_DST),
        index: usage.contains(BufferUsage::INDEX),
        vertex: usage.contains(BufferUsage::VERTEX),
        uniform: usage.contains(BufferUsage::UNIFORM),
        storage: usage.contains(BufferUsage::STORAGE),
        indirect: usage.contains(BufferUsage::INDIRECT),
        query_resolve: usage.contains(BufferUsage::QUERY_RESOLVE),
    }
}

/// Enumerates buffer map state values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferMapState {
    /// Unmapped variant.
    Unmapped,
    /// Pending variant.
    Pending,
    /// Mapped variant.
    Mapped,
}

/// Stores buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub(crate) inner: Arc<BufferInner>,
}

/// Holds shared state for the buffer handle.
#[derive(Debug)]
pub(crate) struct BufferInner {
    pub(crate) hal: Option<HalBuffer>,
    pub(crate) usage: BufferUsage,
    pub(crate) size: u64,
    pub(crate) host: HostBuffer,
    pub(crate) state: Mutex<BufferState>,
}

/// Tracks the lifecycle state for buffer.
#[derive(Debug)]
pub(crate) struct BufferState {
    pub(crate) map_state: BufferMapState,
    pub(crate) is_error: bool,
    pub(crate) is_destroyed: bool,
    pub(crate) pending_map: Option<PendingMap>,
    pub(crate) active_map: Option<ActiveMap>,
    pub(crate) mapped_ranges: Vec<MappedRange>,
}

/// Stores pending map data used by validation and backend submission.
#[derive(Debug)]
pub(crate) struct PendingMap {
    pub(crate) mode: MapMode,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) outcome: MapAsyncStatus,
}

/// Stores active map data used by validation and backend submission.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ActiveMap {
    pub(crate) mode: MapMode,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MappedRange {
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// Stores host buffer data used by validation and backend submission.
pub(crate) struct HostBuffer {
    pub(crate) bytes: Box<[UnsafeCell<u8>]>,
}

impl HostBuffer {
    /// Creates a new instance.
    pub(crate) fn new(size: u64) -> Self {
        debug_assert!(
            usize::try_from(size).is_ok(),
            "buffer sizes above usize::MAX must be rejected before allocation"
        );
        let len = match usize::try_from(size) {
            Ok(len) => len,
            Err(_) => usize::MAX,
        };
        let bytes = (0..len)
            .map(|_| UnsafeCell::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { bytes }
    }

    /// Returns ptr at.
    pub(crate) fn ptr_at(&self, offset: u64) -> Option<*mut u8> {
        let offset = usize::try_from(offset).ok()?;
        if offset > self.bytes.len() {
            return None;
        }
        // One-past-the-end is valid for zero-sized mapped ranges.
        Some(unsafe { self.bytes.as_ptr().add(offset).cast::<u8>().cast_mut() })
    }

    /// Records a write command.
    pub(crate) fn write(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        let offset = usize::try_from(offset).map_err(|_| "host buffer offset is too large")?;
        let end = offset
            .checked_add(data.len())
            .ok_or("host buffer write range overflows")?;
        if end > self.bytes.len() {
            return Err("host buffer write range exceeds buffer size".to_owned());
        }
        for (cell, byte) in self.bytes[offset..end].iter().zip(data) {
            unsafe {
                *cell.get() = *byte;
            }
        }
        Ok(())
    }

    /// Reads `size` bytes starting at `offset` from the host-side backing store.
    pub(crate) fn read(&self, offset: u64, size: u64) -> Result<Vec<u8>, String> {
        let offset = usize::try_from(offset).map_err(|_| "host buffer offset is too large")?;
        let size = usize::try_from(size).map_err(|_| "host buffer read size is too large")?;
        let end = offset
            .checked_add(size)
            .ok_or("host buffer read range overflows")?;
        if end > self.bytes.len() {
            return Err("host buffer read range exceeds buffer size".to_owned());
        }
        let mut data = Vec::with_capacity(size);
        for cell in &self.bytes[offset..end] {
            data.push(unsafe { *cell.get() });
        }
        Ok(data)
    }
}

impl fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostBuffer")
            .field("len", &self.bytes.len())
            .finish()
    }
}

// Mapped ranges expose raw pointers whose synchronization is governed by the
// WebGPU map/unmap state machine rather than Rust references.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl Buffer {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: BufferDescriptor,
        hal: Option<HalBuffer>,
        is_error: bool,
    ) -> Self {
        let map_state = if descriptor.mapped_at_creation {
            BufferMapState::Mapped
        } else {
            BufferMapState::Unmapped
        };
        let active_map = if descriptor.mapped_at_creation {
            Some(ActiveMap {
                mode: MapMode::Write,
                offset: 0,
                size: descriptor.size,
            })
        } else {
            None
        };
        Self {
            inner: Arc::new(BufferInner {
                hal,
                usage: descriptor.usage,
                size: descriptor.size,
                host: HostBuffer::new(if is_error && !descriptor.mapped_at_creation {
                    0
                } else {
                    descriptor.size
                }),
                state: Mutex::new(BufferState {
                    map_state,
                    is_error,
                    is_destroyed: false,
                    pending_map: None,
                    active_map,
                    mapped_ranges: Vec::new(),
                }),
            }),
        }
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Returns the usage.
    #[must_use]
    pub fn usage(&self) -> BufferUsage {
        self.inner.usage
    }

    /// Converts state into the corresponding yawgpu representation.
    #[must_use]
    pub fn map_state(&self) -> BufferMapState {
        self.inner.state.lock().map_state
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    /// Returns true when this object is destroyed.
    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    /// Returns true when both handles share the same backing object.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Marks any pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn destroy(&self) {
        let mut state = self.inner.state.lock();
        state.is_destroyed = true;
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
        }
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
        state.mapped_ranges.clear();
    }

    /// Marks any pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn unmap(&self) -> Option<DeviceError> {
        let mut state = self.inner.state.lock();
        let active_map = state.active_map;
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
        }
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
        state.mapped_ranges.clear();
        drop(state);

        let active_map = active_map?;
        if active_map.mode != MapMode::Write {
            return None;
        }
        let Some(hal) = &self.inner.hal else {
            return None;
        };
        if hal.mapped_ptr().is_some() {
            return None;
        }
        let data = match self.inner.host.read(active_map.offset, active_map.size) {
            Ok(data) => data,
            Err(message) => return Some(DeviceError::internal(message)),
        };
        hal.write(active_map.offset, &data)
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    /// Begins an asynchronous map of the buffer range, moving it into the pending-map state.
    pub fn begin_map(&self, mode: MapMode, offset: u64, size: u64) -> Result<(), &'static str> {
        let mut state = self.inner.state.lock();
        if state.is_error {
            return Err("cannot map an error buffer");
        }
        if state.is_destroyed {
            return Err("cannot map a destroyed buffer");
        }
        if state.map_state == BufferMapState::Mapped {
            return Err("buffer is already mapped");
        }
        if state.map_state == BufferMapState::Pending {
            return Err("buffer already has a pending map");
        }

        match mode {
            MapMode::Read if !self.inner.usage.contains(BufferUsage::MAP_READ) => {
                return Err("read mapping requires MapRead usage");
            }
            MapMode::Write if !self.inner.usage.contains(BufferUsage::MAP_WRITE) => {
                return Err("write mapping requires MapWrite usage");
            }
            _ => {}
        }

        if !offset.is_multiple_of(8) {
            return Err("map offset must be 8-byte aligned");
        }
        if !size.is_multiple_of(4) {
            return Err("map size must be 4-byte aligned");
        }
        let Some(end) = offset.checked_add(size) else {
            return Err("map range overflows");
        };
        if offset > self.inner.size || end > self.inner.size {
            return Err("map range exceeds buffer size");
        }

        state.map_state = BufferMapState::Pending;
        state.pending_map = Some(PendingMap {
            mode,
            offset,
            size,
            outcome: MapAsyncStatus::Success,
        });
        state.active_map = None;
        state.mapped_ranges.clear();
        Ok(())
    }

    /// Completes the pending map, transitioning the buffer to the mapped state.
    #[must_use]
    pub fn resolve_pending_map(&self) -> MapAsyncStatus {
        self.resolve_pending_map_with_gpu_completion(|| true)
    }

    /// Completes the pending map after ensuring GPU work has completed for read maps.
    #[must_use]
    pub fn resolve_pending_map_with_gpu_completion(
        &self,
        ensure_gpu_complete: impl FnOnce() -> bool,
    ) -> MapAsyncStatus {
        let mut state = self.inner.state.lock();
        let pending = state.pending_map.take();
        let mut outcome = pending
            .as_ref()
            .map(|pending| pending.outcome)
            .unwrap_or(MapAsyncStatus::Aborted);
        if outcome == MapAsyncStatus::Success {
            if let Some(pending) = pending.as_ref() {
                if pending.mode == MapMode::Read {
                    if let Some(hal) = &self.inner.hal {
                        if should_wait_for_read_map(outcome, pending, true) {
                            if !ensure_gpu_complete() {
                                outcome = MapAsyncStatus::Error;
                            } else if hal.mapped_ptr().is_none() {
                                outcome = match hal.read(pending.offset, pending.size) {
                                    Ok(bytes)
                                        if self
                                            .inner
                                            .host
                                            .write(pending.offset, &bytes)
                                            .is_ok() =>
                                    {
                                        MapAsyncStatus::Success
                                    }
                                    _ => MapAsyncStatus::Error,
                                };
                            }
                        }
                    }
                }
            }
        }
        state.map_state = if outcome == MapAsyncStatus::Success {
            state.active_map = pending.map(|pending| ActiveMap {
                mode: pending.mode,
                offset: pending.offset,
                size: pending.size,
            });
            state.mapped_ranges.clear();
            BufferMapState::Mapped
        } else {
            state.active_map = None;
            state.mapped_ranges.clear();
            BufferMapState::Unmapped
        };
        outcome
    }

    /// Marks a pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn abort_pending_map(&self) {
        let mut state = self.inner.state.lock();
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
            state.map_state = BufferMapState::Unmapped;
            state.active_map = None;
            state.mapped_ranges.clear();
        }
    }

    /// Returns mapped range.
    #[must_use]
    pub fn mapped_range(
        &self,
        const_access: bool,
        offset: u64,
        size: Option<u64>,
    ) -> Option<*mut u8> {
        let mut state = self.inner.state.lock();
        if state.is_destroyed || state.map_state != BufferMapState::Mapped {
            return None;
        }
        let active = state.active_map?;
        if !const_access && active.mode == MapMode::Read {
            return None;
        }
        if !offset.is_multiple_of(8) {
            return None;
        }
        let map_end = active.offset.checked_add(active.size)?;
        let size = match size {
            Some(size) => size,
            None => self.inner.size.checked_sub(offset)?,
        };
        if !size.is_multiple_of(4) {
            return None;
        }
        if offset < active.offset || offset > map_end {
            return None;
        }
        let end = offset.checked_add(size)?;
        if end > map_end {
            return None;
        }
        let range = MappedRange { offset, size };
        if state
            .mapped_ranges
            .iter()
            .any(|existing| ranges_overlap(*existing, range))
        {
            return None;
        }
        state.mapped_ranges.push(range);
        drop(state);
        if let Some(mapped_ptr) = self.inner.hal.as_ref().and_then(HalBuffer::mapped_ptr) {
            let offset = usize::try_from(offset).ok()?;
            return Some(unsafe { mapped_ptr.as_ptr().add(offset) });
        }
        self.inner.host.ptr_at(offset)
    }

    /// Applies a queue-side write of `data` at `offset` to the buffer's backing store.
    pub(crate) fn write_from_queue(&self, offset: u64, data: &[u8]) -> Option<DeviceError> {
        let size = match u64::try_from(data.len()) {
            Ok(size) => size,
            Err(_) => {
                return Some(DeviceError::validation("queue write size is too large"));
            }
        };
        if let Err(message) = self.validate_queue_write(offset, size) {
            return Some(DeviceError::validation(message));
        }
        if let Some(hal) = &self.inner.hal {
            if let Err(error) = hal.write(offset, data) {
                return Some(DeviceError::internal(error.to_string()));
            }
        }
        None
    }

    /// Returns the HAL.
    pub fn hal(&self) -> Option<HalBuffer> {
        self.inner.hal.clone()
    }

    /// Validates queue write and returns a descriptive error on failure.
    pub fn validate_queue_write(&self, offset: u64, size: u64) -> Result<(), &'static str> {
        let state = self.inner.state.lock();
        if state.is_error {
            return Err("cannot write to an error buffer");
        }
        if state.is_destroyed {
            return Err("cannot write to a destroyed buffer");
        }
        if state.map_state != BufferMapState::Unmapped {
            return Err("cannot write to a mapped buffer");
        }
        if !self.inner.usage.contains(BufferUsage::COPY_DST) {
            return Err("queue write requires CopyDst usage");
        }
        if !offset.is_multiple_of(4) {
            return Err("queue write offset must be 4-byte aligned");
        }
        if !size.is_multiple_of(4) {
            return Err("queue write size must be 4-byte aligned");
        }
        let Some(end) = offset.checked_add(size) else {
            return Err("queue write range overflows");
        };
        if end > self.inner.size {
            return Err("queue write range exceeds buffer size");
        }
        Ok(())
    }
}

fn ranges_overlap(a: MappedRange, b: MappedRange) -> bool {
    let Some(a_end) = a.offset.checked_add(a.size) else {
        return true;
    };
    let Some(b_end) = b.offset.checked_add(b.size) else {
        return true;
    };
    !(a.offset >= b_end || b.offset >= a_end)
}

fn should_wait_for_read_map(
    outcome: MapAsyncStatus,
    pending: &PendingMap,
    has_live_hal_buffer: bool,
) -> bool {
    outcome == MapAsyncStatus::Success && pending.mode == MapMode::Read && has_live_hal_buffer
}

/// Validates buffer descriptor and returns a descriptive error on failure.
pub(crate) fn validate_buffer_descriptor(
    descriptor: &BufferDescriptor,
    limits: Limits,
) -> Option<&'static str> {
    let usage = descriptor.usage;
    if usage.bits() == 0 {
        return Some("buffer usage must be non-zero");
    }
    if usage.bits() & !BufferUsage::ALL.bits() != 0 {
        return Some("buffer usage contains unknown bits");
    }
    if usage.contains(BufferUsage::MAP_READ) {
        let allowed = (BufferUsage::MAP_READ | BufferUsage::COPY_DST).bits();
        if usage.bits() & !allowed != 0 {
            return Some("MapRead buffers may only combine with CopyDst");
        }
    }
    if usage.contains(BufferUsage::MAP_WRITE) {
        let allowed = (BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC).bits();
        if usage.bits() & !allowed != 0 {
            return Some("MapWrite buffers may only combine with CopySrc");
        }
    }
    if descriptor.size > limits.max_buffer_size {
        return Some("buffer size exceeds device limit");
    }
    if descriptor.mapped_at_creation && !descriptor.size.is_multiple_of(4) {
        return Some("mappedAtCreation buffer size must be 4-byte aligned");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn buffer_usage_from_bits_retain_round_trips_known_and_unknown_bits() {
        let raw = (BufferUsage::MAP_READ | BufferUsage::COPY_DST).bits() | (1_u64 << 40);
        let usage = BufferUsage::from_bits_retain(raw);

        assert_eq!(usage.bits(), raw);
    }

    #[test]
    fn validate_buffer_descriptor_rejects_unknown_usage_bits() {
        let descriptor = BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::from_bits_retain(1_u64 << 40),
            size: 4,
            mapped_at_creation: false,
        };

        assert_eq!(
            validate_buffer_descriptor(&descriptor, Limits::DEFAULT),
            Some("buffer usage contains unknown bits")
        );
    }

    #[test]
    fn hal_buffer_usage_maps_every_bit() {
        let all = BufferUsage::MAP_READ
            | BufferUsage::MAP_WRITE
            | BufferUsage::COPY_SRC
            | BufferUsage::COPY_DST
            | BufferUsage::INDEX
            | BufferUsage::VERTEX
            | BufferUsage::UNIFORM
            | BufferUsage::STORAGE
            | BufferUsage::INDIRECT
            | BufferUsage::QUERY_RESOLVE;
        let usage = hal_buffer_usage(all);

        assert!(usage.map_read);
        assert!(usage.map_write);
        assert!(usage.copy_src);
        assert!(usage.copy_dst);
        assert!(usage.index);
        assert!(usage.vertex);
        assert!(usage.uniform);
        assert!(usage.storage);
        assert!(usage.indirect);
        assert!(usage.query_resolve);

        let empty = hal_buffer_usage(BufferUsage::NONE);
        assert!(!empty.map_read);
        assert!(!empty.map_write);
        assert!(!empty.copy_src);
        assert!(!empty.copy_dst);
        assert!(!empty.index);
        assert!(!empty.vertex);
        assert!(!empty.uniform);
        assert!(!empty.storage);
        assert!(!empty.indirect);
        assert!(!empty.query_resolve);
    }

    #[test]
    fn buffer_accessors_error_same_destroy_hal_and_validate_queue_write() {
        let device = noop_device();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        });
        let clone = buffer.clone();
        let other = noop_buffer(16, BufferUsage::COPY_DST);

        assert_eq!(buffer.size(), 16);
        assert_eq!(buffer.usage(), BufferUsage::COPY_DST);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert!(!buffer.is_error());
        assert!(buffer.same(&clone));
        assert!(!buffer.same(&other));
        assert!(matches!(buffer.hal(), Some(yawgpu_hal::HalBuffer::Noop(_))));
        assert_eq!(buffer.validate_queue_write(0, 4), Ok(()));
        assert_eq!(
            buffer.validate_queue_write(12, 8),
            Err("queue write range exceeds buffer size")
        );

        buffer.destroy();
        buffer.destroy();
        assert_eq!(
            buffer.validate_queue_write(0, 4),
            Err("cannot write to a destroyed buffer")
        );

        device.push_error_scope(ErrorFilter::Validation);
        let error_buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 16,
            mapped_at_creation: false,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid buffer should be scoped");

        assert!(error_buffer.is_error());
        assert_eq!(error.message, "buffer usage must be non-zero");
    }

    #[test]
    fn buffer_map_state_machine_transitions_and_mapped_range_bounds() {
        let mapped = noop_device().create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: true,
        });
        assert_eq!(mapped.map_state(), BufferMapState::Mapped);
        assert_eq!(
            mapped.begin_map(MapMode::Write, 0, 4),
            Err("buffer is already mapped")
        );
        assert_eq!(mapped.unmap(), None);
        assert_eq!(mapped.map_state(), BufferMapState::Unmapped);
        assert_eq!(mapped.unmap(), None);

        let buffer = noop_buffer(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST);
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Success);
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        assert!(buffer.mapped_range(true, 0, Some(8)).is_some());
        assert_eq!(buffer.mapped_range(false, 0, Some(8)), None);
        assert_eq!(buffer.mapped_range(true, 12, Some(8)), None);
        assert_eq!(buffer.mapped_range(true, 0, Some(2)), None);
        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);

        let partial = noop_buffer(64, BufferUsage::MAP_WRITE);
        assert_eq!(partial.begin_map(MapMode::Write, 16, 16), Ok(()));
        assert_eq!(partial.resolve_pending_map(), MapAsyncStatus::Success);
        assert_eq!(partial.mapped_range(false, 16, None), None);
        assert!(partial.mapped_range(false, 16, Some(16)).is_some());
    }

    #[test]
    fn buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted() {
        let buffer = noop_buffer(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST);

        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        buffer.abort_pending_map();
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Aborted);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn resolve_pending_map_gpu_completion_hook_runs_only_for_successful_read() {
        let calls = Arc::new(AtomicUsize::new(0));
        let read_buffer = noop_buffer(8, BufferUsage::MAP_READ);
        assert_eq!(read_buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        let read_calls = Arc::clone(&calls);
        assert_eq!(
            read_buffer.resolve_pending_map_with_gpu_completion(|| {
                read_calls.fetch_add(1, Ordering::Relaxed);
                true
            }),
            MapAsyncStatus::Success
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let write_buffer = noop_buffer(8, BufferUsage::MAP_WRITE);
        assert_eq!(write_buffer.begin_map(MapMode::Write, 0, 8), Ok(()));
        let write_calls = Arc::clone(&calls);
        assert_eq!(
            write_buffer.resolve_pending_map_with_gpu_completion(|| {
                write_calls.fetch_add(1, Ordering::Relaxed);
                true
            }),
            MapAsyncStatus::Success
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let aborted_buffer = noop_buffer(8, BufferUsage::MAP_READ);
        assert_eq!(aborted_buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        aborted_buffer.abort_pending_map();
        let aborted_calls = Arc::clone(&calls);
        assert_eq!(
            aborted_buffer.resolve_pending_map_with_gpu_completion(|| {
                aborted_calls.fetch_add(1, Ordering::Relaxed);
                true
            }),
            MapAsyncStatus::Aborted
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn resolve_pending_map_gpu_completion_failure_returns_error() {
        let buffer = noop_buffer(8, BufferUsage::MAP_READ);

        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        assert_eq!(
            buffer.resolve_pending_map_with_gpu_completion(|| false),
            MapAsyncStatus::Error
        );
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn read_map_wait_decision_depends_on_mode_outcome_and_live_hal_only() {
        let read = PendingMap {
            mode: MapMode::Read,
            offset: 0,
            size: 4,
            outcome: MapAsyncStatus::Success,
        };
        let write = PendingMap {
            mode: MapMode::Write,
            offset: 0,
            size: 4,
            outcome: MapAsyncStatus::Success,
        };

        assert!(should_wait_for_read_map(
            MapAsyncStatus::Success,
            &read,
            true
        ));
        assert!(!should_wait_for_read_map(
            MapAsyncStatus::Success,
            &write,
            true
        ));
        assert!(!should_wait_for_read_map(
            MapAsyncStatus::Aborted,
            &read,
            true
        ));
        assert!(!should_wait_for_read_map(
            MapAsyncStatus::Error,
            &read,
            true
        ));
        assert!(!should_wait_for_read_map(
            MapAsyncStatus::Success,
            &read,
            false
        ));
    }

    #[test]
    fn mapped_ranges_must_be_disjoint_until_unmap_or_remap() {
        let buffer = noop_device().create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE,
            size: 80,
            mapped_at_creation: false,
        });
        buffer
            .begin_map(MapMode::Write, 0, 80)
            .expect("map should begin");
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Success);

        assert!(buffer.mapped_range(false, 16, Some(20)).is_some());
        assert_eq!(buffer.mapped_range(false, 24, Some(0)), None);
        assert!(buffer.mapped_range(false, 40, Some(8)).is_some());

        assert_eq!(buffer.unmap(), None);
        buffer
            .begin_map(MapMode::Write, 0, 80)
            .expect("map should begin after unmap");
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Success);
        assert!(buffer.mapped_range(false, 24, Some(0)).is_some());
    }

    #[test]
    fn invalid_mapped_at_creation_buffer_still_exposes_initial_mapping() {
        let buffer = noop_device().create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 16,
            mapped_at_creation: true,
        });

        assert!(buffer.is_error());
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        assert!(buffer.mapped_range(false, 0, None).is_some());
        assert_eq!(
            buffer.begin_map(MapMode::Write, 0, 4),
            Err("cannot map an error buffer")
        );
    }

    #[test]
    fn map_mode_from_bits_rejects_none_both_and_unsupported_bits() {
        assert_eq!(MapMode::from_bits(1), Ok(MapMode::Read));
        assert_eq!(MapMode::from_bits(2), Ok(MapMode::Write));
        assert_eq!(
            MapMode::from_bits(0),
            Err("map mode must be exactly Read or Write")
        );
        assert_eq!(
            MapMode::from_bits(1 | 2),
            Err("map mode must be exactly Read or Write")
        );
        assert_eq!(MapMode::from_bits(4), Err("map mode has unsupported bits"));
    }
}
