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

#[cfg(test)]
type WritebackTestHook = (std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>);

/// Holds shared state for the buffer handle.
#[derive(Debug)]
pub(crate) struct BufferInner {
    pub(crate) hal: Option<HalBuffer>,
    pub(crate) usage: BufferUsage,
    pub(crate) size: u64,
    pub(crate) host: HostBuffer,
    pub(crate) state: Mutex<BufferState>,
    #[cfg(test)]
    writeback_test_hook: Mutex<Option<WritebackTestHook>>,
}

/// Tracks the lifecycle state for buffer.
#[derive(Debug)]
pub(crate) struct BufferState {
    pub(crate) map_state: BufferMapState,
    pub(crate) is_error: bool,
    pub(crate) is_destroyed: bool,
    pub(crate) pending_map: Option<PendingMap>,
    next_map_serial: u64,
    pub(crate) active_map: Option<ActiveMap>,
    pub(crate) mapped_ranges: Vec<MappedRange>,
}

/// Stores pending map data used by validation and backend submission.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingMap {
    pub(crate) serial: u64,
    pub(crate) mode: MapMode,
    pub(crate) offset: u64,
    pub(crate) size: u64,
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
    bytes: Mutex<Box<[UnsafeCell<u8>]>>,
}

impl HostBuffer {
    /// Creates a zero-length host buffer used for error buffers and non-mapped buffers.
    pub(crate) fn empty() -> Self {
        Self {
            bytes: Mutex::new(Box::new([])),
        }
    }

    /// Attempts to allocate `size` bytes of host storage for a mapped buffer.
    ///
    /// Returns `Err(())` if the size cannot be represented as `usize` or if
    /// the system allocator cannot satisfy the request (out-of-memory).
    /// Callers must treat allocation failure as an out-of-memory condition and
    /// return an error buffer with no host storage rather than aborting.
    pub(crate) fn try_new(size: u64) -> Result<Self, ()> {
        let len = usize::try_from(size).map_err(|_| ())?;
        let mut v: Vec<UnsafeCell<u8>> = Vec::new();
        v.try_reserve_exact(len).map_err(|_| ())?;
        // `resize` after a successful reserve does not reallocate.
        v.resize_with(len, || UnsafeCell::new(0));
        Ok(Self {
            bytes: Mutex::new(v.into_boxed_slice()),
        })
    }

    /// Returns ptr at.
    pub(crate) fn ptr_at(&self, offset: u64) -> Option<*mut u8> {
        let offset = usize::try_from(offset).ok()?;
        let bytes = self.bytes.lock();
        if offset > bytes.len() {
            return None;
        }
        // One-past-the-end is valid for zero-sized mapped ranges.
        Some(unsafe { bytes.as_ptr().add(offset).cast::<u8>().cast_mut() })
    }

    /// Records a write command.
    pub(crate) fn write(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        let offset = usize::try_from(offset).map_err(|_| "host buffer offset is too large")?;
        let end = offset
            .checked_add(data.len())
            .ok_or("host buffer write range overflows")?;
        let bytes = self.bytes.lock();
        if end > bytes.len() {
            return Err("host buffer write range exceeds buffer size".to_owned());
        }
        for (cell, byte) in bytes[offset..end].iter().zip(data) {
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
        let bytes = self.bytes.lock();
        if end > bytes.len() {
            return Err("host buffer read range exceeds buffer size".to_owned());
        }
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(size)
            .map_err(|_| "host buffer read allocation failed".to_owned())?;
        for cell in &bytes[offset..end] {
            data.push(unsafe { *cell.get() });
        }
        Ok(data)
    }
}

impl fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostBuffer")
            .field("len", &self.bytes.lock().len())
            .finish()
    }
}

impl Buffer {
    /// Creates an error buffer, optionally with a fallible host mapping.
    ///
    /// When `descriptor.mapped_at_creation` is true the constructor attempts
    /// a fallible host allocation via `HostBuffer::try_new`.  On success the
    /// buffer starts in the `Mapped` state with a full-range `ActiveMap`
    /// (write mode), mirroring the spec requirement that "a buffer created
    /// with mappedAtCreation=true is mapped at creation even if it is an
    /// error buffer" (WebGPU § 6.4).  On OOM the buffer falls back to
    /// `HostBuffer::empty()` and `Unmapped` — no abort is possible because
    /// the allocation goes through `try_new` only (F-073).
    ///
    /// When `mapped_at_creation` is false the buffer is always `Unmapped`
    /// with no host storage.
    pub(crate) fn new(
        descriptor: BufferDescriptor,
        hal: Option<HalBuffer>,
        is_error: bool,
    ) -> Self {
        debug_assert!(
            is_error,
            "Buffer::new must only be called for error buffers; use Buffer::try_new for valid buffers"
        );

        // For mappedAtCreation error buffers attempt a small, fallible host
        // allocation so that getMappedRange can return a writable scratch
        // pointer.  Giant sizes (e.g. the 9 PB OOM case) fail try_new and
        // silently fall back to empty / Unmapped — the process cannot abort.
        let (host, map_state, active_map) = if descriptor.mapped_at_creation {
            match HostBuffer::try_new(descriptor.size) {
                Ok(host) => (
                    host,
                    BufferMapState::Mapped,
                    Some(ActiveMap {
                        mode: MapMode::Write,
                        offset: 0,
                        size: descriptor.size,
                    }),
                ),
                Err(()) => (HostBuffer::empty(), BufferMapState::Unmapped, None),
            }
        } else {
            (HostBuffer::empty(), BufferMapState::Unmapped, None)
        };

        Self {
            inner: Arc::new(BufferInner {
                hal,
                usage: descriptor.usage,
                size: descriptor.size,
                host,
                #[cfg(test)]
                writeback_test_hook: Mutex::new(None),
                state: Mutex::new(BufferState {
                    map_state,
                    is_error: true,
                    is_destroyed: false,
                    pending_map: None,
                    next_map_serial: 0,
                    active_map,
                    mapped_ranges: Vec::new(),
                }),
            }),
        }
    }

    /// Attempts to create a valid buffer, allocating host storage only for fallback maps.
    ///
    /// Returns `Err(hal)` if the host-side backing allocation fails
    /// (out-of-memory).  The caller is responsible for dispatching an
    /// `OutOfMemory` error through the device error sink and creating an
    /// error buffer via `Buffer::new` (F-073).
    pub(crate) fn try_new(
        descriptor: BufferDescriptor,
        hal: Option<HalBuffer>,
    ) -> Result<Self, Option<HalBuffer>> {
        // Direct HAL mappings already provide storage. Other buffers need a
        // host copy only if a map can expose it. Keep allocation fallible (F-073).
        let needs_host = (descriptor.mapped_at_creation
            || (descriptor.usage.contains(BufferUsage::MAP_READ)
                || descriptor.usage.contains(BufferUsage::MAP_WRITE)))
            && hal.as_ref().and_then(HalBuffer::mapped_ptr).is_none();
        let host = if needs_host {
            HostBuffer::try_new(descriptor.size).map_err(|_| hal.clone())?
        } else {
            HostBuffer::empty()
        };

        let (map_state, active_map) = if descriptor.mapped_at_creation {
            (
                BufferMapState::Mapped,
                Some(ActiveMap {
                    mode: MapMode::Write,
                    offset: 0,
                    size: descriptor.size,
                }),
            )
        } else {
            (BufferMapState::Unmapped, None)
        };

        Ok(Self {
            inner: Arc::new(BufferInner {
                hal,
                usage: descriptor.usage,
                size: descriptor.size,
                host,
                #[cfg(test)]
                writeback_test_hook: Mutex::new(None),
                state: Mutex::new(BufferState {
                    map_state,
                    is_error: false,
                    is_destroyed: false,
                    pending_map: None,
                    next_map_serial: 0,
                    active_map,
                    mapped_ranges: Vec::new(),
                }),
            }),
        })
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

    /// Cancels any pending map and clears the mapped state.
    pub fn destroy(&self) {
        let mut state = self.inner.state.lock();
        state.is_destroyed = true;
        state.pending_map = None;
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
        state.mapped_ranges.clear();
        *self.inner.host.bytes.lock() = Box::new([]);
    }

    /// Cancels any pending map and clears the mapped state.
    pub fn unmap(&self) -> Option<DeviceError> {
        let mut state = self.inner.state.lock();
        let active_map = state.active_map;
        // Keep the state locked through host write-back before publishing Unmapped.
        let result = self.flush_host_map(active_map);
        state.pending_map = None;
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
        state.mapped_ranges.clear();
        drop(state);
        if !self.usage().contains(BufferUsage::MAP_READ)
            && !self.usage().contains(BufferUsage::MAP_WRITE)
        {
            *self.inner.host.bytes.lock() = Box::new([]);
        }
        result
    }

    fn flush_host_map(&self, active_map: Option<ActiveMap>) -> Option<DeviceError> {
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
        #[cfg(test)]
        if let Some((reached, resume)) = self.inner.writeback_test_hook.lock().take() {
            reached.send(()).unwrap();
            resume.recv().unwrap();
        }
        let data = match self.inner.host.read(active_map.offset, active_map.size) {
            Ok(data) => data,
            Err(message) => return Some(DeviceError::internal(message)),
        };
        hal.write(active_map.offset, &data)
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    /// Begins an asynchronous map and returns its serial for completion.
    pub fn begin_map(&self, mode: MapMode, offset: u64, size: u64) -> Result<u64, &'static str> {
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

        let serial = state
            .next_map_serial
            .checked_add(1)
            .ok_or("map serial exhausted")?;
        state.next_map_serial = serial;
        state.map_state = BufferMapState::Pending;
        state.pending_map = Some(PendingMap {
            serial,
            mode,
            offset,
            size,
        });
        state.active_map = None;
        state.mapped_ranges.clear();
        Ok(serial)
    }

    /// Completes the identified pending map; stale serials abort without changing state.
    #[must_use]
    pub fn resolve_pending_map(&self, serial: u64) -> MapAsyncStatus {
        self.resolve_pending_map_with_gpu_completion(serial, || true)
    }

    /// Completes the identified map after GPU completion for reads.
    /// Stale serials abort without calling the hook or changing buffer state.
    #[must_use]
    pub fn resolve_pending_map_with_gpu_completion(
        &self,
        serial: u64,
        ensure_gpu_complete: impl FnOnce() -> bool,
    ) -> MapAsyncStatus {
        let pending = {
            let state = self.inner.state.lock();
            match state.pending_map {
                Some(pending) if pending.serial == serial => pending,
                _ => return MapAsyncStatus::Aborted,
            }
        };
        // GPU completion may take the submission lock or reenter buffer APIs.
        // Never call it while holding buffer state.
        let complete =
            pending.mode != MapMode::Read || self.inner.hal.is_none() || ensure_gpu_complete();
        let mut state = self.inner.state.lock();
        if state
            .pending_map
            .as_ref()
            .is_none_or(|pending| pending.serial != serial)
        {
            return MapAsyncStatus::Aborted;
        }
        let pending = state.pending_map.take();
        let mut outcome = if complete {
            MapAsyncStatus::Success
        } else {
            MapAsyncStatus::Error
        };
        if let Some(pending) = pending.as_ref() {
            if should_wait_for_read_map(outcome, pending, self.inner.hal.is_some()) {
                if let Some(hal) = self
                    .inner
                    .hal
                    .as_ref()
                    .filter(|hal| hal.mapped_ptr().is_none())
                {
                    outcome = match hal.read(pending.offset, pending.size) {
                        Ok(bytes) if self.inner.host.write(pending.offset, &bytes).is_ok() => {
                            MapAsyncStatus::Success
                        }
                        _ => MapAsyncStatus::Error,
                    };
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

    /// Cancels any pending map when the owning handle is released.
    pub fn abort_pending_map(&self) {
        self.cancel_pending_map(None);
    }

    /// Cancels the identified pending map; stale serials leave current state unchanged.
    pub fn abort_pending_map_with_serial(&self, serial: u64) {
        self.cancel_pending_map(Some(serial));
    }

    fn cancel_pending_map(&self, serial: Option<u64>) {
        let mut state = self.inner.state.lock();
        if state
            .pending_map
            .as_ref()
            .is_some_and(|pending| serial.is_none_or(|serial| pending.serial == serial))
        {
            state.pending_map = None;
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
        let range = self.validate_mapped_range(&state, const_access, offset, size)?;
        if state
            .mapped_ranges
            .iter()
            .any(|existing| ranges_overlap(*existing, range))
        {
            return None;
        }
        state.mapped_ranges.push(range);
        self.mapped_ptr_at(&state, range.offset)
    }

    /// Resolves a mapped-range pointer without registering the range.
    #[must_use]
    pub fn mapped_range_for_copy(
        &self,
        const_access: bool,
        offset: u64,
        size: u64,
    ) -> Option<*mut u8> {
        let state = self.inner.state.lock();
        let range = self.validate_mapped_range(&state, const_access, offset, Some(size))?;
        self.mapped_ptr_at(&state, range.offset)
    }

    fn validate_mapped_range(
        &self,
        state: &BufferState,
        const_access: bool,
        offset: u64,
        size: Option<u64>,
    ) -> Option<MappedRange> {
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
        Some(MappedRange { offset, size })
    }

    fn mapped_ptr_at(&self, state: &BufferState, offset: u64) -> Option<*mut u8> {
        if state.is_destroyed || state.map_state != BufferMapState::Mapped {
            return None;
        }
        if let Some(mapped_ptr) = self.inner.hal.as_ref().and_then(HalBuffer::mapped_ptr) {
            let offset = usize::try_from(offset).ok()?;
            return Some(unsafe { mapped_ptr.as_ptr().add(offset) });
        }
        self.inner.host.ptr_at(offset)
    }

    /// Returns the HAL.
    pub fn hal(&self) -> Option<HalBuffer> {
        self.inner.hal.clone()
    }

    /// Validates queue write and returns a descriptive error on failure.
    pub fn validate_queue_write(&self, offset: u64, size: u64) -> Result<(), &'static str> {
        validate_buffer_write_or_clear(
            self,
            offset,
            size,
            BufferWriteOrClearValidation {
                error_buffer: "cannot write to an error buffer",
                destroyed_buffer: Some("cannot write to a destroyed buffer"),
                mapped_buffer: Some("cannot write to a mapped buffer"),
                missing_copy_dst: "queue write requires CopyDst usage",
                size_mode: BufferWriteSizeMode::Exact,
                offset_alignment: "queue write offset must be 4-byte aligned",
                size_alignment: "queue write size must be 4-byte aligned",
                range_overflows: "queue write range overflows",
                range_exceeds: "queue write range exceeds buffer size",
            },
        )
    }
}

/// Configures shared buffer write/clear validation messages and optional checks.
pub(crate) struct BufferWriteOrClearValidation {
    pub(crate) error_buffer: &'static str,
    pub(crate) destroyed_buffer: Option<&'static str>,
    pub(crate) mapped_buffer: Option<&'static str>,
    pub(crate) missing_copy_dst: &'static str,
    pub(crate) size_mode: BufferWriteSizeMode,
    pub(crate) offset_alignment: &'static str,
    pub(crate) size_alignment: &'static str,
    pub(crate) range_overflows: &'static str,
    pub(crate) range_exceeds: &'static str,
}

/// Defines whether `size` is exact or `u64::MAX` means "to buffer end".
#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferWriteSizeMode {
    Exact,
    ResolveToEnd {
        offset_exceeds_message: &'static str,
    },
}

/// Validates buffer writes and clears while preserving caller-specific messages.
pub(crate) fn validate_buffer_write_or_clear(
    buffer: &Buffer,
    offset: u64,
    size: u64,
    options: BufferWriteOrClearValidation,
) -> Result<(), &'static str> {
    let state = buffer.inner.state.lock();
    if state.is_error {
        return Err(options.error_buffer);
    }
    if let Some(message) = options.destroyed_buffer {
        if state.is_destroyed {
            return Err(message);
        }
    }
    if let Some(message) = options.mapped_buffer {
        if state.map_state != BufferMapState::Unmapped {
            return Err(message);
        }
    }
    drop(state);
    if !buffer.inner.usage.contains(BufferUsage::COPY_DST) {
        return Err(options.missing_copy_dst);
    }
    let resolved_size = match options.size_mode {
        BufferWriteSizeMode::Exact => size,
        BufferWriteSizeMode::ResolveToEnd {
            offset_exceeds_message,
        } => {
            if offset > buffer.inner.size {
                return Err(offset_exceeds_message);
            }
            if size == u64::MAX {
                buffer
                    .inner
                    .size
                    .checked_sub(offset)
                    .ok_or(offset_exceeds_message)?
            } else {
                size
            }
        }
    };
    if !offset.is_multiple_of(4) {
        return Err(options.offset_alignment);
    }
    if !resolved_size.is_multiple_of(4) {
        return Err(options.size_alignment);
    }
    let Some(end) = offset.checked_add(resolved_size) else {
        return Err(options.range_overflows);
    };
    if offset > buffer.inner.size || end > buffer.inner.size {
        return Err(options.range_exceeds);
    }
    Ok(())
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
    #[test]
    fn audit_map_completion_wait_runs_without_buffer_state_lock() {
        let device = noop_device();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        });
        let serial = buffer.begin_map(MapMode::Read, 0, 4).unwrap();
        assert_eq!(
            buffer.resolve_pending_map_with_gpu_completion(serial, || buffer
                .inner
                .state
                .try_lock()
                .is_some()),
            MapAsyncStatus::Success
        );
    }

    #[test]
    fn audit_c8_unmap_holds_state_until_host_writeback_finishes() {
        let device = noop_device();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 4,
            mapped_at_creation: true,
        });
        let (reached_tx, reached_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        *buffer.inner.writeback_test_hook.lock() = Some((reached_tx, resume_rx));
        let other = buffer.clone();
        let worker = std::thread::spawn(move || other.unmap());
        reached_rx.recv().unwrap();
        let held = buffer.inner.state.try_lock().is_none();
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), None);
        assert!(
            held,
            "Unmapped was published while host writeback was blocked"
        );
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

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
    fn validate_buffer_write_or_clear_preserves_queue_state_order() {
        let buffer = noop_buffer(16, BufferUsage::COPY_DST);
        buffer.destroy();

        assert_eq!(
            validate_buffer_write_or_clear(
                &buffer,
                1,
                2,
                BufferWriteOrClearValidation {
                    error_buffer: "error",
                    destroyed_buffer: Some("destroyed"),
                    mapped_buffer: Some("mapped"),
                    missing_copy_dst: "usage",
                    size_mode: BufferWriteSizeMode::Exact,
                    offset_alignment: "offset",
                    size_alignment: "size",
                    range_overflows: "queue write range overflows",
                    range_exceeds: "queue write range exceeds buffer size",
                },
            ),
            Err("destroyed")
        );
    }

    #[test]
    fn validate_buffer_write_or_clear_handles_to_end_alignment_and_range() {
        let buffer = noop_buffer(18, BufferUsage::COPY_DST);

        assert_eq!(
            validate_buffer_write_or_clear(
                &buffer,
                16,
                u64::MAX,
                BufferWriteOrClearValidation {
                    error_buffer: "error",
                    destroyed_buffer: None,
                    mapped_buffer: None,
                    missing_copy_dst: "usage",
                    size_mode: BufferWriteSizeMode::ResolveToEnd {
                        offset_exceeds_message: "offset exceeds",
                    },
                    offset_alignment: "offset align",
                    size_alignment: "size align",
                    range_overflows: "clear buffer range overflows",
                    range_exceeds: "clear buffer range exceeds buffer size",
                },
            ),
            Err("size align")
        );
        assert_eq!(
            validate_buffer_write_or_clear(
                &buffer,
                20,
                u64::MAX,
                BufferWriteOrClearValidation {
                    error_buffer: "error",
                    destroyed_buffer: None,
                    mapped_buffer: None,
                    missing_copy_dst: "usage",
                    size_mode: BufferWriteSizeMode::ResolveToEnd {
                        offset_exceeds_message: "offset exceeds",
                    },
                    offset_alignment: "offset align",
                    size_alignment: "size align",
                    range_overflows: "clear buffer range overflows",
                    range_exceeds: "clear buffer range exceeds buffer size",
                },
            ),
            Err("offset exceeds")
        );
    }

    #[test]
    fn stale_map_serial_cannot_consume_or_unmap_a_new_request() {
        for complete_new_first in [false, true] {
            let buffer = noop_buffer(16, BufferUsage::MAP_READ);
            let first = buffer.begin_map(MapMode::Read, 0, 8).unwrap();
            assert_eq!(buffer.unmap(), None);
            assert!(buffer.inner.state.lock().pending_map.is_none());
            let second = buffer.begin_map(MapMode::Read, 8, 8).unwrap();
            assert_ne!(first, second);
            buffer.abort_pending_map_with_serial(first);
            assert_eq!(buffer.map_state(), BufferMapState::Pending);
            if complete_new_first {
                assert_eq!(buffer.resolve_pending_map(second), MapAsyncStatus::Success);
            }
            assert_eq!(
                buffer.resolve_pending_map_with_gpu_completion(first, || {
                    panic!("a stale map must not wait for or read GPU data")
                }),
                MapAsyncStatus::Aborted
            );
            assert_eq!(
                buffer.map_state(),
                if complete_new_first {
                    BufferMapState::Mapped
                } else {
                    BufferMapState::Pending
                }
            );
            if !complete_new_first {
                assert_eq!(buffer.resolve_pending_map(second), MapAsyncStatus::Success);
            }
            assert_eq!(buffer.resolve_pending_map(first), MapAsyncStatus::Aborted);
            assert_eq!(buffer.map_state(), BufferMapState::Mapped);
            assert!(buffer.mapped_range(true, 8, Some(8)).is_some());
        }
    }

    #[test]
    fn abort_pending_map_with_serial_cancels_only_its_request() {
        let buffer = noop_buffer(8, BufferUsage::MAP_WRITE);
        let first = buffer.begin_map(MapMode::Write, 0, 8).unwrap();
        buffer.abort_pending_map_with_serial(first + 1);
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        buffer.abort_pending_map_with_serial(first);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert!(buffer.inner.state.lock().pending_map.is_none());
        assert_eq!(buffer.resolve_pending_map(first), MapAsyncStatus::Aborted);
    }

    #[test]
    fn pending_map_completion_succeeds_once_and_repeated_completion_aborts() {
        for (usage, mode) in [
            (BufferUsage::MAP_READ, MapMode::Read),
            (BufferUsage::MAP_WRITE, MapMode::Write),
        ] {
            let buffer = noop_buffer(16, usage);
            let serial = buffer.begin_map(mode, 0, 16).unwrap();
            assert_eq!(buffer.resolve_pending_map(serial), MapAsyncStatus::Success);
            assert_eq!(buffer.resolve_pending_map(serial), MapAsyncStatus::Aborted);
            assert_eq!(buffer.map_state(), BufferMapState::Mapped);
            assert!(buffer.mapped_range(true, 0, Some(16)).is_some());
        }
    }

    #[test]
    fn destroy_frees_mapped_at_creation_scratch_and_unmapped_pointer_is_absent() {
        let device = noop_device();
        for destroy in [true, false] {
            let buffer = device.create_buffer(BufferDescriptor {
                usage: BufferUsage::COPY_SRC,
                size: 16,
                mapped_at_creation: true,
            });
            assert_eq!(buffer.inner.host.bytes.lock().len(), 16);
            assert!(buffer
                .mapped_ptr_at(&buffer.inner.state.lock(), 0)
                .is_some());
            if destroy {
                buffer.destroy();
            } else {
                assert_eq!(buffer.unmap(), None);
            }
            assert!(buffer.inner.host.bytes.lock().is_empty());
            assert_eq!(buffer.mapped_ptr_at(&buffer.inner.state.lock(), 0), None);
            assert_eq!(buffer.mapped_range(false, 0, Some(0)), None);
            assert_eq!(buffer.mapped_range_for_copy(false, 0, 0), None);
        }
    }

    #[test]
    fn destroy_clears_pending_map_and_old_serial_aborts() {
        let buffer = noop_buffer(8, BufferUsage::MAP_WRITE);
        let serial = buffer.begin_map(MapMode::Write, 0, 8).unwrap();
        buffer.destroy();
        assert!(buffer.inner.state.lock().pending_map.is_none());
        assert_eq!(buffer.resolve_pending_map(serial), MapAsyncStatus::Aborted);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn host_storage_is_only_allocated_for_fallback_maps() {
        // No HAL allocation is needed to prove even an unrepresentably large
        // unmappable buffer never allocates a host copy.
        let storage = Buffer::try_new(
            BufferDescriptor {
                usage: BufferUsage::STORAGE,
                size: u64::MAX,
                mapped_at_creation: false,
            },
            None,
        )
        .unwrap();
        assert!(storage.inner.host.bytes.lock().is_empty());
        let storage = noop_buffer(1024 * 1024, BufferUsage::STORAGE);
        assert!(storage.inner.host.bytes.lock().is_empty());
        for usage in [BufferUsage::MAP_READ, BufferUsage::MAP_WRITE] {
            let buffer = noop_buffer(16, usage);
            assert_eq!(buffer.inner.host.bytes.lock().len(), 16);
        }
        let mapped = noop_device().create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 16,
            mapped_at_creation: true,
        });
        assert_eq!(mapped.inner.host.bytes.lock().len(), 16);
        let ptr = mapped.mapped_range(false, 0, Some(16)).unwrap();
        unsafe {
            *ptr = 42;
        }
        assert_eq!(mapped.unmap(), None);
        assert!(mapped.inner.host.bytes.lock().is_empty());
        assert_eq!(mapped.hal().unwrap().read(0, 1).unwrap(), vec![42]);
        assert!(Buffer::try_new(
            BufferDescriptor {
                usage: BufferUsage::MAP_READ,
                size: u64::MAX,
                mapped_at_creation: false,
            },
            None
        )
        .is_err());
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
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(1));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        assert_eq!(buffer.resolve_pending_map(1), MapAsyncStatus::Success);
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        assert!(buffer.mapped_range(true, 0, Some(8)).is_some());
        assert_eq!(buffer.mapped_range(false, 0, Some(8)), None);
        assert_eq!(buffer.mapped_range(true, 12, Some(8)), None);
        assert_eq!(buffer.mapped_range(true, 0, Some(2)), None);
        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);

        let partial = noop_buffer(64, BufferUsage::MAP_WRITE);
        assert_eq!(partial.begin_map(MapMode::Write, 16, 16), Ok(1));
        assert_eq!(partial.resolve_pending_map(1), MapAsyncStatus::Success);
        assert_eq!(partial.mapped_range(false, 16, None), None);
        assert!(partial.mapped_range(false, 16, Some(16)).is_some());
    }

    #[test]
    fn buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted() {
        let buffer = noop_buffer(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST);

        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(1));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        buffer.abort_pending_map();
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert_eq!(buffer.resolve_pending_map(1), MapAsyncStatus::Aborted);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn resolve_pending_map_gpu_completion_hook_runs_only_for_successful_read() {
        let calls = Arc::new(AtomicUsize::new(0));
        let read_buffer = noop_buffer(8, BufferUsage::MAP_READ);
        assert_eq!(read_buffer.begin_map(MapMode::Read, 0, 8), Ok(1));
        let read_calls = Arc::clone(&calls);
        assert_eq!(
            read_buffer.resolve_pending_map_with_gpu_completion(1, || {
                read_calls.fetch_add(1, Ordering::Relaxed);
                true
            }),
            MapAsyncStatus::Success
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let write_buffer = noop_buffer(8, BufferUsage::MAP_WRITE);
        assert_eq!(write_buffer.begin_map(MapMode::Write, 0, 8), Ok(1));
        let write_calls = Arc::clone(&calls);
        assert_eq!(
            write_buffer.resolve_pending_map_with_gpu_completion(1, || {
                write_calls.fetch_add(1, Ordering::Relaxed);
                true
            }),
            MapAsyncStatus::Success
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let aborted_buffer = noop_buffer(8, BufferUsage::MAP_READ);
        assert_eq!(aborted_buffer.begin_map(MapMode::Read, 0, 8), Ok(1));
        aborted_buffer.abort_pending_map();
        let aborted_calls = Arc::clone(&calls);
        assert_eq!(
            aborted_buffer.resolve_pending_map_with_gpu_completion(1, || {
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

        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(1));
        assert_eq!(
            buffer.resolve_pending_map_with_gpu_completion(1, || false),
            MapAsyncStatus::Error
        );
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn read_map_wait_decision_depends_on_mode_outcome_and_live_hal_only() {
        let read = PendingMap {
            serial: 1,
            mode: MapMode::Read,
            offset: 0,
            size: 4,
        };
        let write = PendingMap {
            serial: 1,
            mode: MapMode::Write,
            offset: 0,
            size: 4,
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
        assert_eq!(buffer.resolve_pending_map(1), MapAsyncStatus::Success);

        assert!(buffer.mapped_range(false, 16, Some(20)).is_some());
        assert_eq!(buffer.mapped_range(false, 24, Some(0)), None);
        assert!(buffer.mapped_range(false, 40, Some(8)).is_some());

        assert_eq!(buffer.unmap(), None);
        buffer
            .begin_map(MapMode::Write, 0, 80)
            .expect("map should begin after unmap");
        assert_eq!(buffer.resolve_pending_map(2), MapAsyncStatus::Success);
        assert!(buffer.mapped_range(false, 24, Some(0)).is_some());
    }

    #[test]
    fn mapped_range_for_copy_is_stateless_and_shares_mapped_range_validation() {
        let buffer = noop_buffer(32, BufferUsage::MAP_WRITE);
        assert_eq!(buffer.mapped_range_for_copy(false, 0, 4), None);

        assert_eq!(buffer.begin_map(MapMode::Write, 0, 32), Ok(1));
        assert_eq!(buffer.resolve_pending_map(1), MapAsyncStatus::Success);
        assert!(buffer.mapped_range_for_copy(false, 0, 4).is_some());
        assert!(buffer.mapped_range_for_copy(false, 0, 4).is_some());
        assert!(buffer.mapped_range(false, 0, Some(4)).is_some());
        assert_eq!(buffer.mapped_range_for_copy(false, 4, 4), None);
        assert_eq!(buffer.mapped_range_for_copy(false, 8, 2), None);
        assert_eq!(buffer.mapped_range_for_copy(false, 32, 4), None);

        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.mapped_range_for_copy(false, 0, 4), None);

        let read_buffer = noop_buffer(16, BufferUsage::MAP_READ);
        assert_eq!(read_buffer.begin_map(MapMode::Read, 0, 16), Ok(1));
        assert_eq!(read_buffer.resolve_pending_map(1), MapAsyncStatus::Success);
        assert!(read_buffer.mapped_range_for_copy(true, 0, 4).is_some());
        assert_eq!(read_buffer.mapped_range_for_copy(false, 0, 4), None);
        read_buffer.destroy();
        assert_eq!(read_buffer.mapped_range_for_copy(true, 0, 4), None);
    }

    #[test]
    fn invalid_mapped_at_creation_buffer_still_exposes_initial_mapping() {
        // WebGPU § 6.4: a buffer created with mappedAtCreation=true must start
        // in the 'mapped' state even if it is an error buffer (invalid usage,
        // etc.).  The CTS usageType="invalid" case exercises exactly this path:
        // usage=NONE (validation error), size=8, mappedAtCreation=true.
        //
        // F-073 caveat: the allocation is fallible (try_new) so that a genuine
        // OOM (e.g. 9 PB) still falls back to Unmapped without aborting.
        let device = noop_device();
        device.push_error_scope(ErrorFilter::Validation);
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 8,
            mapped_at_creation: true,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid usage must produce a validation error");

        assert_eq!(error.message, "buffer usage must be non-zero");
        assert!(buffer.is_error());
        // mapState must be 'mapped' even for the error buffer.
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        // getMappedRange must return a valid (non-null) writable pointer.
        assert!(
            buffer.mapped_range(false, 0, Some(8)).is_some(),
            "getMappedRange must return a writable pointer for a small error buffer with mappedAtCreation"
        );
        // Unmap transitions to Unmapped.
        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        // MapAsync is still rejected on error buffers.
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

    #[test]
    fn oom_mapped_at_creation_does_not_abort_and_returns_no_mapping() {
        // F-073: wgpuDeviceCreateBuffer with mappedAtCreation=true and a size
        // that exceeds maxBufferSize must not abort the process.  The buffer
        // fails validation (B4: size > maxBufferSize), is_error becomes true,
        // and getMappedRange must return null (no host allocation is attempted).
        // kMaxSafeMultipleOf8 from the CTS: Number.MAX_SAFE_INTEGER - 7.
        const MAX_SAFE_MULTIPLE_OF_8: u64 = 9_007_199_254_740_984;

        let device = noop_device();
        device.push_error_scope(ErrorFilter::Validation);
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            size: MAX_SAFE_MULTIPLE_OF_8,
            mapped_at_creation: true,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("huge buffer must produce a validation error");

        // Validation error must be raised (B4: size exceeds maxBufferSize).
        assert_eq!(error.kind, ErrorKind::Validation);
        assert_eq!(error.message, "buffer size exceeds device limit");

        // The buffer must be an error buffer with no mapping — no abort.
        assert!(buffer.is_error());
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        // getMappedRange must return null (None in Rust API).
        assert_eq!(buffer.mapped_range(false, 0, None), None);
        assert_eq!(buffer.mapped_range(false, 0, Some(0)), None);
    }

    #[test]
    fn oom_mapped_at_creation_works_for_all_buffer_usages() {
        // F-073: verify that no usage combination + huge size can abort the process.
        const HUGE_SIZE: u64 = 9_007_199_254_740_984;
        let usages = [
            BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            BufferUsage::COPY_SRC,
            BufferUsage::COPY_DST,
            BufferUsage::INDEX,
            BufferUsage::VERTEX,
            BufferUsage::UNIFORM,
            BufferUsage::STORAGE,
            BufferUsage::INDIRECT,
            BufferUsage::QUERY_RESOLVE,
        ];
        let device = noop_device();
        for usage in usages {
            device.push_error_scope(ErrorFilter::Validation);
            let buffer = device.create_buffer(BufferDescriptor {
                usage,
                size: HUGE_SIZE,
                mapped_at_creation: true,
            });
            let _error = device.pop_error_scope().expect("scope should exist");

            // Buffer must be an error buffer; no mapping; no abort.
            assert!(
                buffer.is_error(),
                "expected error buffer for usage={usage:?}"
            );
            assert_eq!(
                buffer.map_state(),
                BufferMapState::Unmapped,
                "expected Unmapped for usage={usage:?}"
            );
            assert_eq!(
                buffer.mapped_range(false, 0, None),
                None,
                "expected null getMappedRange for usage={usage:?}"
            );
        }
    }

    #[test]
    fn small_mapped_at_creation_buffer_still_maps_and_round_trips() {
        // F-073 regression: ensure the normal mappedAtCreation path is unbroken.
        let device = noop_device();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: true,
        });
        assert!(!buffer.is_error());
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        assert!(
            buffer.mapped_range(false, 0, Some(16)).is_some(),
            "getMappedRange must succeed for a valid mappedAtCreation buffer"
        );
        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn zero_size_buffer_map_async_and_get_mapped_range_succeed() {
        // F-072: zero-size buffers and zero-length map ranges are valid WebGPU.
        // Verify on the Noop path (core validation must pass before any HAL).
        let device = noop_device();

        // Full-buffer read map on a size=0 MAP_READ buffer.
        let read_buf = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            size: 0,
            mapped_at_creation: false,
        });
        assert_eq!(read_buf.size(), 0);
        assert_eq!(read_buf.begin_map(MapMode::Read, 0, 0), Ok(1));
        assert_eq!(read_buf.resolve_pending_map(1), MapAsyncStatus::Success);
        assert_eq!(read_buf.map_state(), BufferMapState::Mapped);
        // WGPU_WHOLE_MAP_SIZE == None → uses buffer size (0) minus offset (0) = 0.
        assert!(
            read_buf.mapped_range(true, 0, None).is_some(),
            "get_const_mapped_range(0, WHOLE) must succeed on a zero-size buffer"
        );
        assert_eq!(read_buf.unmap(), None);
        assert_eq!(read_buf.map_state(), BufferMapState::Unmapped);

        // Full-buffer write map on a size=0 MAP_WRITE buffer (mappedAtCreation path).
        let mac_buf = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            size: 0,
            mapped_at_creation: true,
        });
        assert_eq!(mac_buf.size(), 0);
        assert_eq!(mac_buf.map_state(), BufferMapState::Mapped);
        assert!(
            mac_buf.mapped_range(false, 0, Some(0)).is_some(),
            "get_mapped_range(0, 0) must succeed on a zero-size mappedAtCreation buffer"
        );
        assert_eq!(mac_buf.unmap(), None);
        assert_eq!(mac_buf.map_state(), BufferMapState::Unmapped);
    }
}
