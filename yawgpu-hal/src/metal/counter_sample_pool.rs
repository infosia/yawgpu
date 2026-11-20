use std::ops::Range;
use std::sync::{PoisonError, Weak};

use objc2_metal::{MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor};

use super::*;

const COUNTER_SAMPLE_BUFFER_CAPACITY: usize = 4096;

type SampleBuffer = ProtocolObject<dyn MTLCounterSampleBuffer>;

struct CounterSampleBuffer(Retained<SampleBuffer>);

// SAFETY: Metal counter sample buffers support thread-safe retain/release and
// command encoding. The private sample memory is accessed only by GPU commands.
unsafe impl Send for CounterSampleBuffer {}
unsafe impl Sync for CounterSampleBuffer {}

/// Owns the counter sample buffers for one Metal device.
pub(super) struct CounterSamplePool {
    slots: Arc<Mutex<SlotAllocator<CounterSampleBuffer>>>,
}

impl Default for CounterSamplePool {
    fn default() -> Self {
        Self {
            slots: Arc::new(Mutex::new(SlotAllocator {
                buffers: Vec::new(),
            })),
        }
    }
}

impl CounterSamplePool {
    /// Reserves a contiguous run of samples, creating a private buffer if needed.
    #[must_use = "slot allocation can fail"]
    pub(super) fn allocate(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        count: u32,
    ) -> Result<CounterSampleAllocation, HalError> {
        let reservation = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .allocate(to_ns(u64::from(count))?, |capacity| {
                create_sample_buffer(device, capacity)
            })?;
        Ok(CounterSampleAllocation {
            pool: Arc::downgrade(&self.slots),
            buffer: Some(reservation.buffer),
            slots: reservation.slots,
        })
    }
}

/// Keeps a sample buffer and its slot reservation alive for a timestamp query set.
pub(super) struct CounterSampleAllocation {
    pool: Weak<Mutex<SlotAllocator<CounterSampleBuffer>>>,
    // Taken by Drop so the buffer reference is released under the pool lock.
    buffer: Option<Arc<CounterSampleBuffer>>,
    slots: Range<usize>,
}

impl CounterSampleAllocation {
    /// Returns the backing Metal buffer for this live reservation.
    #[must_use]
    pub(super) fn sample_buffer(&self) -> Option<&SampleBuffer> {
        self.buffer.as_deref().map(|buffer| &*buffer.0)
    }

    /// Returns the first sample index in this live reservation.
    #[must_use]
    pub(super) fn base_index(&self) -> usize {
        self.slots.start
    }
}

impl Drop for CounterSampleAllocation {
    fn drop(&mut self) {
        if let (Some(pool), Some(buffer)) = (self.pool.upgrade(), self.buffer.take()) {
            pool.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .release(SlotReservation {
                    buffer,
                    slots: self.slots.clone(),
                });
        }
    }
}

fn create_sample_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    capacity: usize,
) -> Result<CounterSampleBuffer, HalError> {
    let counter_set = metal_timestamp_counter_set(device)
        .ok_or_else(|| buffer_error("Metal device has no timestamp counter set"))?;
    let descriptor = MTLCounterSampleBufferDescriptor::new();
    descriptor.setCounterSet(Some(&counter_set));
    descriptor.setStorageMode(MTLStorageMode::Private);
    // SAFETY: The allocator requests at least one sample with a representable count.
    unsafe { descriptor.setSampleCount(capacity) };
    device
        .newCounterSampleBufferWithDescriptor_error(&descriptor)
        .map(CounterSampleBuffer)
        .map_err(|_| sample_buffer_out_of_memory())
}

fn sample_buffer_out_of_memory() -> HalError {
    HalError::OutOfMemory {
        backend: BACKEND,
        resource: "timestamp counter sample buffer",
    }
}

struct SlotReservation<T> {
    buffer: Arc<T>,
    slots: Range<usize>,
}

struct SlotBuffer<T> {
    buffer: Arc<T>,
    occupied: Vec<bool>,
    occupied_count: usize,
    // Skip the occupied prefix when searching for a run. Capacity marks a full buffer.
    first_free: usize,
}

impl<T> SlotBuffer<T> {
    fn reserve(&mut self, count: usize) -> Option<Range<usize>> {
        if count > self.occupied.len() - self.occupied_count {
            return None;
        }

        let mut run = 0;
        for (index, &occupied) in self.occupied.iter().enumerate().skip(self.first_free) {
            run = if occupied { 0 } else { run + 1 };
            if run == count {
                let end = index + 1;
                let slots = end - count..end;
                self.occupied.get_mut(slots.clone())?.fill(true);
                self.occupied_count += count;
                if slots.start == self.first_free {
                    self.first_free = self
                        .occupied
                        .iter()
                        .enumerate()
                        .skip(end)
                        .find_map(|(index, &occupied)| (!occupied).then_some(index))
                        .unwrap_or(self.occupied.len());
                }
                return Some(slots);
            }
        }
        None
    }
}

// Tracks slots independently of the backing resource type so it can be tested without Metal.
struct SlotAllocator<T> {
    buffers: Vec<SlotBuffer<T>>,
}

impl<T> SlotAllocator<T> {
    fn allocate(
        &mut self,
        count: usize,
        create_buffer: impl FnOnce(usize) -> Result<T, HalError>,
    ) -> Result<SlotReservation<T>, HalError> {
        let count = count.max(1);
        for entry in &mut self.buffers {
            if let Some(slots) = entry.reserve(count) {
                return Ok(SlotReservation {
                    buffer: Arc::clone(&entry.buffer),
                    slots,
                });
            }
        }

        let capacity = count.max(COUNTER_SAMPLE_BUFFER_CAPACITY);
        let buffer = Arc::new(create_buffer(capacity)?);
        let mut occupied = Vec::new();
        occupied
            .try_reserve_exact(capacity)
            .map_err(|_| sample_buffer_out_of_memory())?;
        occupied.resize(count, true);
        occupied.resize(capacity, false);
        self.buffers
            .try_reserve(1)
            .map_err(|_| sample_buffer_out_of_memory())?;
        self.buffers.push(SlotBuffer {
            buffer: Arc::clone(&buffer),
            occupied,
            occupied_count: count,
            first_free: count,
        });
        Ok(SlotReservation {
            buffer,
            slots: 0..count,
        })
    }

    fn release(&mut self, reservation: SlotReservation<T>) {
        let Some(index) = self
            .buffers
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.buffer, &reservation.buffer))
        else {
            return;
        };
        let SlotReservation { buffer, slots } = reservation;
        // Drop both backing buffer references under the same lock when the buffer
        // becomes empty, so concurrent allocations can use the Metal buffer limit.
        drop(buffer);
        let entry = &mut self.buffers[index];
        let Some(occupied) = entry.occupied.get_mut(slots.clone()) else {
            return;
        };
        occupied.fill(false);
        entry.first_free = entry.first_free.min(slots.start);
        entry.occupied_count -= slots.len();
        if entry.occupied_count == 0 {
            self.buffers.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocate(allocator: &mut SlotAllocator<usize>, count: usize) -> SlotReservation<usize> {
        allocator.allocate(count, Ok).expect("reserve sample slots")
    }

    #[test]
    fn first_fit_uses_buffer_creation_order() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let first = allocate(&mut allocator, 3);
        let tail = allocate(&mut allocator, 4093);
        let second = allocate(&mut allocator, 2);
        allocator.release(tail);

        let fit = allocate(&mut allocator, 2);
        assert!(Arc::ptr_eq(&fit.buffer, &first.buffer));
        assert!(!Arc::ptr_eq(&fit.buffer, &second.buffer));
        assert_eq!(fit.slots, 3..5);
    }

    #[test]
    fn reuses_a_freed_run() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let first = allocate(&mut allocator, 4);
        let middle = allocate(&mut allocator, 3);
        let last = allocate(&mut allocator, 2);
        assert_eq!(middle.slots, 4..7);
        assert_eq!(last.slots, 7..9);
        allocator.release(middle);

        let reused = allocate(&mut allocator, 2);
        assert!(Arc::ptr_eq(&reused.buffer, &first.buffer));
        assert_eq!(reused.slots, 4..6);
        assert_eq!(allocate(&mut allocator, 1).slots, 6..7);
        assert_eq!(allocate(&mut allocator, 1).slots, 9..10);
    }

    #[test]
    fn no_fit_creates_a_new_buffer() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let first = allocate(&mut allocator, 4095);
        let second = allocate(&mut allocator, 2);
        assert!(!Arc::ptr_eq(&first.buffer, &second.buffer));
        assert_eq!(second.slots, 0..2);
        assert_eq!(allocator.buffers.len(), 2);
    }

    #[test]
    fn last_release_removes_the_empty_buffer() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let first = allocate(&mut allocator, 1);
        let second = allocate(&mut allocator, 1);
        let buffer = Arc::downgrade(&first.buffer);
        allocator.release(first);
        assert_eq!(allocator.buffers.len(), 1);
        assert!(buffer.upgrade().is_some());

        allocator.release(second);
        assert!(allocator.buffers.is_empty());
        assert!(buffer.upgrade().is_none());
    }

    #[test]
    fn zero_count_reserves_one_slot() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let zero = allocate(&mut allocator, 0);
        let next = allocate(&mut allocator, 1);
        assert!(Arc::ptr_eq(&zero.buffer, &next.buffer));
        assert_eq!(zero.slots, 0..1);
        assert_eq!(next.slots, 1..2);
    }

    #[test]
    fn request_of_4096_fills_one_buffer() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let full = allocate(&mut allocator, 4096);
        assert_eq!(*full.buffer, 4096);
        assert_eq!(full.slots, 0..4096);
        assert_eq!(allocator.buffers.len(), 1);
        assert_eq!(allocator.buffers[0].occupied_count, 4096);
    }

    #[test]
    fn fragmented_free_slots_cannot_satisfy_a_larger_request() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let first = allocate(&mut allocator, 1024);
        let middle = allocate(&mut allocator, 1024);
        let _third = allocate(&mut allocator, 1024);
        allocator.release(middle);

        let larger = allocate(&mut allocator, 1025);
        assert!(!Arc::ptr_eq(&first.buffer, &larger.buffer));
        assert_eq!(larger.slots, 0..1025);
        let fit = allocate(&mut allocator, 1024);
        assert!(Arc::ptr_eq(&fit.buffer, &first.buffer));
        assert_eq!(fit.slots, 1024..2048);
    }

    #[test]
    fn failed_creation_leaves_the_pool_usable() {
        let mut allocator = SlotAllocator {
            buffers: Vec::new(),
        };
        let full = allocate(&mut allocator, 4096);
        let failure = allocator.allocate(4097, |capacity| {
            assert_eq!(capacity, 4097);
            Err(sample_buffer_out_of_memory())
        });
        assert!(matches!(
            failure,
            Err(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "timestamp counter sample buffer",
            })
        ));
        assert_eq!(allocator.buffers.len(), 1);
        assert!(Arc::ptr_eq(&full.buffer, &allocator.buffers[0].buffer));
        let oversized = allocate(&mut allocator, 4097);
        assert_eq!(*oversized.buffer, 4097);
        assert_eq!(oversized.slots, 0..4097);
    }
}
