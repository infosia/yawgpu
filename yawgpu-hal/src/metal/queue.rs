use super::*;

/// Stores metal queue data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalQueue {
    /// Cached sampling capabilities and the lazily allocated mock blit buffer.
    pub(super) timestamp_resources: Arc<MetalTimestampResources>,
    pub(super) submissions: Arc<Mutex<MetalSubmissionTracker>>,
    pub(super) inner: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(super) submission_lock: Arc<Mutex<()>>,
}

struct MetalTrackedSubmission {
    index: SubmissionIndex,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _retained_copies: Vec<HalCopy>,
    _serialize_event: Option<Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
}

// SAFETY: Entries are published only after encoding has validated that every
// retained HalCopy resource is a thread-safe Metal wrapper, and only after the
// command buffer is committed. From then on yawgpu calls only Metal's
// synchronization-safe status, error, and wait methods on the command buffer.
unsafe impl Send for MetalTrackedSubmission {}

/// Tracks committed Metal command buffers and their Rust resource owners.
pub(super) struct MetalSubmissionTracker {
    last_issued: SubmissionIndex,
    completed: SubmissionIndex,
    command_buffers: VecDeque<MetalTrackedSubmission>,
    first_error: Option<(SubmissionIndex, String)>,
}

impl MetalSubmissionTracker {
    /// Creates an empty Metal submission timeline.
    pub(super) fn new() -> Self {
        Self {
            last_issued: SubmissionIndex::NONE,
            completed: SubmissionIndex::NONE,
            command_buffers: VecDeque::new(),
            first_error: None,
        }
    }

    fn reserve(&mut self) -> Result<SubmissionIndex, HalError> {
        let next = self
            .last_issued
            .0
            .checked_add(1)
            .map(SubmissionIndex)
            .ok_or_else(|| queue_submission_error("submission index exhausted"))?;
        self.last_issued = next;
        Ok(next)
    }

    fn register(
        &mut self,
        index: SubmissionIndex,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        retained_copies: Vec<HalCopy>,
        serialize_event: Option<Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
    ) {
        self.command_buffers.push_back(MetalTrackedSubmission {
            index,
            command_buffer,
            _retained_copies: retained_copies,
            _serialize_event: serialize_event,
        });
    }

    fn retire_terminal_prefix(&mut self) {
        while let Some(submission) = self.command_buffers.front() {
            let status = submission.command_buffer.status();
            if !command_buffer_status_is_terminal(status) {
                break;
            }
            let Some(submission) = self.command_buffers.pop_front() else {
                break;
            };
            self.completed = submission.index;
            if status == MTLCommandBufferStatus::Error && self.first_error.is_none() {
                let message = submission
                    .command_buffer
                    .error()
                    .map(|error| error.localizedDescription().to_string())
                    .unwrap_or_else(|| "Metal command buffer execution failed".to_owned());
                self.first_error = Some((submission.index, message));
            }
            // Dropping this entry releases both the retained command buffer
            // and every Rust HAL resource cloned from its HalCopy list.
            drop(submission);
        }
    }

    fn result_through(&self, index: SubmissionIndex) -> Result<(), HalError> {
        if let Some((failed_index, message)) = &self.first_error {
            if *failed_index <= index {
                return Err(queue_submission_error(format!(
                    "submission {} failed: {message}",
                    failed_index.0
                )));
            }
        }
        Ok(())
    }
}

impl Drop for MetalSubmissionTracker {
    fn drop(&mut self) {
        autoreleasepool(|_| {
            if let Some(submission) = self.command_buffers.back() {
                submission.command_buffer.waitUntilCompleted();
            }
            self.retire_terminal_prefix();
        });
    }
}

fn command_buffer_status_is_terminal(status: MTLCommandBufferStatus) -> bool {
    matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

impl std::fmt::Debug for MetalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQueue").finish_non_exhaustive()
    }
}

fn queue_submission_error(message: impl Into<String>) -> HalError {
    HalError::QueueSubmissionFailed {
        backend: BACKEND,
        message: message.into(),
    }
}

// Owns the retained encoder for one run; taking it ensures it is ended once.
#[derive(Default)]
struct CoalescedBlitEncoder {
    inner: Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>,
}

impl CoalescedBlitEncoder {
    fn get(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        creation_error: &'static str,
    ) -> Result<&ProtocolObject<dyn MTLBlitCommandEncoder>, HalError> {
        if self.inner.is_none() {
            self.inner = Some(
                command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| queue_submission_error(creation_error))?,
            );
        }
        self.inner
            .as_deref()
            .ok_or_else(|| queue_submission_error(creation_error))
    }

    fn end(&mut self) {
        if let Some(encoder) = self.inner.take() {
            encoder.endEncoding();
        }
    }
}

impl Drop for CoalescedBlitEncoder {
    fn drop(&mut self) {
        self.end();
    }
}

// Coalescable arms must encode only through the shared blit.get(...) encoder
// and must never touch command_buffer directly.
fn is_coalescable_blit(copy: &HalCopy) -> bool {
    match copy {
        HalCopy::Buffer(_)
        | HalCopy::BufferClear(_)
        | HalCopy::BufferToTexture(_)
        | HalCopy::TextureToBuffer(_)
        | HalCopy::TextureToTexture(_) => true,
        HalCopy::ClearTexture(_)
        | HalCopy::WriteTimestamp(_)
        | HalCopy::ResolveQuerySet(_)
        | HalCopy::ComputePass(_)
        | HalCopy::RenderPassCommandStream(_) => false,
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(_) => false,
    }
}

impl MetalQueue {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        Ok(MetalDevice::new()?.queue().clone())
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<SubmissionIndex, HalError> {
        autoreleasepool(|_| {
            let command_buffer = self.inner.commandBuffer().ok_or_else(|| {
                queue_submission_error("submit-empty command buffer creation returned nil")
            })?;
            self.commit_tracked(command_buffer, Vec::new(), None)
        })
    }

    /// Returns the highest submission index proven complete without blocking.
    pub fn completed_submission_index(&self) -> Result<SubmissionIndex, HalError> {
        autoreleasepool(|_| {
            let mut submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Metal command buffers committed to one queue finish in commit
            // order, so the first unfinished entry bounds the completed prefix.
            submissions.retire_terminal_prefix();
            let completed = submissions.completed;
            submissions.result_through(completed)?;
            Ok(completed)
        })
    }

    /// Blocks until the requested submission index has completed.
    pub fn wait_for_submission(&self, index: SubmissionIndex) -> Result<(), HalError> {
        let command_buffer = {
            let submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if index <= submissions.completed {
                return submissions.result_through(index);
            }
            if index > submissions.last_issued {
                return Err(queue_submission_error(
                    "submission index has not been issued",
                ));
            }
            submissions
                .command_buffers
                .iter()
                .find_map(|submission| {
                    (submission.index == index).then(|| submission.command_buffer.clone())
                })
                .ok_or_else(|| queue_submission_error("submission command buffer is unavailable"))?
        };
        autoreleasepool(|_| {
            command_buffer.waitUntilCompleted();
            let mut submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            submissions.retire_terminal_prefix();
            if submissions.completed < index {
                return Err(queue_submission_error(
                    "submission wait did not complete the queue timeline",
                ));
            }
            submissions.result_through(index)
        })
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        let _submission = self
            .submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let last_issued = self
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .last_issued;
        self.wait_for_submission(last_issued)
    }

    /// Signals and waits on this submission's event between encoders.
    /// Apple8+ needs Dawn's timestamp-generation/resolve workaround.
    fn serialize_timestamp_resolution(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        value: u64,
        event: &mut Option<Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
    ) -> Result<(), HalError> {
        if event.is_none() {
            *event = Some(self.inner.device().newSharedEvent().ok_or_else(|| {
                queue_submission_error("timestamp serialization shared event creation returned nil")
            })?);
        }
        let event = event.as_ref().ok_or_else(|| {
            queue_submission_error("timestamp serialization shared event is unavailable")
        })?;
        let event: &ProtocolObject<dyn objc2_metal::MTLEvent> = ProtocolObject::from_ref(&**event);
        command_buffer.encodeSignalEvent_value(event, value);
        command_buffer.encodeWaitForEvent_value(event, value);
        Ok(())
    }

    /// Records and submits the given buffer/texture copy operations.
    #[must_use = "submission can fail"]
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<SubmissionIndex, HalError> {
        if copies.is_empty() {
            return self.submit_empty();
        }

        autoreleasepool(|_| {
            let command_buffer = self.inner.commandBuffer().ok_or_else(|| {
                queue_submission_error("submit command buffer creation returned nil")
            })?;
            let mut serialize_event = None;
            let mut serialize_value = 0;
            let mut blit = CoalescedBlitEncoder::default();
            for copy in copies {
                if !is_coalescable_blit(copy) {
                    blit.end();
                }
                match copy {
                    HalCopy::Buffer(copy) => {
                        encode_buffer_copy(
                            blit.get(
                                &command_buffer,
                                "buffer-copy blit encoder creation returned nil",
                            )?,
                            copy,
                        )?;
                    }
                    HalCopy::BufferClear(clear) => {
                        encode_buffer_clear(
                            blit.get(
                                &command_buffer,
                                "buffer-clear blit encoder creation returned nil",
                            )?,
                            clear,
                        )?;
                    }
                    HalCopy::ClearTexture(clear) => {
                        encode_texture_clear(&command_buffer, clear)?;
                    }
                    HalCopy::WriteTimestamp(write) => {
                        let HalQuerySet::Metal(query_set) = &write.query_set else {
                            return Err(buffer_error("query set is not Metal-backed"));
                        };
                        let sample_buffer = query_set
                            .sample_buffer()
                            .ok_or_else(|| buffer_error("query set is not a timestamp set"))?;
                        if write.query_index >= query_set.count() {
                            return Err(buffer_error("timestamp query index exceeds query count"));
                        }
                        let index = to_ns(u64::from(write.query_index))?;
                        let resources = &self.timestamp_resources;
                        match timestamp_sampling_mode(
                            resources.counter_sampling_at_stage_boundary,
                            resources.counter_sampling_at_command_boundary,
                        ) {
                            Some(TimestampSamplingMode::StageBoundary) => {
                                // Submission encoding may race; OnceLock retains the winning allocation.
                                if resources.mock_blit.get().is_none() {
                                    let mock = self
                                        .inner
                                        .device()
                                        .newBufferWithLength_options(
                                            1,
                                            MTLResourceOptions::StorageModePrivate,
                                        )
                                        .ok_or(HalError::OutOfMemory {
                                            backend: BACKEND,
                                            resource: "timestamp mock blit buffer",
                                        })?;
                                    let _ = resources.mock_blit.set(MetalBuffer {
                                        inner: Some(mock),
                                        mapped_ptr: None,
                                        size: 1,
                                    });
                                }
                                let mock = resources
                                    .mock_blit
                                    .get()
                                    .ok_or_else(|| {
                                        buffer_error("timestamp mock blit buffer is unavailable")
                                    })?
                                    .inner()?;
                                let descriptor = objc2_metal::MTLBlitPassDescriptor::new();
                                // SAFETY: Blit descriptors provide attachment slot zero.
                                let attachment = unsafe {
                                    descriptor
                                        .sampleBufferAttachments()
                                        .objectAtIndexedSubscript(0)
                                };
                                attachment.setSampleBuffer(Some(sample_buffer));
                                // Sample at the START of this mock encoder, not the end.
                                // Dawn samples `endOfEncoderSampleIndex`, but on Apple8
                                // (M2, macOS 26) a blit pass whose end-of-encoder sample is
                                // requested right after another blit encoder (a buffer copy
                                // or clear) fails the whole command buffer with
                                // kIOGPUCommandBufferCallbackErrorOutOfMemory; measured
                                // 2026-09-22 across every ordering (Block 102 S2 review).
                                // Start-of-encoder sampling on the same descriptor works in
                                // every ordering, and since the encoder is a 1-byte mock
                                // fill the two points are the same instant for the caller.
                                // SAFETY: The start index was checked against the sample
                                // count; the end index explicitly disables sampling.
                                unsafe {
                                    attachment.setStartOfEncoderSampleIndex(index);
                                    attachment.setEndOfEncoderSampleIndex(
                                        objc2_metal::MTLCounterDontSample,
                                    );
                                }
                                let blit = command_buffer
                                    .blitCommandEncoderWithDescriptor(&descriptor)
                                    .ok_or_else(|| {
                                        queue_submission_error(
                                            "timestamp blit encoder creation returned nil",
                                        )
                                    })?;
                                blit.fillBuffer_range_value(mock, NSRange::new(0, 1), 0);
                                blit.endEncoding();
                            }
                            Some(TimestampSamplingMode::CommandBoundary) => {
                                let blit =
                                    command_buffer.blitCommandEncoder().ok_or_else(|| {
                                        queue_submission_error(
                                            "timestamp blit encoder creation returned nil",
                                        )
                                    })?;
                                // SAFETY: The sample index is in range and this mode is supported.
                                unsafe {
                                    blit.sampleCountersInBuffer_atSampleIndex_withBarrier(
                                        sample_buffer,
                                        index,
                                        true,
                                    );
                                }
                                blit.endEncoding();
                            }
                            None => {
                                return Err(buffer_error(
                                    "Metal device has no timestamp sampling mode",
                                ))
                            }
                        }
                    }
                    HalCopy::ResolveQuerySet(resolve) => {
                        let timestamp = matches!(&resolve.query_set,
                            HalQuerySet::Metal(set) if set.sample_buffer().is_some());
                        if self.timestamp_resources.serialize_timestamps {
                            if let Some(value) =
                                next_timestamp_resolution_value(&mut serialize_value, timestamp)?
                            {
                                self.serialize_timestamp_resolution(
                                    &command_buffer,
                                    value,
                                    &mut serialize_event,
                                )?;
                            }
                        }
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "query-resolve blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_resolve_query_set(&blit, resolve);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::BufferToTexture(copy) => {
                        encode_buffer_to_texture(
                            blit.get(
                                &command_buffer,
                                "buffer-to-texture blit encoder creation returned nil",
                            )?,
                            copy,
                        )?;
                    }
                    HalCopy::TextureToBuffer(copy) => {
                        encode_texture_to_buffer(
                            blit.get(
                                &command_buffer,
                                "texture-to-buffer blit encoder creation returned nil",
                            )?,
                            copy,
                        )?;
                    }
                    HalCopy::TextureToTexture(copy) => {
                        encode_texture_to_texture(
                            blit.get(
                                &command_buffer,
                                "texture-to-texture blit encoder creation returned nil",
                            )?,
                            copy,
                        )?;
                    }
                    HalCopy::ComputePass(pass) => {
                        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                            queue_submission_error("compute command encoder creation returned nil")
                        })?;
                        let result = encode_compute_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    HalCopy::RenderPassCommandStream(pass) => {
                        let descriptor = render_pass_command_stream_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| {
                                queue_submission_error(
                                    "render command encoder creation returned nil",
                                )
                            })?;
                        let result = encode_render_pass_command_stream(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    #[cfg(feature = "tiled")]
                    HalCopy::SubpassRenderPass(pass) => {
                        let descriptor = subpass_render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| {
                                queue_submission_error(
                                    "subpass render command encoder creation returned nil",
                                )
                            })?;
                        let result = encode_subpass_render_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                }
            }
            blit.end();
            self.commit_tracked(command_buffer, copies.to_vec(), serialize_event)
        })
    }

    fn commit_tracked(
        &self,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        retained_copies: Vec<HalCopy>,
        serialize_event: Option<Retained<ProtocolObject<dyn objc2_metal::MTLSharedEvent>>>,
    ) -> Result<SubmissionIndex, HalError> {
        let _submission = self
            .submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut submissions = self
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = submissions.reserve()?;
        command_buffer.commit();
        submissions.register(index, command_buffer, retained_copies, serialize_event);
        Ok(index)
    }
}

/// Advances only for timestamp resolves; each submission starts at zero.
fn next_timestamp_resolution_value(
    counter: &mut u64,
    timestamp: bool,
) -> Result<Option<u64>, HalError> {
    if !timestamp {
        return Ok(None);
    }
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| queue_submission_error("timestamp serialization value exhausted"))?;
    Ok(Some(*counter))
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use crate::HalBufferCopy;

    #[test]
    fn coalescable_blit_kinds_define_maximal_runs_without_gpu() {
        let device = crate::noop::NoopDevice::new();
        let buffer = HalBuffer::Noop(
            device
                .create_buffer(1024, HalBufferUsage::default())
                .expect("Noop buffer"),
        );
        let descriptor = texture_descriptor();
        let texture = HalTexture::Noop(device.create_texture(&descriptor).expect("Noop texture"));
        let origin = crate::HalOrigin3d { x: 0, y: 0, z: 0 };
        let extent = HalExtent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let transfer = HalBufferTextureCopy {
            buffer: buffer.clone(),
            buffer_layout: crate::HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
            texture: texture.clone(),
            format: descriptor.format,
            aspect: crate::HalTextureAspect::All,
            mip_level: 0,
            origin,
            extent,
        };
        let blits = [
            HalCopy::Buffer(HalBufferCopy {
                source: buffer.clone(),
                source_offset: 0,
                destination: buffer.clone(),
                destination_offset: 16,
                size: 16,
            }),
            HalCopy::BufferClear(HalBufferClear {
                buffer: buffer.clone(),
                offset: 0,
                size: 16,
            }),
            HalCopy::BufferToTexture(transfer.clone()),
            HalCopy::TextureToBuffer(transfer),
            HalCopy::TextureToTexture(HalTextureCopy {
                source: texture.clone(),
                source_mip_level: 0,
                source_origin: origin,
                destination: texture.clone(),
                destination_mip_level: 0,
                destination_origin: origin,
                extent,
            }),
        ];
        let query_set = HalQuerySet::Noop { count: 1 };
        let boundaries = [
            HalCopy::ClearTexture(HalTextureClear {
                texture,
                format: descriptor.format,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                base_array_layer: 0,
                array_layer_count: 1,
            }),
            HalCopy::WriteTimestamp(crate::HalWriteTimestamp {
                query_set: query_set.clone(),
                query_index: 0,
            }),
            HalCopy::ResolveQuerySet(HalResolveQuerySet {
                query_set,
                first_query: 0,
                query_count: 1,
                written_queries: vec![0],
                destination: buffer,
                destination_offset: 0,
            }),
            HalCopy::ComputePass(HalComputePass {
                pipeline: crate::HalComputePipeline::Noop,
                bind_buffers: Vec::new(),
                bind_textures: Vec::new(),
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            }),
            HalCopy::RenderPassCommandStream(HalRenderPassCommandStream::default()),
            #[cfg(feature = "tiled")]
            HalCopy::SubpassRenderPass(crate::HalSubpassRenderPassCommand {
                layout: crate::HalSubpassPassLayout {
                    color_attachments: Vec::new(),
                    depth_stencil_attachment: None,
                    subpasses: vec![crate::HalSubpassLayout {
                        color_attachment_indices: Vec::new(),
                        uses_depth_stencil: false,
                        input_attachments: Vec::new(),
                    }],
                    dependencies: Vec::new(),
                },
                extent,
                color_attachments: Vec::new(),
                depth_stencil_attachment: None,
                draws: Vec::new(),
            }),
        ];
        for boundary in &boundaries {
            assert!(!is_coalescable_blit(boundary), "{boundary:?}");
        }
        let mixed = vec![
            blits[0].clone(),
            blits[1].clone(),
            boundaries[0].clone(),
            blits[2].clone(),
            boundaries[1].clone(),
            boundaries[2].clone(),
            blits[3].clone(),
            blits[4].clone(),
        ];
        for (copies, expected) in [
            (vec![], vec![]),
            (blits.to_vec(), vec![true; 5]),
            (vec![boundaries[0].clone()], vec![false]),
            (
                mixed,
                vec![true, true, false, true, false, false, true, true],
            ),
        ] {
            let actual: Vec<_> = copies.iter().map(is_coalescable_blit).collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_coalesced_buffer_copies_last_write_wins() {
        const COPY_COUNT: u32 = 512;
        let device = metal_device();
        let staging = device
            .create_buffer(u64::from(COPY_COUNT) * 16, HalBufferUsage::default())
            .expect("staging buffer");
        let destination = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("destination buffer");
        let bytes: Vec<_> = (0..COPY_COUNT)
            .flat_map(|index| index.to_ne_bytes().repeat(4))
            .collect();
        staging.write(0, &bytes).expect("seed distinct regions");
        let copies: Vec<_> = (0..COPY_COUNT)
            .map(|index| {
                HalCopy::Buffer(HalBufferCopy {
                    source: HalBuffer::Metal(staging.clone()),
                    source_offset: u64::from(index) * 16,
                    destination: HalBuffer::Metal(destination.clone()),
                    destination_offset: 0,
                    size: 16,
                })
            })
            .collect();
        let submitted = device
            .queue()
            .submit_copies(&copies)
            .expect("submit copies");
        device
            .queue()
            .wait_for_submission(submitted)
            .expect("wait for copies");
        assert_eq!(
            destination.read(0, 16).expect("read destination"),
            (COPY_COUNT - 1).to_ne_bytes().repeat(4)
        );
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_coalesced_mixed_copies_across_compute_pass() {
        let device = metal_device();
        let staging = device
            .create_buffer(256, HalBufferUsage::default())
            .expect("staging buffer");
        let scratch = device
            .create_buffer(256, HalBufferUsage::default())
            .expect("scratch buffer");
        let readback = device
            .create_buffer(544, HalBufferUsage::default())
            .expect("readback buffer");
        staging
            .write(0, &[0x31, 0x72, 0xb4, 0xf5].repeat(64))
            .expect("seed upload");
        readback.write(0, &[0xa5; 544]).expect("seed readback");
        let mut descriptor = texture_descriptor();
        descriptor.width = 8;
        descriptor.height = 1;
        let texture = device.create_texture(&descriptor).expect("upload texture");
        let mirror = device.create_texture(&descriptor).expect("mirror texture");
        let origin = crate::HalOrigin3d { x: 0, y: 0, z: 0 };
        let extent = HalExtent3d {
            width: 8,
            height: 1,
            depth_or_array_layers: 1,
        };
        let transfer =
            |buffer: &MetalBuffer, texture: &MetalTexture, offset| HalBufferTextureCopy {
                buffer: HalBuffer::Metal(buffer.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset,
                    bytes_per_row: 256,
                    rows_per_image: 1,
                },
                texture: HalTexture::Metal(texture.clone()),
                format: descriptor.format,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin,
                extent,
            };
        let pipeline = device
            .create_compute_pipeline(compute_msl(), "main0", (1, 1, 1), &[])
            .expect("boundary compute pipeline");
        let copies = [
            HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(staging),
                source_offset: 0,
                destination: HalBuffer::Metal(scratch.clone()),
                destination_offset: 0,
                size: 256,
            }),
            HalCopy::BufferClear(HalBufferClear {
                buffer: HalBuffer::Metal(scratch.clone()),
                offset: 16,
                size: 16,
            }),
            HalCopy::BufferToTexture(transfer(&scratch, &texture, 0)),
            // Preserve the old bytes before overwriting their source in this run.
            HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(scratch.clone()),
                source_offset: 0,
                destination: HalBuffer::Metal(readback.clone()),
                destination_offset: 528,
                size: 16,
            }),
            HalCopy::BufferClear(HalBufferClear {
                buffer: HalBuffer::Metal(scratch.clone()),
                offset: 0,
                size: 16,
            }),
            HalCopy::TextureToTexture(HalTextureCopy {
                source: HalTexture::Metal(texture.clone()),
                source_mip_level: 0,
                source_origin: origin,
                destination: HalTexture::Metal(mirror.clone()),
                destination_mip_level: 0,
                destination_origin: origin,
                extent,
            }),
            HalCopy::TextureToBuffer(transfer(&readback, &mirror, 0)),
            HalCopy::ComputePass(HalComputePass {
                pipeline: crate::HalComputePipeline::Metal(pipeline),
                bind_buffers: Vec::new(),
                bind_textures: Vec::new(),
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            }),
            HalCopy::TextureToBuffer(transfer(&readback, &texture, 256)),
            HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(scratch.clone()),
                source_offset: 16,
                destination: HalBuffer::Metal(readback.clone()),
                destination_offset: 512,
                size: 16,
            }),
        ];
        let submitted = device
            .queue()
            .submit_copies(&copies)
            .expect("submit mixed copies");
        device
            .queue()
            .wait_for_submission(submitted)
            .expect("wait for mixed copies");
        let mut expected = [0x31, 0x72, 0xb4, 0xf5].repeat(4);
        expected.extend_from_slice(&[0; 16]);
        assert_eq!(readback.read(0, 32).expect("read mirror"), expected);
        assert_eq!(readback.read(256, 32).expect("read upload"), expected);
        assert_eq!(
            readback.read(512, 16).expect("read cleared buffer"),
            [0; 16]
        );
        assert_eq!(
            readback
                .read(528, 16)
                .expect("read before source overwrite"),
            [0x31, 0x72, 0xb4, 0xf5].repeat(4)
        );
        assert_eq!(
            scratch.read(0, 16).expect("read overwritten source"),
            [0; 16]
        );
    }

    #[test]
    fn timestamp_resolution_numbering_is_per_submission() {
        for kinds in [vec![], vec![false; 3], vec![true, false, true, true]] {
            for _ in 0..2 {
                let mut counter = 0;
                let values: Vec<_> = kinds
                    .iter()
                    .filter_map(|&timestamp| {
                        next_timestamp_resolution_value(&mut counter, timestamp).expect("numbering")
                    })
                    .collect();
                assert_eq!(
                    values,
                    (1..=kinds.iter().filter(|&&v| v).count() as u64).collect::<Vec<_>>()
                );
            }
        }
        let mut exhausted = u64::MAX;
        assert!(next_timestamp_resolution_value(&mut exhausted, true).is_err());
    }

    #[must_use]
    fn resolve_timestamp_ticks(written: &[u32]) -> [u64; 2] {
        let device = metal_device();
        let query_set = HalQuerySet::Metal(
            device
                .create_query_set(HalQueryKind::Timestamp, 2)
                .expect("create timestamps"),
        );
        // Metal buffers have no usage flags; private storage exercises the GPU-only resolve path.
        let destination = HalBuffer::Metal(MetalBuffer {
            inner: Some(
                device
                    .device
                    .newBufferWithLength_options(16, MTLResourceOptions::StorageModePrivate)
                    .expect("private destination"),
            ),
            mapped_ptr: None,
            size: 16,
        });
        let readback = device
            .create_buffer(
                16,
                HalBufferUsage {
                    copy_dst: true,
                    copy_src: true,
                    map_read: true,
                    ..Default::default()
                },
            )
            .expect("readback");
        let busy = HalBuffer::Metal(
            device
                .create_buffer(
                    1024 * 1024,
                    HalBufferUsage {
                        copy_dst: true,
                        ..Default::default()
                    },
                )
                .expect("busy buffer"),
        );
        // Seed every destination byte so the unwritten-slot test proves zero-fill.
        readback.write(0, &[0xff; 16]).expect("seed readback");
        let mut copies = vec![HalCopy::Buffer(HalBufferCopy {
            source: HalBuffer::Metal(readback.clone()),
            source_offset: 0,
            destination: destination.clone(),
            destination_offset: 0,
            size: 16,
        })];
        for &index in written {
            copies.push(HalCopy::WriteTimestamp(crate::HalWriteTimestamp {
                query_set: query_set.clone(),
                query_index: index,
            }));
            copies.push(HalCopy::BufferClear(HalBufferClear {
                buffer: busy.clone(),
                offset: 0,
                size: 1024 * 1024,
            }));
        }
        copies.push(HalCopy::ResolveQuerySet(HalResolveQuerySet {
            query_set: query_set.clone(),
            first_query: 0,
            query_count: 2,
            destination: destination.clone(),
            destination_offset: 0,
            written_queries: written.to_vec(),
        }));
        copies.push(HalCopy::Buffer(HalBufferCopy {
            source: destination,
            source_offset: 0,
            destination: HalBuffer::Metal(readback.clone()),
            destination_offset: 0,
            size: 16,
        }));
        device
            .queue()
            .submit_copies(&copies)
            .expect("submit timestamp operations");
        let retained_count = copies.len();
        drop(copies);
        drop(query_set);
        assert_eq!(
            device
                .queue()
                .submissions
                .lock()
                .expect("tracker")
                .command_buffers
                .back()
                .expect("submission")
                ._retained_copies
                .len(),
            retained_count
        );
        device.queue().wait_idle().expect("wait for timestamps");
        let bytes = readback.read(0, 16).expect("read ticks");
        [
            u64::from_ne_bytes(bytes[..8].try_into().expect("first tick")),
            u64::from_ne_bytes(bytes[8..].try_into().expect("second tick")),
        ]
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_write_timestamp_then_resolve_yields_increasing_ticks() {
        let [t0, t1] = resolve_timestamp_ticks(&[0, 1]);
        assert!(t0 > 0);
        assert!(t1 > t0);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_resolve_timestamp_query_set_zero_fills_unwritten_slots() {
        let [t0, t1] = resolve_timestamp_ticks(&[1]);
        assert_eq!(t0, 0);
        assert!(t1 > 0);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_timestamp_operations_reject_invalid_sets_and_ranges() {
        let device = metal_device();
        for (kind, index) in [(HalQueryKind::Occlusion, 0), (HalQueryKind::Timestamp, 2)] {
            let query_set =
                HalQuerySet::Metal(device.create_query_set(kind, 2).expect("query set"));
            assert!(device
                .queue()
                .submit_copies(&[HalCopy::WriteTimestamp(crate::HalWriteTimestamp {
                    query_set,
                    query_index: index
                })])
                .is_err());
        }
        let query_set = HalQuerySet::Metal(
            device
                .create_query_set(HalQueryKind::Timestamp, 2)
                .expect("timestamps"),
        );
        let destination = HalBuffer::Metal(
            device
                .create_buffer(16, HalBufferUsage::default())
                .expect("destination"),
        );
        for (first_query, query_count, destination_offset, written_queries) in [
            (1, 2, 0, vec![]),
            (u32::MAX, 2, 0, vec![]),
            (0, 2, 8, vec![]),
            (1, 1, 0, vec![0]),
            (0, 1, 0, vec![1]),
        ] {
            assert!(device
                .queue()
                .submit_copies(&[HalCopy::ResolveQuerySet(HalResolveQuerySet {
                    query_set: query_set.clone(),
                    destination: destination.clone(),
                    first_query,
                    query_count,
                    destination_offset,
                    written_queries,
                })])
                .is_err());
        }
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_new_constructs_queue() {
        MetalQueue::new().expect("create Metal queue");
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_empty_is_tracked_until_waited() {
        let device = metal_device();
        let queue = device.queue();
        let submitted = queue.submit_empty().expect("submit empty queue work");
        assert_eq!(
            queue
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .command_buffers
                .len(),
            1
        );
        queue
            .wait_for_submission(submitted)
            .expect("wait for empty queue work");
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_completion_poll_advances_and_evicts_finished_submissions() {
        let device = metal_device();
        let queue = device.queue();

        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query initial completion"),
            SubmissionIndex::NONE
        );
        let first = queue.submit_empty().expect("submit first empty work");
        let second = queue.submit_empty().expect("submit second empty work");

        assert!(first < second);
        let second_command_buffer = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .back()
            .map(|submission| submission.command_buffer.clone())
            .expect("second command buffer is tracked");
        autoreleasepool(|_| second_command_buffer.waitUntilCompleted());
        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query completed submission"),
            second
        );
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_wait_for_submission_advances_and_evicts_in_commit_order() {
        let device = metal_device();
        let queue = device.queue();
        let first = queue.submit_empty().expect("submit first empty work");
        let second = queue.submit_empty().expect("submit second empty work");

        queue
            .wait_for_submission(second)
            .expect("wait for second submission");
        let submissions = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(first < second);
        assert_eq!(submissions.completed, second);
        assert!(submissions.command_buffers.is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_wait_idle_waits_for_and_evicts_all_submissions() {
        let device = metal_device();
        let queue = device.queue();
        let submitted = queue.submit_empty().expect("submit empty work");

        queue.wait_idle().expect("wait for queue idle");
        let submissions = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(submissions.completed, submitted);
        assert!(submissions.command_buffers.is_empty());
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_command_buffer_error_status_is_terminal() {
        assert!(command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Completed
        ));
        assert!(command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Error
        ));
        assert!(!command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Scheduled
        ));
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_submission_tracker_starts_empty() {
        let tracker = MetalSubmissionTracker::new();
        assert_eq!(tracker.last_issued, SubmissionIndex::NONE);
        assert_eq!(tracker.completed, SubmissionIndex::NONE);
        assert!(tracker.command_buffers.is_empty());
        assert!(tracker.first_error.is_none());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_copies_accepts_buffer_copy() {
        let device = metal_device();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal source buffer allocation should succeed");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal destination buffer allocation should succeed");
        source.write(0, &[1, 2, 3, 4]).expect("write source");
        let submitted = device
            .queue()
            .submit_copies(&[HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(source),
                source_offset: 0,
                destination: HalBuffer::Metal(destination.clone()),
                destination_offset: 0,
                size: 4,
            })])
            .expect("submit buffer copy");
        device
            .queue()
            .wait_for_submission(submitted)
            .expect("wait for buffer copy");
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_copies_retains_hal_copies_until_completion() {
        let device = metal_device();
        let queue = device.queue();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("create source buffer");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("create destination buffer");
        source.write(0, &[1, 2, 3, 4]).expect("write source");
        let copies = vec![HalCopy::Buffer(HalBufferCopy {
            source: HalBuffer::Metal(source),
            source_offset: 0,
            destination: HalBuffer::Metal(destination.clone()),
            destination_offset: 0,
            size: 4,
        })];

        let submitted = queue.submit_copies(&copies).expect("submit buffer copy");
        drop(copies);
        assert_eq!(
            queue
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .command_buffers
                .front()
                .map(|submission| submission._retained_copies.len()),
            Some(1)
        );
        queue
            .wait_for_submission(submitted)
            .expect("wait for retained copy");
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
