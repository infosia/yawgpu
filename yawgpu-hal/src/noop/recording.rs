use super::NoopBufferStorage;
use crate::{HalBuffer, HalComputeDispatch, HalCopy, HalRenderPassCommand};

fn release_buffer_storage(buffer: &mut HalBuffer) {
    #[allow(irrefutable_let_patterns)]
    if let HalBuffer::Noop(buffer) = buffer {
        buffer.data = NoopBufferStorage::Recorded(buffer.downgrade_storage());
    }
}

/// Releases allocations held by a completed copy's inspection record.
pub(super) fn release_copy_storage(copy: &mut HalCopy) {
    match copy {
        HalCopy::Buffer(copy) => {
            release_buffer_storage(&mut copy.source);
            release_buffer_storage(&mut copy.destination);
        }
        HalCopy::BufferClear(clear) => release_buffer_storage(&mut clear.buffer),
        HalCopy::BufferToTexture(copy) | HalCopy::TextureToBuffer(copy) => {
            release_buffer_storage(&mut copy.buffer);
        }
        HalCopy::ResolveQuerySet(resolve) => release_buffer_storage(&mut resolve.destination),
        HalCopy::ComputePass(pass) => {
            for bound in &mut pass.bind_buffers {
                release_buffer_storage(&mut bound.buffer);
            }
            for bound in &mut pass.bind_external_textures {
                release_buffer_storage(&mut bound.params);
            }
            if let HalComputeDispatch::Indirect { buffer } = &mut pass.dispatch {
                release_buffer_storage(&mut buffer.buffer);
            }
        }
        HalCopy::RenderPassCommandStream(pass) => release_render_storage(&mut pass.commands),
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => {
            for draw in &mut pass.draws {
                for bound in draw.bind_buffers.iter_mut().chain(&mut draw.vertex_buffers) {
                    release_buffer_storage(&mut bound.buffer);
                }
                if let Some(bound) = &mut draw.index_buffer {
                    release_buffer_storage(&mut bound.buffer);
                }
            }
        }
        // Noop textures and query sets contain only value metadata.
        HalCopy::ClearTexture(_) | HalCopy::TextureToTexture(_) | HalCopy::WriteTimestamp(_) => {}
    }
}

/// Releases allocations held by completed render commands, including bundles.
pub(super) fn release_render_storage(commands: &mut [HalRenderPassCommand]) {
    for command in commands {
        match command {
            HalRenderPassCommand::SetBindGroup {
                buffers,
                external_textures,
                ..
            } => {
                for bound in buffers {
                    release_buffer_storage(&mut bound.buffer);
                }
                for bound in external_textures {
                    release_buffer_storage(&mut bound.params);
                }
            }
            HalRenderPassCommand::SetVertexBuffer {
                buffer: Some(bound),
                ..
            } => release_buffer_storage(&mut bound.buffer),
            HalRenderPassCommand::SetIndexBuffer(bound) => {
                release_buffer_storage(&mut bound.buffer);
            }
            HalRenderPassCommand::DrawIndirect { indirect_buffer }
            | HalRenderPassCommand::DrawIndexedIndirect { indirect_buffer } => {
                release_buffer_storage(&mut indirect_buffer.buffer);
            }
            HalRenderPassCommand::ExecuteRenderBundle(bundle) => {
                release_render_storage(&mut bundle.commands);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HalBoundIndirectBuffer, HalBufferClear, HalRenderBundle};

    #[test]
    fn completed_copy_record_preserves_metadata_without_owning_storage() {
        let buffer = super::super::NoopBuffer::new(16);
        let weak = buffer.downgrade_storage();
        buffer.write(0, &[42]).unwrap();
        let mut copy = HalCopy::BufferClear(HalBufferClear {
            buffer: HalBuffer::Noop(buffer.clone()),
            offset: 4,
            size: 8,
        });
        release_copy_storage(&mut copy);
        release_copy_storage(&mut copy);
        let HalCopy::BufferClear(clear) = copy else {
            panic!("expected buffer clear");
        };
        assert_eq!(clear.buffer.read(0, 1).unwrap(), [42]);
        drop(buffer);
        assert!(weak.upgrade().is_none());
        assert_eq!(clear.buffer.size(), 16);
        assert!(clear.buffer.read(0, 1).is_err());
        assert!(clear.buffer.write(0, &[0]).is_err());
    }

    #[test]
    fn completed_render_bundle_releases_indirect_buffer_storage() {
        let buffer = super::super::NoopBuffer::new(16);
        let weak = buffer.downgrade_storage();
        let mut commands = vec![HalRenderPassCommand::ExecuteRenderBundle(HalRenderBundle {
            commands: vec![HalRenderPassCommand::DrawIndirect {
                indirect_buffer: HalBoundIndirectBuffer {
                    buffer: HalBuffer::Noop(buffer),
                    offset: 0,
                },
            }],
        })];
        assert!(weak.upgrade().is_some());
        release_render_storage(&mut commands);
        assert!(weak.upgrade().is_none());
        assert_eq!(commands.len(), 1);
    }
}
