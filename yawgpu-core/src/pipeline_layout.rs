use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::limits::*;

/// Describes pipeline layout descriptor.
#[derive(Debug, Clone)]
pub struct PipelineLayoutDescriptor {
    /// Bind group layouts.
    pub bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    /// Immediate size.
    pub immediate_size: u32,
    /// Error.
    pub error: Option<String>,
}

/// Stores layout metadata.
#[derive(Debug, Clone)]
pub struct PipelineLayout {
    pub(crate) inner: Arc<PipelineLayoutInner>,
}

/// Stores layout metadata.
#[derive(Debug)]
pub(crate) struct PipelineLayoutInner {
    pub(crate) _bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    pub(crate) _immediate_size: u32,
    pub(crate) is_error: bool,
}

impl PipelineLayout {
    /// Creates a new instance.
    pub(crate) fn new(
        bind_group_layouts: Vec<Arc<BindGroupLayout>>,
        immediate_size: u32,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(PipelineLayoutInner {
                _bind_group_layouts: bind_group_layouts,
                _immediate_size: immediate_size,
                is_error,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the bind group layouts.
    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner._bind_group_layouts
    }
}

/// Validates pipeline layout descriptor and returns a descriptive error on failure.
pub(crate) fn validate_pipeline_layout_descriptor(
    bind_group_layouts: &[Arc<BindGroupLayout>],
    immediate_size: u32,
    limits: Limits,
) -> Option<String> {
    if bind_group_layouts.len() > limits.max_bind_groups as usize {
        return Some("pipeline layout bindGroupLayoutCount exceeds the device limit".to_owned());
    }
    if bind_group_layouts.iter().any(|layout| layout.is_error()) {
        return Some("pipeline layout cannot contain an error bind group layout".to_owned());
    }
    if bind_group_layouts.iter().any(|layout| layout.is_default()) {
        return Some("pipeline layout cannot contain a default bind group layout".to_owned());
    }
    if immediate_size > limits.max_immediate_size {
        return Some("pipeline layout immediateSize exceeds the device limit".to_owned());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

    #[test]
    fn pipeline_layout_accessors_pin_bind_group_layouts_and_is_error() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: None,
        }));

        let pipeline_layout = device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![layout.clone()],
            immediate_size: 0,
            error: None,
        });
        assert!(!pipeline_layout.is_error());
        assert_eq!(pipeline_layout.bind_group_layouts().len(), 1);
        assert!(pipeline_layout.bind_group_layouts()[0].same(&layout));

        device.push_error_scope(ErrorFilter::Validation);
        let error_layout = device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::new(BindGroupLayout::error())],
            immediate_size: 0,
            error: None,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid pipeline layout should be scoped");
        assert!(error_layout.is_error());
        assert_eq!(
            error.message,
            "pipeline layout cannot contain an error bind group layout"
        );
    }
}
