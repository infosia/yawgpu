use std::sync::Arc;

use yawgpu_hal::HalError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hal(#[from] HalError),
    #[error("{0}")]
    Validation(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    Validation,
    OutOfMemory,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorFilter {
    Validation,
    OutOfMemory,
    Internal,
}

impl ErrorFilter {
    #[must_use]
    pub(crate) fn matches(self, kind: ErrorKind) -> bool {
        matches!(
            (self, kind),
            (Self::Validation, ErrorKind::Validation)
                | (Self::OutOfMemory, ErrorKind::OutOfMemory)
                | (Self::Internal, ErrorKind::Internal)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PopErrorScopeError {
    EmptyStack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceError {
    pub kind: ErrorKind,
    pub message: String,
}

impl DeviceError {
    #[must_use]
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Validation,
            message: message.into(),
        }
    }

    #[must_use]
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
}

impl DeviceError {
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

pub(crate) type UncapturedErrorCallback = Arc<dyn Fn(DeviceError) + Send + Sync>;

#[derive(Default)]
pub(crate) struct ErrorSink {
    pub(crate) uncaptured_error_callback: Option<UncapturedErrorCallback>,
    pub(crate) scopes: Vec<ErrorScope>,
}

pub(crate) struct ErrorScope {
    pub(crate) filter: ErrorFilter,
    pub(crate) error: Option<DeviceError>,
}

impl std::fmt::Debug for ErrorSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorSink")
            .field(
                "uncaptured_error_callback",
                &self.uncaptured_error_callback.is_some(),
            )
            .field("scopes", &self.scopes)
            .finish()
    }
}

impl std::fmt::Debug for ErrorScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorScope")
            .field("filter", &self.filter)
            .field("error", &self.error)
            .finish()
    }
}
