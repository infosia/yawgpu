use std::sync::Arc;

use yawgpu_hal::HalError;

/// Enumerates error values.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    /// Hal variant.
    Hal(#[from] HalError),
    #[error("{0}")]
    /// Validation variant.
    Validation(String),
}

/// Enumerates error kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Validation variant.
    Validation,
    /// Out of memory variant.
    OutOfMemory,
    /// Internal variant.
    Internal,
}

/// Enumerates error filter values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorFilter {
    /// Validation variant.
    Validation,
    /// Out of memory variant.
    OutOfMemory,
    /// Internal variant.
    Internal,
}

impl ErrorFilter {
    /// Returns true when the value matches the requested condition.
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

/// Enumerates pop error scope error values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PopErrorScopeError {
    /// Empty stack variant.
    EmptyStack,
}

/// Stores device error data used by validation and backend submission.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceError {
    /// Kind.
    pub kind: ErrorKind,
    /// Message.
    pub message: String,
}

impl DeviceError {
    /// Builds a validation `DeviceError` carrying `message`.
    #[must_use]
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Validation,
            message: message.into(),
        }
    }

    /// Builds an internal `DeviceError` carrying `message`.
    #[must_use]
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
}

impl DeviceError {
    /// Creates a new instance.
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Alias for uncaptured error callback.
pub(crate) type UncapturedErrorCallback = Arc<dyn Fn(DeviceError) + Send + Sync>;

/// Stores error sink data used by validation and backend submission.
#[derive(Default)]
pub(crate) struct ErrorSink {
    pub(crate) uncaptured_error_callback: Option<UncapturedErrorCallback>,
    pub(crate) scopes: Vec<ErrorScope>,
}

/// Stores error scope data used by validation and backend submission.
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
