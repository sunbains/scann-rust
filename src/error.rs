//! Error types for ScaNN.
//!
//! This module provides error handling similar to the C++ Status/StatusOr pattern.

use std::fmt;
use thiserror::Error;

/// Error codes matching the C++ implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Operation completed successfully.
    Ok,
    /// Operation was cancelled.
    Cancelled,
    /// Unknown error.
    Unknown,
    /// Invalid argument provided.
    InvalidArgument,
    /// Deadline exceeded.
    DeadlineExceeded,
    /// Resource not found.
    NotFound,
    /// Resource already exists.
    AlreadyExists,
    /// Permission denied.
    PermissionDenied,
    /// Resource exhausted.
    ResourceExhausted,
    /// Failed precondition.
    FailedPrecondition,
    /// Operation aborted.
    Aborted,
    /// Operation out of range.
    OutOfRange,
    /// Operation not implemented.
    Unimplemented,
    /// Internal error.
    Internal,
    /// Service unavailable.
    Unavailable,
    /// Data loss occurred.
    DataLoss,
    /// Unauthenticated request.
    Unauthenticated,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::Ok => write!(f, "OK"),
            ErrorCode::Cancelled => write!(f, "CANCELLED"),
            ErrorCode::Unknown => write!(f, "UNKNOWN"),
            ErrorCode::InvalidArgument => write!(f, "INVALID_ARGUMENT"),
            ErrorCode::DeadlineExceeded => write!(f, "DEADLINE_EXCEEDED"),
            ErrorCode::NotFound => write!(f, "NOT_FOUND"),
            ErrorCode::AlreadyExists => write!(f, "ALREADY_EXISTS"),
            ErrorCode::PermissionDenied => write!(f, "PERMISSION_DENIED"),
            ErrorCode::ResourceExhausted => write!(f, "RESOURCE_EXHAUSTED"),
            ErrorCode::FailedPrecondition => write!(f, "FAILED_PRECONDITION"),
            ErrorCode::Aborted => write!(f, "ABORTED"),
            ErrorCode::OutOfRange => write!(f, "OUT_OF_RANGE"),
            ErrorCode::Unimplemented => write!(f, "UNIMPLEMENTED"),
            ErrorCode::Internal => write!(f, "INTERNAL"),
            ErrorCode::Unavailable => write!(f, "UNAVAILABLE"),
            ErrorCode::DataLoss => write!(f, "DATA_LOSS"),
            ErrorCode::Unauthenticated => write!(f, "UNAUTHENTICATED"),
        }
    }
}

/// Main error type for ScaNN operations.
#[derive(Error, Debug, Clone)]
pub struct ScannError {
    code: ErrorCode,
    message: String,
}

impl ScannError {
    /// Create a new error with the given code and message.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Get the error code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Get the error message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Check if this is an OK status (should not be used for errors).
    pub fn ok(&self) -> bool {
        self.code == ErrorCode::Ok
    }

    // Convenience constructors

    /// Create an invalid argument error.
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, msg)
    }

    /// Create a not found error.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, msg)
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, msg)
    }

    /// Create a failed precondition error.
    pub fn failed_precondition(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::FailedPrecondition, msg)
    }

    /// Create an unimplemented error.
    pub fn unimplemented(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::Unimplemented, msg)
    }

    /// Create an out of range error.
    pub fn out_of_range(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::OutOfRange, msg)
    }

    /// Create a resource exhausted error.
    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceExhausted, msg)
    }
}

impl fmt::Display for ScannError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// Result type alias for ScaNN operations.
pub type Result<T> = std::result::Result<T, ScannError>;

/// Status type similar to C++ Status (unit result).
pub type Status = Result<()>;

/// Extension trait for Result to provide status-like methods.
pub trait StatusExt<T> {
    /// Check if the result is OK.
    fn is_ok(&self) -> bool;

    /// Get the status (error) if present.
    fn status(&self) -> Option<&ScannError>;
}

impl<T> StatusExt<T> for Result<T> {
    fn is_ok(&self) -> bool {
        self.as_ref().is_ok()
    }

    fn status(&self) -> Option<&ScannError> {
        self.as_ref().err()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ScannError::invalid_argument("bad value");
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
        assert_eq!(err.message(), "bad value");
        assert!(!err.ok());
    }

    #[test]
    fn test_error_display() {
        let err = ScannError::not_found("item not found");
        let display = format!("{}", err);
        assert!(display.contains("NOT_FOUND"));
        assert!(display.contains("item not found"));
    }

    #[test]
    fn test_result_ext() {
        let ok_result: Result<i32> = Ok(42);
        assert!(ok_result.is_ok());
        assert!(ok_result.status().is_none());

        let err_result: Result<i32> = Err(ScannError::internal("test"));
        assert!(err_result.is_err());
        assert!(err_result.status().is_some());
    }
}
