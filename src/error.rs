use std::sync::Arc;
use std::{error, fmt, result};

#[derive(Debug)]
pub enum CudaError {
    Runtime(i32),
    InvalidParams(String),
    Device(String),
    Memory(String),
    Stream(String),
    Kernel(String),
    Initialization(String),
    Synchronization(String),
    Context(Arc<CudaError>, String),
}

impl error::Error for CudaError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            CudaError::Context(err, _) => Some(err.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::Runtime(code) => write!(f, "CUDA runtime error: {}", code),
            CudaError::InvalidParams(msg) => write!(f, "Invalid parameters: {}", msg),
            CudaError::Device(msg) => write!(f, "Device error: {}", msg),
            CudaError::Memory(msg) => write!(f, "Memory error: {}", msg),
            CudaError::Stream(msg) => write!(f, "Stream error: {}", msg),
            CudaError::Kernel(msg) => write!(f, "Kernel error: {}", msg),
            CudaError::Initialization(msg) => write!(f, "Initialization error: {}", msg),
            CudaError::Synchronization(msg) => write!(f, "Synchronization error: {}", msg),
            CudaError::Context(err, ctx) => write!(f, "{}: {}", ctx, err),
        }
    }
}

pub type Result<T> = result::Result<T, CudaError>;

impl From<std::io::Error> for CudaError {
    fn from(error: std::io::Error) -> Self {
        CudaError::Device(error.to_string())
    }
}

pub trait CudaErrorExt {
    fn with_context<F, S>(self, f: F) -> Self
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T> CudaErrorExt for Result<T> {
    fn with_context<F, S>(self, f: F) -> Self
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|err| CudaError::Context(Arc::new(err), f().into()))
    }
}

#[macro_export]
macro_rules! cuda_try {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => {
                return Err(CudaError::Context(
                    Arc::new(err),
                    format!("at {}:{}", file!(), line!()),
                ))
            }
        }
    };
    ($expr:expr, $msg:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => return Err(CudaError::Context(Arc::new(err), $msg.into())),
        }
    };
}

#[macro_export]
macro_rules! cuda_ensure {
    ($cond:expr, $err:expr) => {
        if !$cond {
            return Err($err);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CudaError::Runtime(1);
        assert_eq!(err.to_string(), "CUDA runtime error: 1");

        let err = CudaError::InvalidParams("test".into());
        assert_eq!(err.to_string(), "Invalid parameters: test");

        let err = CudaError::Context(
            Arc::new(CudaError::Memory("out of memory".into())),
            "allocation failed".into(),
        );
        assert_eq!(
            err.to_string(),
            "allocation failed: Memory error: out of memory"
        );
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "test");
        let cuda_err: CudaError = io_err.into();
        assert!(matches!(cuda_err, CudaError::Device(_)));
    }

    #[test]
    fn test_with_context() {
        let result: Result<()> = Err(CudaError::Memory("test".into()));
        let result = result.with_context(|| "context");
        assert!(matches!(result, Err(CudaError::Context(_, _))));
    }

    #[test]
    fn test_cuda_try() {
        fn inner() -> Result<()> {
            cuda_try!(Err::<(), _>(CudaError::Memory("test".into())));
            Ok(())
        }
        assert!(matches!(inner(), Err(CudaError::Context(_, _))));

        fn inner_with_msg() -> Result<()> {
            cuda_try!(
                Err::<(), _>(CudaError::Memory("test".into())),
                "custom message"
            );
            Ok(())
        }
        assert!(matches!(inner_with_msg(), Err(CudaError::Context(_, _))));
    }

    #[test]
    fn test_cuda_ensure() {
        fn inner() -> Result<()> {
            cuda_ensure!(false, CudaError::InvalidParams("test".into()));
            Ok(())
        }
        assert!(matches!(inner(), Err(CudaError::InvalidParams(_))));
    }
}
