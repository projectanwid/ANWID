use crate::{cuda_runtime, CudaError, Result};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFlags {
    Default = 0x0,
    NonBlocking = 0x1,
    PriorityHigh = 0x2,
    PriorityLow = 0x4,
}

pub struct CudaStream {
    stream: cuda_runtime::cudaStream_t,
    flags: StreamFlags,
    priority: i32,
    active: AtomicBool,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self> {
        Self::with_flags(StreamFlags::Default)
    }

    pub fn with_flags(flags: StreamFlags) -> Result<Self> {
        let mut stream = std::ptr::null_mut();
        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaStreamCreateWithFlags(
                &mut stream,
                flags as u32,
            ))?;
        }

        Ok(Self {
            stream,
            flags,
            priority: 0,
            active: AtomicBool::new(true),
        })
    }

    pub fn with_priority(flags: StreamFlags, priority: i32) -> Result<Self> {
        let mut stream = std::ptr::null_mut();
        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaStreamCreateWithPriority(
                &mut stream,
                flags as u32,
                priority,
            ))?;
        }

        Ok(Self {
            stream,
            flags,
            priority,
            active: AtomicBool::new(true),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(CudaError::Stream("Stream is not active".into()));
        }

        unsafe { cuda_runtime::check_cuda_error(cuda_runtime::cudaStreamSynchronize(self.stream)) }
    }

    pub fn wait_event(&self, event: cuda_runtime::cudaEvent_t) -> Result<()> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(CudaError::Stream("Stream is not active".into()));
        }

        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaStreamWaitEvent(self.stream, event, 0))
        }
    }

    pub fn as_ptr(&self) -> cuda_runtime::cudaStream_t {
        self.stream
    }

    pub fn flags(&self) -> StreamFlags {
        self.flags
    }

    pub fn priority(&self) -> i32 {
        self.priority
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.active.load(Ordering::SeqCst) {
            unsafe {
                let _ = cuda_runtime::cudaStreamSynchronize(self.stream);
                let _ = cuda_runtime::cudaStreamDestroy(self.stream);
            }
            self.active.store(false, Ordering::SeqCst);
        }
    }
}

pub struct StreamGuard<'a> {
    stream: &'a CudaStream,
}

impl<'a> StreamGuard<'a> {
    pub fn new(stream: &'a CudaStream) -> Result<Self> {
        Ok(Self { stream })
    }

    pub fn stream(&self) -> &CudaStream {
        self.stream
    }
}

impl<'a> Drop for StreamGuard<'a> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
