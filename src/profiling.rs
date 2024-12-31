use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use crate::{cuda_runtime, CudaError, CudaStream, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventFlags {
    Default = 0x0,
    BlockingSync = 0x1,
    DisableTiming = 0x2,
    Interprocess = 0x4,
}

pub struct CudaEvent {
    event: cuda_runtime::cudaEvent_t,
    flags: EventFlags,
    active: AtomicBool,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self> {
        Self::with_flags(EventFlags::Default)
    }

    pub fn with_flags(flags: EventFlags) -> Result<Self> {
        let mut event = std::ptr::null_mut();
        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaEventCreateWithFlags(
                &mut event,
                flags as u32,
            ))?;
        }

        Ok(Self {
            event,
            flags,
            active: AtomicBool::new(true),
        })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(CudaError::Device("Event is not active".into()));
        }

        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaEventRecord(
                self.event,
                stream.as_ptr(),
            ))
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        if !self.active.load(Ordering::SeqCst) {
            return Err(CudaError::Device("Event is not active".into()));
        }

        unsafe { cuda_runtime::check_cuda_error(cuda_runtime::cudaEventSynchronize(self.event)) }
    }

    pub fn elapsed_time(&self, end: &CudaEvent) -> Result<Duration> {
        if !self.active.load(Ordering::SeqCst) || !end.active.load(Ordering::SeqCst) {
            return Err(CudaError::Device(
                "One or both events are not active".into(),
            ));
        }

        let mut ms = 0.0f32;
        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaEventElapsedTime(
                &mut ms, self.event, end.event,
            ))?;
        }

        Ok(Duration::from_secs_f32(ms / 1000.0))
    }

    pub fn as_ptr(&self) -> cuda_runtime::cudaEvent_t {
        self.event
    }

    pub fn flags(&self) -> EventFlags {
        self.flags
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if self.active.load(Ordering::SeqCst) {
            unsafe {
                let _ = cuda_runtime::cudaEventDestroy(self.event);
            }
            self.active.store(false, Ordering::SeqCst);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimingStats {
    pub elapsed: Duration,
    pub kernel_time: Duration,
    pub memory_transfer_time: Duration,
}

pub struct CudaTimer {
    start: CudaEvent,
    end: CudaEvent,
    running: bool,
}

impl CudaTimer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            start: CudaEvent::new()?,
            end: CudaEvent::new()?,
            running: false,
        })
    }

    pub fn start(&mut self, stream: &CudaStream) -> Result<()> {
        if self.running {
            return Err(CudaError::InvalidParams("Timer is already running".into()));
        }
        self.start.record(stream)?;
        self.running = true;
        Ok(())
    }

    pub fn stop(&mut self, stream: &CudaStream) -> Result<Duration> {
        if !self.running {
            return Err(CudaError::InvalidParams("Timer is not running".into()));
        }
        self.end.record(stream)?;
        self.end.synchronize()?;
        self.running = false;
        self.start.elapsed_time(&self.end)
    }

    pub fn is_running(&self) -> bool {
        self.running
    }
}

pub struct ProfilingSession {
    events: HashMap<String, (CudaEvent, CudaEvent)>,
    timings: HashMap<String, Duration>,
}

impl ProfilingSession {
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
            timings: HashMap::new(),
        }
    }

    pub fn mark_event(&mut self, name: &str, stream: &CudaStream) -> Result<()> {
        if !self.events.contains_key(name) {
            let start = CudaEvent::new()?;
            let end = CudaEvent::new()?;
            start.record(stream)?;
            self.events.insert(name.to_string(), (start, end));
        } else {
            let (_, end) = self.events.get(name).unwrap();
            end.record(stream)?;
            end.synchronize()?;

            let (start, end) = self.events.get(name).unwrap();
            let duration = start.elapsed_time(end)?;
            self.timings.insert(name.to_string(), duration);
        }
        Ok(())
    }

    pub fn get_timings(&self) -> Result<Vec<(String, Duration)>> {
        let mut result = Vec::new();
        for (name, duration) in &self.timings {
            result.push((name.clone(), *duration));
        }
        Ok(result)
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.timings.clear();
    }
}
