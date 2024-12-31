use crate::{cuda_runtime, CudaError, Result};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub struct CudaConfig {
    pub device_id: i32,
    pub max_batch_size: usize,
    pub stream_flags: u32,
    pub stream_priority: i32,
    pub memory_pool_size: usize,
    pub use_unified_memory: bool,
    pub enable_profiling: bool,
    pub enable_peer_access: bool,
    pub compute_mode: ComputeMode,
    initialized: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    Default = 0,
    Exclusive = 1,
    ProhibitDefault = 2,
    ExclusiveProcess = 3,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            max_batch_size: 1024,
            stream_flags: 0,
            stream_priority: 0,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            use_unified_memory: false,
            enable_profiling: false,
            enable_peer_access: false,
            compute_mode: ComputeMode::Default,
            initialized: AtomicBool::new(false),
        }
    }
}

impl CudaConfig {
    pub fn new(device_id: i32) -> Result<Self> {
        let config = Self {
            device_id,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_stream_flags(mut self, flags: u32) -> Self {
        self.stream_flags = flags;
        self
    }

    pub fn with_stream_priority(mut self, priority: i32) -> Self {
        self.stream_priority = priority;
        self
    }

    pub fn with_memory_pool_size(mut self, size: usize) -> Self {
        self.memory_pool_size = size;
        self
    }

    pub fn with_unified_memory(mut self, enabled: bool) -> Self {
        self.use_unified_memory = enabled;
        self
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    pub fn with_peer_access(mut self, enabled: bool) -> Self {
        self.enable_peer_access = enabled;
        self
    }

    pub fn with_compute_mode(mut self, mode: ComputeMode) -> Self {
        self.compute_mode = mode;
        self
    }

    pub fn validate(&self) -> Result<()> {
        // Check device ID
        let device_count = unsafe { cuda_runtime::get_device_count()? };
        if self.device_id < 0 || self.device_id >= device_count {
            return Err(CudaError::InvalidParams(format!(
                "Invalid device ID: {}. Available devices: {}",
                self.device_id, device_count
            )));
        }

        // Check device properties
        let props = unsafe { cuda_runtime::get_device_properties(self.device_id)? };

        // Validate batch size against device limits
        if self.max_batch_size > props.maxThreadsPerBlock as usize {
            return Err(CudaError::InvalidParams(format!(
                "Batch size {} exceeds device maximum threads per block: {}",
                self.max_batch_size, props.maxThreadsPerBlock
            )));
        }

        // Validate memory pool size against device limits
        if self.memory_pool_size > props.totalGlobalMem {
            return Err(CudaError::InvalidParams(format!(
                "Memory pool size {} exceeds device total memory: {}",
                self.memory_pool_size, props.totalGlobalMem
            )));
        }

        // Check unified memory support
        if self.use_unified_memory && props.unifiedAddressing == 0 {
            return Err(CudaError::InvalidParams(
                "Device does not support unified memory".into(),
            ));
        }

        // Check peer access support
        if self.enable_peer_access && device_count < 2 {
            return Err(CudaError::InvalidParams(
                "Peer access requires at least 2 devices".into(),
            ));
        }

        Ok(())
    }

    pub fn initialize(&self) -> Result<()> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        unsafe {
            // Set device
            cuda_runtime::check_cuda_error(cuda_runtime::cudaSetDevice(self.device_id))?;
        }

        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }
}
