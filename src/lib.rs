pub mod config;
pub mod cuda_runtime;
pub mod error;
pub mod memory;
pub mod profiling;
pub mod stream;

pub use config::CudaConfig;
pub use error::{CudaError, Result};
pub use memory::{CudaMemory, MemoryType};
pub use profiling::{CudaEvent, CudaTimer, EventFlags, ProfilingSession, TimingStats};
pub use stream::{CudaStream, StreamFlags, StreamGuard};

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (i32, i32),
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub warp_size: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub max_shared_memory_per_block: usize,
    pub max_registers_per_block: i32,
    pub multi_processor_count: i32,
}

pub struct CudaDevice {
    id: i32,
    initialized: AtomicBool,
    properties: Option<DeviceProperties>,
}

impl CudaDevice {
    pub fn new(id: i32) -> Result<Self> {
        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaSetDevice(id))?;
            let props = cuda_runtime::get_device_properties(id)?;

            let name = String::from_utf8(
                props
                    .name
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| CudaError::Device("Invalid device name".into()))?;

            let properties = DeviceProperties {
                name,
                total_memory: props.totalGlobalMem,
                compute_capability: (props.major, props.minor),
                max_threads_per_block: props.maxThreadsPerBlock,
                max_threads_per_multiprocessor: props.maxThreadsPerMultiProcessor,
                warp_size: props.warpSize,
                memory_clock_rate: props.memoryClockRate,
                memory_bus_width: props.memoryBusWidth,
                l2_cache_size: props.totalConstMem as i32,
                max_shared_memory_per_block: props.sharedMemPerBlock,
                max_registers_per_block: props.regsPerBlock,
                multi_processor_count: props.multiProcessorCount,
            };

            Ok(Self {
                id,
                initialized: AtomicBool::new(true),
                properties: Some(properties),
            })
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(CudaError::Device("Device not initialized".into()));
        }
        unsafe { cuda_runtime::check_cuda_error(cuda_runtime::cudaDeviceSynchronize()) }
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn properties(&self) -> Option<&DeviceProperties> {
        self.properties.as_ref()
    }

    pub fn create_stream(&self) -> Result<CudaStream> {
        CudaStream::new()
    }

    pub fn create_stream_with_priority(&self, priority: i32) -> Result<CudaStream> {
        CudaStream::with_priority(StreamFlags::Default, priority)
    }

    pub fn allocate_memory<T>(
        &self,
        size: usize,
        memory_type: MemoryType,
    ) -> Result<CudaMemory<T>> {
        unsafe { CudaMemory::new_with_type(size, memory_type) }
    }
}

pub struct BatchProcessor {
    device: CudaDevice,
    stream: CudaStream,
    max_batch_size: usize,
    profiling: bool,
    current_session: Option<ProfilingSession>,
}

impl BatchProcessor {
    pub fn new(device_id: i32, max_batch_size: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)?;
        let stream = CudaStream::new()?;
        Ok(Self {
            device,
            stream,
            max_batch_size,
            profiling: false,
            current_session: None,
        })
    }

    pub fn with_profiling(mut self) -> Self {
        self.profiling = true;
        self.current_session = Some(ProfilingSession::new());
        self
    }

    pub fn process_batch<T, U>(
        &mut self,
        inputs: &[T],
        outputs: &mut [U],
    ) -> Result<Option<TimingStats>>
    where
        T: Copy,
        U: Copy,
    {
        if inputs.len() > self.max_batch_size {
            return Err(CudaError::InvalidParams("Batch size too large".into()));
        }
        if inputs.len() != outputs.len() {
            return Err(CudaError::InvalidParams(
                "Input and output size mismatch".into(),
            ));
        }

        if self.profiling {
            if let Some(session) = &mut self.current_session {
                session.mark_event("batch_start", &self.stream)?;
                // Process batch here
                session.mark_event("batch_end", &self.stream)?;
                let timings = session.get_timings()?;
                if let Some((_, duration)) = timings.first() {
                    return Ok(Some(TimingStats {
                        elapsed: *duration,
                        kernel_time: *duration,
                        memory_transfer_time: Duration::from_secs(0),
                    }));
                }
            }
        }

        self.stream.synchronize()?;
        Ok(None)
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

pub struct CudaContext {
    pub device: CudaDevice,
    pub stream: CudaStream,
    pub batch_processor: BatchProcessor,
    profiling_session: Option<ProfilingSession>,
}

impl CudaContext {
    pub fn new(device_id: i32, max_batch_size: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)?;
        let stream = CudaStream::new()?;
        let batch_processor = BatchProcessor::new(device_id, max_batch_size)?;
        Ok(Self {
            device,
            stream,
            batch_processor,
            profiling_session: None,
        })
    }

    pub fn with_profiling(mut self) -> Self {
        self.profiling_session = Some(ProfilingSession::new());
        self
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        self.stream.synchronize()
    }

    pub fn get_profiling_results(&self) -> Result<Option<Vec<(String, Duration)>>> {
        Ok(self
            .profiling_session
            .as_ref()
            .map(|session| session.get_timings())
            .transpose()?)
    }
}
