use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{cuda_runtime, CudaError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Host,
    Device,
    Unified,
    Pinned,
}

pub struct CudaMemoryManager {
    total_allocated: usize,
    max_allocation: usize,
    active_allocations: Vec<(NonNull<u8>, usize)>,
}

impl CudaMemoryManager {
    pub fn new(max_allocation: usize) -> Self {
        Self {
            total_allocated: 0,
            max_allocation,
            active_allocations: Vec::new(),
        }
    }

    pub fn allocate<T>(&mut self, count: usize, memory_type: MemoryType) -> Result<CudaMemory<T>> {
        let size = count * size_of::<T>();
        if self.total_allocated + size > self.max_allocation {
            return Err(CudaError::Memory("Exceeded maximum allocation".into()));
        }

        let ptr = unsafe {
            match memory_type {
                MemoryType::Device => {
                    let mut ptr = std::ptr::null_mut();
                    cuda_runtime::check_cuda_error(cuda_runtime::cudaMalloc(&mut ptr, size))?;
                    NonNull::new(ptr as *mut T).unwrap()
                }
                MemoryType::Host => {
                    let mut ptr = std::ptr::null_mut();
                    cuda_runtime::check_cuda_error(cuda_runtime::cudaMallocHost(&mut ptr, size))?;
                    NonNull::new(ptr as *mut T).unwrap()
                }
                MemoryType::Unified => {
                    let mut ptr = std::ptr::null_mut();
                    cuda_runtime::check_cuda_error(cuda_runtime::cudaMallocManaged(
                        &mut ptr,
                        size,
                        cuda_runtime::CUDA_MEM_ATTACH_GLOBAL,
                    ))?;
                    NonNull::new(ptr as *mut T).unwrap()
                }
                MemoryType::Pinned => {
                    let mut ptr = std::ptr::null_mut();
                    cuda_runtime::check_cuda_error(cuda_runtime::cudaMallocHost(&mut ptr, size))?;
                    NonNull::new(ptr as *mut T).unwrap()
                }
            }
        };

        self.total_allocated += size;
        self.active_allocations.push((ptr.cast(), size));

        Ok(CudaMemory {
            ptr,
            size,
            memory_type,
            initialized: AtomicBool::new(false),
            _phantom: PhantomData,
        })
    }
}

pub struct CudaMemory<T> {
    ptr: NonNull<T>,
    size: usize,
    memory_type: MemoryType,
    initialized: AtomicBool,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaMemory<T> {}
unsafe impl<T: Sync> Sync for CudaMemory<T> {}

impl<T> CudaMemory<T> {
    pub unsafe fn new_device(count: usize) -> Result<Self> {
        let mut manager = CudaMemoryManager::new(usize::MAX);
        manager.allocate(count, MemoryType::Device)
    }

    pub unsafe fn new_host(count: usize) -> Result<Self> {
        let mut manager = CudaMemoryManager::new(usize::MAX);
        manager.allocate(count, MemoryType::Host)
    }

    pub unsafe fn new_unified(count: usize) -> Result<Self> {
        let mut manager = CudaMemoryManager::new(usize::MAX);
        manager.allocate(count, MemoryType::Unified)
    }

    pub unsafe fn new_with_type(count: usize, memory_type: MemoryType) -> Result<Self> {
        let mut manager = CudaMemoryManager::new(usize::MAX);
        manager.allocate(count, memory_type)
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.size / size_of::<T>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }

    pub fn copy_from_host(&self, src: &[T]) -> Result<()>
    where
        T: Copy,
    {
        if src.len() > self.len() {
            return Err(CudaError::InvalidParams("Source slice too large".into()));
        }

        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaMemcpy(
                self.as_ptr() as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                src.len() * size_of::<T>(),
                cuda_runtime::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }

        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()>
    where
        T: Copy,
    {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(CudaError::Memory("Memory not initialized".into()));
        }

        if dst.len() > self.len() {
            return Err(CudaError::InvalidParams(
                "Destination slice too large".into(),
            ));
        }

        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.as_ptr() as *const std::ffi::c_void,
                dst.len() * size_of::<T>(),
                cuda_runtime::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))
        }
    }

    pub fn prefetch_to_device(&self, device_id: i32) -> Result<()> {
        if self.memory_type != MemoryType::Unified {
            return Err(CudaError::InvalidParams(
                "Only unified memory can be prefetched".into(),
            ));
        }

        unsafe {
            cuda_runtime::check_cuda_error(cuda_runtime::cudaMemPrefetchAsync(
                self.as_ptr() as *const std::ffi::c_void,
                self.size,
                device_id,
                std::ptr::null_mut(),
            ))
        }
    }
}

impl<T> Drop for CudaMemory<T> {
    fn drop(&mut self) {
        unsafe {
            match self.memory_type {
                MemoryType::Device | MemoryType::Unified => {
                    let _ = cuda_runtime::cudaFree(self.ptr.as_ptr() as *mut std::ffi::c_void);
                }
                MemoryType::Host | MemoryType::Pinned => {
                    let _ = cuda_runtime::cudaFreeHost(self.ptr.as_ptr() as *mut std::ffi::c_void);
                }
            }
        }
    }
}

impl<T> Deref for CudaMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }
}

impl<T> DerefMut for CudaMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }
}
