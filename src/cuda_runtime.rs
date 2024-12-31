#![allow(non_camel_case_types)]

use crate::error::{CudaError, CudaErrorExt, Result};
use std::ffi::c_void;

pub type cudaError_t = i32;
pub type size_t = usize;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;

pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_STREAM_DEFAULT: cudaStream_t = std::ptr::null_mut();
pub const CUDA_CPU_DEVICE_ID: i32 = -1;
pub const CUDA_MEM_ATTACH_GLOBAL: u32 = 1;

#[repr(C)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

#[link(name = "cudart")]
extern "system" {
    // Memory Management
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: size_t) -> cudaError_t;
    pub fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: size_t) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMallocManaged(dev_ptr: *mut *mut c_void, size: size_t, flags: u32) -> cudaError_t;

    // Memory Operations
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemPrefetchAsync(
        dev_ptr: *const c_void,
        count: size_t,
        dst_device: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemAdvise(
        dev_ptr: *const c_void,
        count: size_t,
        advice: i32,
        device: i32,
    ) -> cudaError_t;

    // Device Management
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaSetDevice(device: i32) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut i32) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> cudaError_t;

    // Stream Management
    pub fn cudaStreamCreate(p_stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(p_stream: *mut cudaStream_t, flags: u32) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(
        p_stream: *mut cudaStream_t,
        flags: u32,
        priority: i32,
    ) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: u32)
        -> cudaError_t;

    // Event Management
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: u32) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
pub struct cudaDeviceProp {
    pub name: [i8; 256],
    pub totalGlobalMem: size_t,
    pub sharedMemPerBlock: size_t,
    pub regsPerBlock: i32,
    pub warpSize: i32,
    pub maxThreadsPerBlock: i32,
    pub maxThreadsDim: [i32; 3],
    pub maxGridSize: [i32; 3],
    pub clockRate: i32,
    pub memoryClockRate: i32,
    pub memoryBusWidth: i32,
    pub totalConstMem: size_t,
    pub major: i32,
    pub minor: i32,
    pub multiProcessorCount: i32,
    pub computeMode: i32,
    pub maxThreadsPerMultiProcessor: i32,
    pub memPitch: size_t,
    pub textureAlignment: size_t,
    pub texturePitchAlignment: size_t,
    pub kernelExecTimeoutEnabled: i32,
    pub integrated: i32,
    pub canMapHostMemory: i32,
    pub concurrentKernels: i32,
    pub ECCEnabled: i32,
    pub pciBusID: i32,
    pub pciDeviceID: i32,
    pub pciDomainID: i32,
    pub tccDriver: i32,
    pub asyncEngineCount: i32,
    pub unifiedAddressing: i32,
    pub memoryPoolsSupported: i32,
    pub pageableMemoryAccess: i32,
    pub concurrentManagedAccess: i32,
}

pub unsafe fn check_cuda_error(error: cudaError_t) -> Result<()> {
    if error != CUDA_SUCCESS {
        Err(CudaError::Runtime(error))
    } else {
        Ok(())
    }
}

pub unsafe fn get_device_properties(device: i32) -> Result<cudaDeviceProp> {
    let mut props = std::mem::zeroed();
    check_cuda_error(cudaGetDeviceProperties(&mut props, device))
        .with_context(|| "Failed to get device properties")?;
    Ok(props)
}

pub unsafe fn get_device_count() -> Result<i32> {
    let mut count = 0;
    check_cuda_error(cudaGetDeviceCount(&mut count))
        .with_context(|| "Failed to get device count")?;
    Ok(count)
}

pub unsafe fn get_current_device() -> Result<i32> {
    let mut device = 0;
    check_cuda_error(cudaGetDevice(&mut device)).with_context(|| "Failed to get current device")?;
    Ok(device)
}
