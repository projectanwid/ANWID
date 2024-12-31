use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
};

use rust_cuda::{
    cuda_runtime::{self, check_cuda_error, cudaError_t},
    CudaConfig, CudaContext, CudaError, CudaMemory, Result,
};

#[repr(C, align(8))]
struct ConnectionState {
    last_active: i64,
    client_id: i32,
    active: bool,
    padding: [u8; 3],
}

#[repr(C)]
struct ServerState {
    connections: *mut ConnectionState,
    capacity: usize,
    active_connections: i32,
    total_requests: i32,
    next_client_id: i32,
}

impl Drop for ServerState {
    fn drop(&mut self) {
        unsafe {
            if !self.connections.is_null() {
                let _ = cuda_runtime::cudaFree(self.connections as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl Send for ServerState {}
unsafe impl Sync for ServerState {}

#[link(name = "anwid_kernel", kind = "static")]
extern "system" {
    fn initialize_gpu_server(state: *mut *mut ServerState, initial_capacity: usize) -> cudaError_t;
    fn launch_gpu_server(
        request_data: *mut *mut i8,
        request_lens: *mut i32,
        response_data: *mut *mut i8,
        response_lens: *mut i32,
        state: *mut ServerState,
        num_requests: i32,
    ) -> cudaError_t;
}

struct GpuServerState {
    _context: CudaContext,
    state: *mut ServerState,
    request_buffer: CudaMemory<*mut i8>,
    request_lens: CudaMemory<i32>,
    response_buffer: CudaMemory<*mut i8>,
    response_lens: CudaMemory<i32>,
    request_data: CudaMemory<i8>,
    response_data: CudaMemory<i8>,
}

unsafe impl Send for GpuServerState {}
unsafe impl Sync for GpuServerState {}

struct ThreadSafeGpuServer {
    inner: Arc<Mutex<GpuServerState>>,
}

impl ThreadSafeGpuServer {
    fn new() -> Result<Self> {
        let config = CudaConfig::default();
        let context = CudaContext::new(config.device_id, config.max_batch_size)?;
        let mut state_ptr = std::ptr::null_mut();
        
        unsafe {
            check_cuda_error(initialize_gpu_server(&mut state_ptr, config.max_batch_size))
                .map_err(|e| CudaError::Initialization(format!("Failed to initialize GPU server: {:?}", e)))?;
            
            if state_ptr.is_null() {
                return Err(CudaError::InvalidParams("GPU server state initialization failed".to_string()));
            }
        }

        fn alloc_with_cleanup<T>(size: usize, state_ptr: *mut ServerState) -> Result<CudaMemory<T>> {
            unsafe {
                CudaMemory::new_device(size).map_err(|e| {
                    if !state_ptr.is_null() {
                        drop(Box::from_raw(state_ptr));
                    }
                    e
                })
            }
        }

        let request_data = alloc_with_cleanup::<i8>(8192, state_ptr)?;
        let response_data = alloc_with_cleanup::<i8>(1024 * 1024, state_ptr)?;
        let request_buffer = alloc_with_cleanup::<*mut i8>(1, state_ptr)?;
        let response_buffer = alloc_with_cleanup::<*mut i8>(1, state_ptr)?;
        let request_lens = alloc_with_cleanup::<i32>(1, state_ptr)?;
        let response_lens = alloc_with_cleanup::<i32>(1, state_ptr)?;

        let gpu_state = GpuServerState {
            _context: context,
            state: state_ptr,
            request_buffer,
            request_lens,
            response_buffer,
            response_lens,
            request_data,
            response_data,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(gpu_state)),
        })
    }

    fn handle_request(&self, request: &[u8], _thread_id: i32) -> Result<Vec<u8>> {
        let state = self.inner.lock().unwrap();

        unsafe {
            let zeros = vec![0i8; 1024 * 1024];
            state.response_data.copy_from_host(&zeros)?;

            let request_i8: Vec<i8> = request.iter().map(|&x| x as i8).collect();
            let request_slice = if request_i8.len() > 8192 {
                &request_i8[..8192]
            } else {
                &request_i8
            };
            state.request_data.copy_from_host(request_slice)?;

            let request_ptr = state.request_data.as_ptr();
            state.request_buffer.copy_from_host(&[request_ptr])?;
            state.request_lens.copy_from_host(&[request_slice.len() as i32])?;

            let response_ptr = state.response_data.as_ptr();
            state.response_buffer.copy_from_host(&[response_ptr])?;
            state.response_lens.copy_from_host(&[0])?;

            state._context.synchronize()?;

            check_cuda_error(launch_gpu_server(
                state.request_buffer.as_ptr(),
                state.request_lens.as_ptr(),
                state.response_buffer.as_ptr(),
                state.response_lens.as_ptr(),
                state.state,
                1,
            ))?;

            state._context.synchronize()?;

            let mut response_len = [0i32];
            state.response_lens.copy_to_host(&mut response_len)?;

            if response_len[0] <= 0 {
                return Err(CudaError::InvalidParams("Invalid response length".into()));
            }

            let mut response = vec![0i8; response_len[0] as usize];
            state.response_data.copy_to_host(&mut response)?;

            let response_bytes: Vec<u8> = response.iter().map(|&x| x as u8).collect();
            let response_str = std::str::from_utf8(&response_bytes)
                .map_err(|_| CudaError::InvalidParams("Invalid UTF-8 in response".into()))?;

            Ok(response_str.as_bytes().to_vec())
        }
    }
}

impl Drop for GpuServerState {
    fn drop(&mut self) {
        unsafe {
            if !self.state.is_null() {
                let _ = cuda_runtime::cudaFree(self.state as *mut std::ffi::c_void);
            }
        }
    }
}

impl Clone for ThreadSafeGpuServer {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

fn handle_client(mut stream: TcpStream, server: Arc<ThreadSafeGpuServer>) -> Result<()> {
    let mut buffer = [0; 8192];
    let n = stream.read(&mut buffer)?;

    if n == 0 {
        return Ok(());
    }

    let thread_id = 0;
    let response = server.handle_request(&buffer[..n], thread_id)?;

    let response_str = String::from_utf8_lossy(&response);
    let response_with_headers = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        response_str.len(),
        response_str
    );

    stream.write_all(response_with_headers.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let server = Arc::new(ThreadSafeGpuServer::new()?);
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    println!("Server running on http://127.0.0.1:8080");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let server = Arc::clone(&server);
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, server) {
                        eprintln!("Error handling client: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Error accepting connection: {}", e),
        }
    }

    Ok(())
}
