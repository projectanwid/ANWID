#include <cuda_runtime.h>
#include <cstddef>

extern "C" {
struct alignas(8) ConnectionState
{
    long long last_active;
    int client_id;
    bool active;
    char padding[3];
};

struct ServerState
{
    ConnectionState* connections;
    std::size_t capacity;
    int active_connections;
    int total_requests;
    int next_client_id;
};

class DeviceMemory
{
    ConnectionState* connections;
    ServerState* state;
    std::size_t capacity;

    __host__ static bool checkCudaError(const cudaError_t error)
    {
        return error == cudaSuccess;
    }

public:
    __host__ explicit DeviceMemory(const std::size_t initial_capacity)
        : connections(nullptr), state(nullptr), capacity(initial_capacity)
    {
    }

    __host__ ~DeviceMemory()
    {
        cleanup();
    }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    __host__ bool allocate()
    {
        if (!checkCudaError(cudaMalloc(&connections, sizeof(ConnectionState) * capacity)))
        {
            cleanup();
            return false;
        }

        if (!checkCudaError(cudaMemset(connections, 0, sizeof(ConnectionState) * capacity)))
        {
            cleanup();
            return false;
        }

        if (!checkCudaError(cudaMalloc(&state, sizeof(ServerState))))
        {
            cleanup();
            return false;
        }

        return true;
    }

    __host__ bool initialize_state()
    {
        if (!connections || !state)
            return false;

        const ServerState init_state = {
            connections,
            capacity,
            0,
            0,
            0
        };

        if (!checkCudaError(cudaMemcpy(state, &init_state, sizeof(ServerState), cudaMemcpyHostToDevice)))
        {
            cleanup();
            return false;
        }

        return true;
    }

    __host__ void cleanup()
    {
        if (connections)
        {
            const cudaError_t error = cudaFree(connections);
            if (error != cudaSuccess)
            {
            }
            connections = nullptr;
        }
        if (state)
        {
            const cudaError_t error = cudaFree(state);
            if (error != cudaSuccess)
            {
            }
            state = nullptr;
        }
    }

    __host__ ServerState* release_state()
    {
        if (!state)
        {
            return nullptr;
        }
        ServerState* temp = state;
        state = nullptr;
        connections = nullptr;
        return temp;
    }

    __host__ ConnectionState* get_connections() const { return connections; }
    __host__ ServerState* get_state() const { return state; }
    __host__ std::size_t get_capacity() const { return capacity; }
};

__device__ void writeChar(char* output, int* pos, const char c, const int max_len)
{
    if (output && pos && *pos < max_len - 1)
    {
        output[*pos] = c;
        ++*pos;
    }
}

__device__ void writeString(char* output, int* pos, const char* str, const int max_len)
{
    if (!output || !pos || !str || *pos >= max_len - 1)
        return;

    while (*str && *pos < max_len - 1)
    {
        writeChar(output, pos, *str++, max_len);
    }
}

__device__ void writePositiveInt(char* output, int* pos, const int num, const int max_len)
{
    if (!output || !pos || *pos >= max_len - 1) return;

    if (num == 0)
    {
        writeChar(output, pos, '0', max_len);
        return;
    }

    char buffer[16];
    int idx = 0;

    int n = num;
    while (n > 0 && idx < 15)
    {
        buffer[idx] = static_cast<char>('0' + n % 10);
        idx++;
        n /= 10;
    }
    buffer[idx] = '\0';

    while (idx > 0 && *pos < max_len - 1)
    {
        idx--;
        writeChar(output, pos, buffer[idx], max_len);
    }
}

__device__ void writeInt(char* output, int* pos, const int num, const int max_len)
{
    if (!output || !pos || *pos >= max_len - 1) return;

    if (num < 0)
    {
        writeChar(output, pos, '-', max_len);
        writePositiveInt(output, pos, -num, max_len);
    }
    else
    {
        writePositiveInt(output, pos, num, max_len);
    }
}

__device__ void generateStyles(char* output, int* pos, const int max_len)
{
    if (!output || !pos || *pos >= max_len - 1)
        return;

    writeString(output, pos,
                "<style>body{font-family:Arial,sans-serif;margin:0;padding:20px;background:#1a1a1a;color:#fff}"
                ".message{font-size:24px;text-align:center;padding:40px;background:#2d2d2d;border-radius:8px;margin:20px 0}"
                ".container{max-width:800px;margin:0 auto}</style>",
                max_len);
}

__device__ void generateContent(char* output, int* pos, const ServerState* state, const int thread_id, const int max_len)
{
    if (!output || !pos || !state || *pos >= max_len - 1)
        return;

    writeString(output, pos, "<div class='message'>ANWID says hello from thread ", max_len);
    writeInt(output, pos, thread_id, max_len);
    writeString(output, pos, "</div>", max_len);
}

__global__ void gpu_server_kernel(char** const, int* const,
                                  char** const response_data, int* const response_lens,
                                  const ServerState* const state, const int num_requests)
{
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= num_requests || !response_data || !response_lens || !state)
        return;

    char* current_response = response_data[tid];
    if (!current_response)
        return;

    constexpr int max_response_len = 1024 * 1024;
    int pos = 0;

    writeString(current_response, &pos, "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n", max_response_len);
    writeString(current_response, &pos, "<meta charset=\"UTF-8\">\n", max_response_len);
    writeString(current_response, &pos, "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n", max_response_len);
    writeString(current_response, &pos, "<title>GPU Web Server</title>\n", max_response_len);
    generateStyles(current_response, &pos, max_response_len);
    writeString(current_response, &pos, "</head>\n<body>\n<div class='container'>\n", max_response_len);
    writeString(current_response, &pos, "<h1>GPU-Powered Web Server</h1>\n", max_response_len);
    generateContent(current_response, &pos, state, tid, max_response_len);
    writeString(current_response, &pos, "</div>\n</body>\n</html>", max_response_len);

    if (pos < max_response_len)
    {
        current_response[pos] = '\0';
        response_lens[tid] = pos;
    }
    else
    {
        current_response[max_response_len - 1] = '\0';
        response_lens[tid] = max_response_len - 1;
    }
}

__host__ cudaError_t initialize_gpu_server(ServerState** out_state, const std::size_t initial_capacity)
{
    if (!out_state || initial_capacity == 0)
        return cudaErrorInvalidValue;

    *out_state = nullptr;
    DeviceMemory mem(initial_capacity);

    if (!mem.allocate())
        return cudaErrorMemoryAllocation;

    if (!mem.initialize_state())
        return cudaErrorMemoryAllocation;

    *out_state = mem.release_state();
    return cudaSuccess;
}

__host__ cudaError_t cleanup_gpu_server(ServerState* state)
{
    if (!state)
        return cudaSuccess;

    cudaError_t final_error = cudaSuccess;
    ServerState host_state = {};

    const cudaError_t copy_err = cudaMemcpy(&host_state, state, sizeof(ServerState), cudaMemcpyDeviceToHost);
    if (copy_err != cudaSuccess)
    {
        final_error = copy_err;
    }
    else if (host_state.connections)
    {
        const cudaError_t free_err = cudaFree(host_state.connections);
        if (free_err != cudaSuccess)
            final_error = free_err;
    }

    const cudaError_t state_err = cudaFree(state);
    if (state_err != cudaSuccess && final_error == cudaSuccess)
        final_error = state_err;

    return final_error;
}

__host__ cudaError_t launch_gpu_server(char** const request_data, int* const request_lens,
                                       char** const response_data, int* const response_lens,
                                       const ServerState* const state, const int num_requests)
{
    if (!request_data || !request_lens || !response_data || !response_lens || !state || num_requests <= 0)
        return cudaErrorInvalidValue;

    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = (num_requests + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gpu_server_kernel<<<num_blocks, BLOCK_SIZE>>>(request_data, request_lens,
                                                  response_data, response_lens,
                                                  state, num_requests);
    return cudaGetLastError();
}
}
