#ifndef CUDA_GAUSSIAN_BLUR_LIBRARY_CUH
#define CUDA_GAUSSIAN_BLUR_LIBRARY_CUH

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h> // __syncthreads()
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <string>
#include <stdexcept>
#define min(a, b) a < b ? a : b
#define INDEX(i, j, offset) ((i) * offset) + (j)

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

inline void gpuErrChk(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(string_format("GPU Assert: %s, File: %s, Line: %d",
                                               cudaGetErrorString(code), file, line));
    }
}

#define cudaCheckError(code) { gpuErrChk(code, __FILE__, __LINE__); }

template <typename T>
__global__ void convolution2DKernel(T *in, T *out, float* kernel, int width, int height, int channels, int kSize)
{
    __shared__ float sh_mem[64][64];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int padding = kSize >> 1;
    int shared_mem_width = blockDim.x + padding*2;
    int shared_mem_height = blockDim.y + padding*2;

    if (row < 0 || row >= height || col < 0 || col >= width)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        int r_offset = blockDim.y * blockIdx.y - padding;
        int c_offset = blockDim.x * blockIdx.x - padding;

        // Fill shared memory for each block
        for (int i=0; i<shared_mem_height; i += blockDim.y)
        {
            for (int j=0; j<shared_mem_width; j += blockDim.x)
            {
                int row_ = r_offset + i + threadIdx.y;
                int col_ = c_offset + j + threadIdx.x;
                if (row_ >= 0 && row_ < height &&
                    col_ >= 0 && col_ < width)
                    sh_mem[i+threadIdx.y][j+threadIdx.x] = (float)in[(row_ * width + col_)*channels + ch];
            }
        }
        __syncthreads();

        // element wise multiplication
        float val = 0.0;
        for (int i=-padding; i<=padding; ++i)
        {
            for (int j=-padding; j<=padding; ++j)
                val += kernel[INDEX((i+padding), (j+padding), kSize)] * sh_mem[(padding+threadIdx.y+i)][(padding+threadIdx.x+j)];
        }
        out[(row*width + col)*channels + ch] = (T)val;
        __syncthreads();
    }
}

template <typename T>
__host__ void convolution2DCUDA(T *in, T *out, float *kernel, int kSize, int width, int height, int ch)
{
    int imgSize = width * height * ch;
    T* d_in = new T[imgSize];
    T* d_out = new T[imgSize];
    float * d_kernel = new float[kSize * kSize];
    cudaCheckError(cudaMalloc((void **)&d_in, sizeof(T)*imgSize));
    cudaCheckError(cudaMalloc((void **)&d_out, sizeof(T)*imgSize));
    cudaCheckError(cudaMalloc((void **)&d_kernel, sizeof(float)*kSize*kSize));

    // Copying data to device
    const uint64_t num_of_segments = 64;
    const uint64_t chunk_size = (imgSize + num_of_segments-1) / num_of_segments;
    cudaStream_t streams[num_of_segments];
    for (uint64_t st=0; st<num_of_segments; ++st)
    {
        cudaCheckError(cudaStreamCreate(&streams[st]))
        int lower = st*chunk_size;
        int upper = min(lower + chunk_size, imgSize);
        int width = upper - lower;
        cudaCheckError(cudaMemcpyAsync(&d_in[lower], &in[lower], sizeof(T)*width,
                        cudaMemcpyHostToDevice, streams[st]))
    }
    cudaCheckError(cudaMemcpy(d_kernel, kernel, sizeof(float)*kSize*kSize, cudaMemcpyHostToDevice))

    int threadsPerBlock = 32;
    int dimY = (height + threadsPerBlock - 1)/threadsPerBlock;
    int dimX = (width + threadsPerBlock - 1)/threadsPerBlock;
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(dimX, dimY);
    size_t shared_dim = (kSize/2)*2 + threadsPerBlock;
    size_t shared_mem_size = pow(shared_dim, 2) * sizeof(float);

    // call kernel function
    convolution2DKernel<<<grid, block>>>(d_in, d_out, d_kernel,
                                                         width, height, ch, kSize);
    cudaDeviceSynchronize();

    // Copying result to host
    for (uint64_t st=0; st<num_of_segments; ++st)
    {
        int lower = st*chunk_size;
        int upper = min(lower + chunk_size, imgSize);
        int width = upper - lower;
        cudaCheckError(cudaMemcpyAsync(&out[lower], &d_out[lower], sizeof(T)*width,
                                       cudaMemcpyDeviceToHost, streams[st]))
    }

    // Destroy cudaStreams
    for (int st = 0; st < num_of_segments; ++st)
        cudaCheckError(cudaStreamDestroy(streams[st]))

    // Free device memory
    cudaCheckError(cudaFree(d_in));
    cudaCheckError(cudaFree(d_out));
    cudaCheckError(cudaFree(d_kernel));
}


#endif //CUDA_GAUSSIAN_BLUR_LIBRARY_CUH
