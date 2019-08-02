#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <time.h>
#include <cuda_runtime.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__device__ long store_now[1];

__global__ void BusyLoopKernel(const int* num_delay_cycles, int* out) {
    // As shown in https://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel
    clock_t start = clock();
    
    for(;;) {
        clock_t now = clock();
        // Adjust for overflow
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= num_delay_cycles[0]) {
            break;
        }
        // Prevent nvcc optimizations 
        store_now[0] = cycles;
    }

    start = clock();
    
    for(;;) {
        clock_t now = clock();
        // Adjust for overflow
        clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= num_delay_cycles[0]) {
            break;
        }
        // Prevent nvcc optimizations 
        store_now[0] = cycles;
    }
}

void BusyLoopKernelLauncher(const Eigen::GpuDevice& device, const int* num_delay_cycles, int* out) {
  auto stream = device.stream();
  BusyLoopKernel<<<1, 256, 0, stream>>>(num_delay_cycles, out);
}

#endif
