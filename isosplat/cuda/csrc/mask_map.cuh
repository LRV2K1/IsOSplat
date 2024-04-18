#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void extract_segment_features(
    const int num_points,
    const dim3 mask_size,
    const bool* __restrict__ mask,
    const int* __restrict__ xys,
    bool* __restrict__ xy_mask
);

__global__ void padd_features(
    const int num_points,
    const int kernel_size,
    const dim3 mask_size,
    const int* __restrict__ xys,
    const float* __restrict__ kernel,
    float* __restrict__ mask
);