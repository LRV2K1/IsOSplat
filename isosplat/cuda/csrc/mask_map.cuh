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

__global__ void dilation(
    const int kernel_size,
    const dim3 mask_size,
    const bool* __restrict__ kernel,
    const bool* __restrict__ mask,
    bool* __restrict__ out_mask
);

__global__ void erosion(
    const int kernel_size,
    const dim3 mask_size,
    const bool* __restrict__ kernel,
    const bool* __restrict__ mask,
    bool* __restrict__ out_mask
);

__global__ void filter(
    const int kernel_size,
    const dim3 mask_size,
    const float* __restrict__ kernel,
    const bool* __restrict__ mask,
    float* __restrict__ out_mask
);