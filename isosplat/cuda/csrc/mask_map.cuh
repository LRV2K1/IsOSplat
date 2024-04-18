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