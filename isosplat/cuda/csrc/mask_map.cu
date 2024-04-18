#include "mask_map.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void extract_segment_features(
    const int num_points,
    const dim3 mask_size,
    const bool* __restrict__ mask,
    const int* __restrict__ xys,
    bool* __restrict__ xy_mask
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
    {
        return;
    }

    int x = xys[idx * 2];
    int y = xys[(idx * 2) + 1];

    if (x >= 0 && x < mask_size.x)
    {
        if (y >= 0 && y < mask_size.y)
        {
            int32_t pix_id = x + y * mask_size.x;
            if (mask[pix_id])
            {
                xy_mask[idx] = true;
            }
        }
    }
}

__global__ void padd_features(
    const int num_points,
    const dim3 kernel_size,
    const dim3 mask_size,
    const int* __restrict__ xys,
    const float* __restrict__ kernel,
    float* __restrict__ mask
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
    {
        return;
    }

    int x = xys[idx * 2];
    int y = xys[(idx * 2) + 1];   
    
    if (x >= 0 && x < mask_size.x)
    {
        if (y >= 0 && y < mask_size.y)
        {
            int32_t pix_id = x + y * mask_size.x;
            mask[pix_id] = 1.f;
        }
    }
}