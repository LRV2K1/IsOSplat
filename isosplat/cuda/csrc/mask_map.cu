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
    const int kernel_size,
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

    int cx = xys[idx * 2];
    int cy = xys[(idx * 2) + 1];   

    int offset = kernel_size / 2;
    
    for (int kx = 0; kx < kernel_size; kx++)
    {
        for (int ky = 0; ky < kernel_size; ky++)
        {
            int x = cx + (kx - offset);
            int y = cy + (ky - offset);
            if (x >= 0 && x < mask_size.x)
            {
                if (y >= 0 && y < mask_size.y)
                {
                    int32_t pix_id = x + y * mask_size.x;
                    int k_id = kx + ky * kernel_size;
                    atomicAdd(&(mask[pix_id]), kernel[k_id]);
                    mask[pix_id] = min(mask[pix_id], 1.f);
                }
            }
        }
    }
}