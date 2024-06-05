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

__global__ void dilation(
    const int kernel_size,
    const dim3 mask_size,
    const bool* __restrict__ kernel,
    const bool* __restrict__ mask,
    bool* __restrict__ out_mask
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= mask_size.x * mask_size.y)
    {
        return;
    }

    int offset = kernel_size / 2;

    int32_t pix_x = idx % mask_size.x;
    int32_t pix_y = idx / mask_size.x;

    int kernel_inv_x = kernel_size - 1;
    int kernel_inv_y = kernel_size - 1;

    bool visible = false;
    for (int kx = 0; kx < kernel_size; kx++)
    {
        for (int ky = 0; ky < kernel_size; ky++)
        {
            int x = pix_x + (kx - offset);
            int y = pix_y + (ky - offset);
            if (x < 0 || x >= mask_size.x || y < 0 || y >= mask_size.y)
                continue;
                
            int32_t pix_id = x + y * mask_size.x;
            int k_id = (kernel_inv_x - kx) + (kernel_inv_y - ky) * kernel_size;
            //already visible (or) (pixel in image and in kernel)
            visible = visible || (mask[pix_id] && kernel[k_id]);
        }
    }
    out_mask[idx] = visible;
}

__global__ void erosion(
    const int kernel_size,
    const dim3 mask_size,
    const bool* __restrict__ kernel,
    const bool* __restrict__ mask,
    bool* __restrict__ out_mask
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= mask_size.x * mask_size.y)
    {
        return;
    }

    int offset = kernel_size / 2;

    int32_t pix_x = idx % mask_size.x;
    int32_t pix_y = idx / mask_size.x;

    int kernel_inv_x = kernel_size - 1;
    int kernel_inv_y = kernel_size - 1;

    bool visible = true;
    for (int kx = 0; kx < kernel_size; kx++)
    {
        for (int ky = 0; ky < kernel_size; ky++)
        {
            int x = pix_x + (kx - offset);
            int y = pix_y + (ky - offset);

            int32_t pix_id = x + y * mask_size.x;
            if (x < 0 || x >= mask_size.x || y < 0 || y >= mask_size.y)
                continue;
                
            int k_id = (kernel_inv_x - kx) + (kernel_inv_y - ky) * kernel_size;
            //all previous visible (and) (current pixel in image and kernel (or) pixel not in kernel)
            visible = visible && ((mask[pix_id] && kernel[k_id]) || !kernel[k_id]);
        }
    }
    out_mask[idx] = visible;
}

__global__ void filter(
    const int kernel_size,
    const dim3 mask_size,
    const float* __restrict__ kernel,
    const bool* __restrict__ mask,
    float* __restrict__ out_mask
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= mask_size.x * mask_size.y)
    {
        return;
    }

    int offset = kernel_size / 2;

    int32_t pix_x = idx % mask_size.x;
    int32_t pix_y = idx / mask_size.x;

    int kernel_inv_x = kernel_size - 1;
    int kernel_inv_y = kernel_size - 1;

    float value = 0.0;
    for (int kx = 0; kx < kernel_size; kx++)
    {
        for (int ky = 0; ky < kernel_size; ky++)
        {
            int x = pix_x + (kx - offset);
            int y = pix_y + (ky - offset);

            int32_t pix_id = x + y * mask_size.x;
            if (x < 0 || x >= mask_size.x || y < 0 || y >= mask_size.y)
                continue;
            if (!mask[pix_id])
                continue;
                
            int k_id = (kernel_inv_x - kx) + (kernel_inv_y - ky) * kernel_size;
            value += kernel[k_id];
        }
    }
    out_mask[idx] = min(value, 1.0);
}