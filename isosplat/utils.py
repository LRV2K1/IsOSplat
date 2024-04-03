from typing import Tuple, NewType
import numpy as np
from jaxtyping import Float, Int

import torch
from torch import Tensor

import isosplat.cuda as _C
from isosplat.camera import Camera


PointCloud = NewType("PointCloud", tuple[np.ndarray, np.ndarray, np.ndarray])
Data = NewType("Data", dict[str, tuple[Tensor, Camera, dict[str, any]]])
DataList = NewType("DataList", list[str])


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        block_size,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int,
    isect_ids_sorted: Int[Tensor, "num_intersects 1"],
    tile_bounds: Tuple[int, int, int],
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(
        num_intersects, isect_ids_sorted.contiguous(), tile_bounds
    )


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
        block_size,
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper