from typing import Any, Optional
import torch
import time
from torch import Tensor
from .project_gaussians import project_gaussians, _ProjectGaussians
from .rasterize import rasterize_gaussians, _RasterizeGaussians
from .utils import (
    map_gaussian_to_intersects,
    bin_and_sort_gaussians,
    compute_cumulative_intersects,
    compute_cov2d_bounds,
    get_tile_bin_edges,
)
from utils.sh_utils import spherical_harmonics
from isosplat.camera import Camera
# from isosplat.gaussian_model import GaussianModel
from scene.gaussian_model import GaussianModel
import warnings


__all__ = [
    "__version__",
    "project_gaussians",
    "rasterize_gaussians",
    "spherical_harmonics",
    # utils
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "compute_cov2d_bounds",
    "get_tile_bin_edges",
    "map_gaussian_to_intersects",
    # Function.apply() will be deprecated
    "ProjectGaussians",
    "RasterizeGaussians",
    "BinAndSortGaussians",
    "ComputeCumulativeIntersects",
    "ComputeCov2dBounds",
    "GetTileBinEdges",
    "MapGaussiansToIntersects",
    "SphericalHarmonics",
    "NDRasterizeGaussians",
]

# Define these for backwards compatibility


class MapGaussiansToIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "MapGaussiansToIntersects is deprecated, use map_gaussian_to_intersects instead",
            DeprecationWarning,
        )
        return map_gaussian_to_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCumulativeIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCumulativeIntersects is deprecated, use compute_cumulative_intersects instead",
            DeprecationWarning,
        )
        return compute_cumulative_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCov2dBounds(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCov2dBounds is deprecated, use compute_cov2d_bounds instead",
            DeprecationWarning,
        )
        return compute_cov2d_bounds(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class GetTileBinEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "GetTileBinEdges is deprecated, use get_tile_bin_edges instead",
            DeprecationWarning,
        )
        return get_tile_bin_edges(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class BinAndSortGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "BinAndSortGaussians is deprecated, use bin_and_sort_gaussians instead",
            DeprecationWarning,
        )
        return bin_and_sort_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ProjectGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussians is deprecated, use project_gaussians instead",
            DeprecationWarning,
        )
        return project_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "RasterizeGaussians is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class NDRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "NDRasterizeGaussians is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "SphericalHarmonics is deprecated, use spherical_harmonics instead",
            DeprecationWarning,
        )
        return spherical_harmonics(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
    

BLOCK_WIDTH = 16


def render(
        camera: Camera,
        gm: GaussianModel,
        global_size: float = 1.0,
        background_depth: float = 20.0,
        color: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor, float, float]:
    view_mat = camera.get_view_matrix()
    focalx, focaly = camera.get_focal()
    width, height = camera.get_size()
    cx, cy = camera.get_principal()

    if color is None:
        color = torch.zeros(3, device=gm.get_xyz.device)

    start = time.time()

    xys, depths, radii, conics, compensation, num_tiles_hit, conv3d = _ProjectGaussians.apply(
        gm.get_xyz,
        gm.get_scaling,
        global_size,
        gm.get_rotation,
        view_mat,
        focalx,
        focaly,
        cx,
        cy,
        height,
        width,
        BLOCK_WIDTH
    )

    torch.cuda.synchronize()
    t0 = time.time() - start
    start = time.time()

    out_img, out_alpha, out_depth = _RasterizeGaussians.apply(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        gm.get_colors(camera.get_camera_position()),
        gm.get_opacity,
        height,
        width,
        BLOCK_WIDTH,
        color,
        background_depth,
        True
    )

    torch.cuda.synchronize()
    t1 = time.time() - start
    return out_img, out_alpha, out_depth, t0, t1