from typing import Optional
import numpy as np

import torch
from torch import Tensor

from .optimizer import Optimizer
from arguments import OptimizationParams


# from isosplat.utils import PointCloud

from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import spherical_harmonics
import isosplat.cuda as _C


class GaussianModel:
    def __init__(self, device: torch.device, exp_scales: bool = True):
        self.device = device

        self.num_points: int = 0
        self._means: Tensor = torch.empty(0)
        self._scales: Tensor = torch.empty(0)
        self._opacities: Tensor = torch.empty(0)
        self._sh_base_coeffs: Tensor = torch.empty(0)
        self._sh_rest_coeffs: Tensor = torch.empty(0)
        self._quats: Tensor = torch.empty(0)
        self._acc_grad: Tensor = torch.empty(0)
        self._denom: Tensor = torch.empty(0)
        self._max_radii2D: Tensor = torch.empty(0)
        self.mean_lr: float = 0.001

        self.optimzable_params: Optional[OptimizationParams] = None
        self.optimizer: Optional[Optimizer] = None

        if exp_scales:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        else:
            self.scaling_activation = lambda x: x
            self.scaling_inverse_activation = lambda x: x
        self.rotation_activation = torch.nn.functional.normalize
        self.sigmoid_activation = torch.sigmoid
        self.inverse_sigmoid_activation = inverse_sigmoid

    def capture(self) -> tuple[dict[str, Tensor], dict[str, int | float]]:
        tensors = {
            "means": self._means,
            "scales": self._scales,
            "opacities": self._opacities,
            "sh_coeffs": self.get_sh_coeffs,
            "quats": self._quats,
            "acc_grad": self._acc_grad,
            "denom": self._denom,
            "max_radii2D": self._max_radii2D
        }
        values = {
            "num_points": self.num_points,
            "mean_lr": self.mean_lr
        }

        return tensors, values

    def restore(self, model: tuple[dict[str, Tensor], dict[str, int]]):
        tensors, values = model
        self._means = tensors["means"]
        self._scales = tensors["scales"]
        self._opacities = tensors["opacities"]
        self._sh_base_coeffs, self._sh_rest_coeffs = self._split_sh_coeffs(tensors["sh_coeffs"])
        self._quats = tensors["quats"]
        self._acc_grad = tensors["acc_grad"]
        self._denom = tensors["denom"]
        self._max_radii2D = tensors["max_radii2D"]
        self.num_points = values["num_points"]
        self.mean_lr = values["mean_lr"]

    def create_from_pcd(self, pcd: BasicPointCloud, mean_lr: float) -> int:
        xyzs, rgbs, errors = pcd.points, pcd.colors, pcd.errors
        self.num_points = xyzs.shape[0]

        self._means = torch.tensor(np.float32(xyzs), device=self.device)
        dist2 = torch.clamp_min(_C.distCUDA2(torch.from_numpy(np.asarray(xyzs)).float().cuda()), 0.0000001)
        self._scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)    # TODO scales
        print(self._scales.shape)
        self._opacities = self.inverse_sigmoid_activation(torch.ones(self.num_points, 1, device=self.device) * 0.1)

        colors = inverse_sigmoid(torch.tensor(np.float32(rgbs / 256), device=self.device))
        self._sh_base_coeffs = torch.zeros(self.num_points, 1, 3, device=self.device)
        self._sh_base_coeffs[:, 0, :] = colors
        self._sh_rest_coeffs = torch.zeros(self.num_points, 24, 3, device=self.device)

        self._quats = torch.zeros(self.num_points, 4, device=self.device)
        self._quats[:, 0] = 1

        self._acc_grad = torch.zeros(self.num_points, 1, device=self.device)
        self._denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)
        self._max_radii2D = torch.zeros(self.num_points, device=self.device)

        self.mean_lr = mean_lr
        return self.num_points

    def create_from_random(self, splats: int, mean_lr: float) -> int:
        self.num_points = splats

        self._means = 10 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self._scales = self.scaling_inverse_activation(torch.ones(self.num_points, 3, device=self.device) * 0.1)
        self._opacities = self.inverse_sigmoid_activation(torch.ones(self.num_points, 1, device=self.device) * 0.1)

        self._sh_base_coeffs = self.inverse_sigmoid_activation(torch.ones(self.num_points, 1, 3, device=self.device) * 0.5)
        self._sh_rest_coeffs = torch.zeros(self.num_points, 24, 3, device=self.device)

        self._quats = torch.zeros(self.num_points, 4, device=self.device)
        self._quats[:, 0] = 1

        self._acc_grad = torch.zeros(self.num_points, 1, device=self.device)
        self._denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)
        self._max_radii2D = torch.zeros(self.num_points, device=self.device)

        self.mean_lr = mean_lr
        return self.num_points

    def add_gaussians(self, model: tuple[dict[str, Tensor], dict[str, int]]) -> int:
        tensors, values = model
        self._means = torch.cat((self._means, tensors["means"]), dim=0)
        self._scales = torch.cat((self._scales, tensors["scales"]), dim=0)
        self._opacities = torch.cat((self._opacities, tensors["opacities"]), dim=0)
        add_sh_base_coeffs, add_sh_rest_coeffs = self._split_sh_coeffs(tensors["sh_coeffs"])
        self._sh_base_coeffs = torch.cat((self._sh_base_coeffs, add_sh_base_coeffs), dim=0)
        self._sh_rest_coeffs = torch.cat((self._sh_rest_coeffs, add_sh_rest_coeffs), dim=0)
        self._quats = torch.cat((self._quats, tensors["quats"]), dim=0)
        self._acc_grad = torch.cat((self._acc_grad, tensors["acc_grad"]), dim=0)
        self._denom = torch.cat((self._denom, tensors["denom"]), dim=0)
        self._max_radii2D = torch.cat((self._max_radii2D, tensors["max_radii2D"]), dim=0)
        self.num_points += values["num_points"]

    def init_optimizer(self, optimizable_params: OptimizationParams) -> Optimizer:
        self.optimzable_params = optimizable_params

        optimize_tensors = {
            'sh_base_coeffs': (self._sh_base_coeffs, self.optimzable_params.sh_lr),
            'sh_rest_coeffs': (self._sh_rest_coeffs, self.optimzable_params.sh_lr / 20.0),
            'means': (self._means, self.optimzable_params.position_lr_init),
            'scales': (self._scales, self.optimzable_params.scaling_lr),
            'opacities': (self._opacities, self.optimzable_params.opacity_lr),
            'quats': (self._quats, self.optimzable_params.rotation_lr)
        }

        self.optimizer = Optimizer()
        self._update_tensors(self.optimizer.load_tensor_dict(optimize_tensors))
        self.optimizer.set_learning_rate_scheduler(
            self.optimzable_params.position_lr_init,
            self.optimzable_params.position_lr_final,
            self.optimzable_params.position_lr_delay_mult,
            self.optimzable_params.position_lr_max_steps)

        return self.optimizer

    @staticmethod
    def _split_sh_coeffs(sh_coeffs: Tensor) -> tuple[Tensor, Tensor]:
        return sh_coeffs[:, 0:1, :], sh_coeffs[:, 1:, :]

    @property
    def get_means(self) -> Tensor:
        return self._means

    @property
    def get_scales(self) -> Tensor:
        return self.scaling_activation(self._scales)

    @property
    def get_opacities(self) -> Tensor:
        return self.sigmoid_activation(self._opacities)

    @property
    def get_sh_coeffs(self) -> Tensor:
        return torch.cat((self._sh_base_coeffs, self._sh_rest_coeffs), 1)

    def get_colors(self, degrees_to_use: int, camera_position: Tensor) -> Tensor:
        view_dirs = camera_position.repeat(self.num_points, 1) - self.get_means
        return self.sigmoid_activation(spherical_harmonics(degrees_to_use, view_dirs, self.get_sh_coeffs))

    @property
    def get_quats(self) -> Tensor:
        return self.rotation_activation(self._quats)

    def _update_tensors(self, new_tensors: dict[str, Tensor]):
        if "sh_base_coeffs" in new_tensors:
            self._sh_base_coeffs = new_tensors["sh_base_coeffs"]
        if "sh_rest_coeffs" in new_tensors:
            self._sh_rest_coeffs = new_tensors["sh_rest_coeffs"]
        if "means" in new_tensors:
            self._means = new_tensors["means"]
        if "scales" in new_tensors:
            self._scales = new_tensors["scales"]
        if "opacities" in new_tensors:
            self._opacities = new_tensors["opacities"]
        if "quats" in new_tensors:
            self._quats = new_tensors["quats"]
        self.num_points = self._opacities.shape[0]

    def densify_and_prune(
            self,
            position_grads: Tensor,
            grad_threshold: float,
            opacity_threshold: float,
            size_threshold: float,
            extent: float,
            max_screen_size: Optional[float]
    ) -> tuple[int, int, int]:
        grads = self._acc_grad / self._denom
        grads[grads.isnan()] = 0.0

        clones = self._clone(grads, position_grads, grad_threshold, size_threshold)
        splits = self._split(grads, position_grads, grad_threshold, size_threshold)

        mask = (self.get_opacities < opacity_threshold).squeeze()
        if max_screen_size:
            big_points_vs = self._max_radii2D > max_screen_size
            big_points_ws = self.get_scales.max(dim=1).values > 0.1 * extent
            mask = torch.logical_or(torch.logical_or(mask, big_points_ws), big_points_vs)
        cur_points = self.num_points
        self._update_tensors(self.optimizer.prune_optimizer(~mask))
        culls = cur_points - self.num_points

        self._acc_grad = torch.zeros(self.num_points, 1, device=self.device)
        self._denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)
        self._max_radii2D = torch.zeros(self.num_points, device=self.device)

        torch.cuda.empty_cache()
        return culls, clones, splits

    def _split(self, grads: Tensor, position_grads: Tensor, grad_threshold: float, size_threshold: float) -> int:
        padded_grads = torch.zeros(self.num_points - grads.shape[0], grads.shape[1], device=self.device)
        padded_grads = torch.cat((grads, padded_grads))
        mask = torch.where(torch.norm(padded_grads, dim=-1) >= grad_threshold, True, False)
        mask = torch.logical_and(mask, torch.max(self.get_scales, dim=1).values > size_threshold)

        positional_gradient = position_grads
        padded_positional_gradient = torch.zeros(self.num_points - positional_gradient.shape[0],
                                                 positional_gradient.shape[1], device=self.device)
        padded_positional_gradient = torch.cat((positional_gradient, padded_positional_gradient))
        mask_positional_gradient = padded_positional_gradient[mask]
        padded_grad = torch.cat((torch.zeros_like(mask_positional_gradient, device=self.device),
                                 mask_positional_gradient))

        new_means = ((self._means[mask]).repeat(2, 1)) + (padded_grad * self.mean_lr)
        new_scales = ((self._scales[mask]).repeat(2, 1)) / 1.6
        new_quats = (self._quats[mask]).repeat(2, 1)
        new_sh_base_coeffs = (self._sh_base_coeffs[mask]).repeat(2, 1, 1)
        new_sh_rest_coeffs = (self._sh_rest_coeffs[mask]).repeat(2, 1, 1)
        new_opacities = (self._opacities[mask]).repeat(2, 1)

        new_max_radii = torch.cat(
            (self._max_radii2D[mask], torch.zeros(self._max_radii2D[mask].shape, device=self.device)))
        self._max_radii2D = torch.cat((self._max_radii2D[~mask], new_max_radii))

        optimize_tensors = {
            'sh_base_coeffs': new_sh_base_coeffs,
            'sh_rest_coeffs': new_sh_rest_coeffs,
            'means': new_means,
            'scales': new_scales,
            'opacities': new_opacities,
            'quats': new_quats
        }

        self.optimizer.prune_optimizer(~mask)
        self._update_tensors(self.optimizer.cat_optimizer_tensors(optimize_tensors))
        splits = int(new_opacities.shape[0] / 2)
        return splits

    def _clone(self, grads: Tensor, position_grads: Tensor, grad_threshold: float, size_threshold: float) -> int:
        mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        mask = torch.logical_and(mask, torch.max(self.get_scales, dim=1).values <= size_threshold)

        positional_gradient = position_grads[mask]
        new_means = self._means[mask] + (positional_gradient * self.mean_lr)
        new_scales = self._scales[mask]
        new_quats = self._quats[mask]
        new_sh_base_coeffs = self._sh_base_coeffs[mask]
        new_sh_rest_coeffs = self._sh_rest_coeffs[mask]
        new_opacities = self._opacities[mask]
        self._max_radii2D = torch.cat((self._max_radii2D, torch.zeros(self._max_radii2D[mask].shape, device=self.device)))

        optimize_tensors = {
            'sh_base_coeffs': new_sh_base_coeffs,
            'sh_rest_coeffs': new_sh_rest_coeffs,
            'means': new_means,
            'scales': new_scales,
            'opacities': new_opacities,
            'quats': new_quats
        }
        self._update_tensors(self.optimizer.cat_optimizer_tensors(optimize_tensors))
        clones = new_opacities.shape[0]
        return clones

    def add_densification_states(self, view_space_grads: Tensor, view_depth_grads: Tensor, radii: Tensor):
        view_space_gradients = view_space_grads
        view_depth_gradients = view_depth_grads[:, None]
        view_gradients = torch.cat((view_space_gradients, view_depth_gradients), 1)

        mask = torch.where(torch.norm(view_gradients, dim=-1) > 0, True, False)
        self._acc_grad[mask] += torch.norm(view_gradients[mask, :3], dim=-1, keepdim=True)
        self._denom[mask] += 1
        self._max_radii2D[mask] = torch.max(self._max_radii2D[mask], radii[mask])

    def reset_opacity(self):
        new_opacities = self.inverse_sigmoid_activation(torch.min(self.get_opacities, torch.ones_like(self._opacities) * 0.01))
        self._update_tensors(self.optimizer.replace_optimizer_tensor(new_opacities, "opacities"))
