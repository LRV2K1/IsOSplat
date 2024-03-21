import math
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import random

import torch
from torch import Tensor
from torchrl.record import CSVLogger

from .camera import Camera
from .optimizer import Optimizer
from .optimization_params import OptimizationParams
import isosplat.loss_functions as loss_functions
import isosplat.utils as utils
from isosplat.utils import PointCloud, DataList, Data
from .project_gaussians import _ProjectGaussians
from .rasterize import _RasterizeGaussians
from isosplat import spherical_harmonics


BLOCK_WIDTH = 16


class GaussianSplatting:
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.background: Tensor = torch.zeros(3, device=self.device)
        self.frames: list = []

        self.num_points: int = 0
        self.means: Optional[Tensor] = None
        self.scales: Optional[Tensor] = None
        self.opacities: Optional[Tensor] = None
        self.sh_coeffs: Optional[Tensor] = None
        self.quats: Optional[Tensor] = None
        self.acc_grad: Optional[Tensor] = None
        self.denom: Optional[Tensor] = None
        self.sh_degree: int = 0

        self.optimzable_params: OptimizationParams = OptimizationParams()
        self.optimizer: Optional[Optimizer] = None

        self.splits = 0
        self.clones = 0
        self.culls = 0
        self.times = [0] * 3

    def init_axis(self, add_axis: bool = False):
        means = torch.tensor(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, -1.0],
             [0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0]],
            device=self.device
        )
        scales = torch.ones(self.num_points, 3, device=self.device) * 0.1
        opacities = torch.ones((self.num_points, 1), device=self.device) * 10.0
        sh_coeffs = torch.zeros(7, 25, 3, device=self.device)
        sh_coeffs[:, 0, :] = torch.tensor(
            [
                [-10.0, -10.0, -10.0],
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
                [-10.0, 10.0, -10.0],
                [10.0, 10.0, -10.0],
                [-10.0, 10.0, -10.0]
            ],
            device=self.device
        )
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)
        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w)
            ],
            -1
        )

        if add_axis:
            self.means = torch.cat((self.means, means), 0)
            self.scales = torch.cat((self.scales, scales), 0)
            self.opacities = torch.cat((self.opacities, opacities), 0)
            self.sh_coeffs = torch.cat((self.sh_coeffs, sh_coeffs), 0)
            self.quats = torch.cat((self.quats, quats), 0)
            self.num_points += 7
        else:
            self.means = means
            self.scales = scales
            self.opacities = opacities
            self.sh_coeffs = sh_coeffs
            self.quats = quats
            self.sh_degree = 0
            self.num_points = 7

    def init_gaussians(
            self,
            splats: int,
            load_path: Optional[Path] = None,
            point_cloud: PointCloud = None,
            logger: Optional[CSVLogger] = None
    ):
        if load_path:
            print("Loading existing gaussians")
            self.means = torch.load(load_path / "means.pt")
            self.scales = torch.load(load_path / "scales.pt")
            self.opacities = torch.load(load_path / "opacities.pt")
            self.sh_coeffs = torch.load(load_path / "sh.pt")
            self.quats = torch.load(load_path / "quats.pt")
            self.acc_grad = torch.load(load_path / "acc_grad.pt")
            self.denom = torch.load(load_path / "denom.pt")
            self.sh_degree = 4
            self.num_points = self.opacities.shape[0]
        elif point_cloud:
            print("Creating gaussians from SFM point cloud")
            xyzs, rgbs, errors = point_cloud

            self.means = torch.tensor(np.float32(xyzs), device=self.device)
            self.num_points = self.means.shape[0]
        
            # TODO scales
            self.scales = torch.ones(self.num_points, 3, device=self.device) * 0.01
            self.opacities = torch.ones((self.num_points, 1), device=self.device) * 10.0

            colors = utils.inverse_sigmoid_tensor(torch.tensor(np.float32(rgbs/256), device=self.device))
            self.sh_coeffs = torch.rand(self.num_points, 25, 3, device=self.device)
            self.sh_coeffs[:, 0, :] = colors
            self.acc_grad = torch.zeros(self.num_points, 1, device=self.device)
            self.denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)

            u = torch.rand(self.num_points, 1, device=self.device)
            v = torch.rand(self.num_points, 1, device=self.device)
            w = torch.rand(self.num_points, 1, device=self.device)
            self.quats = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w)
                ],
                -1
            )
            self.sh_degree = 0
        else:
            print("Randomly initialize gaussians")
            self.num_points = splats

            self.means = 2 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
            self.scales = torch.rand(self.num_points, 3, device=self.device)
            self.opacities = torch.ones((self.num_points, 1), device=self.device)
            self.sh_coeffs = torch.rand(self.num_points, 25, 3, device=self.device)
            self.acc_grad = torch.zeros(self.num_points, 1, device=self.device)
            self.denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)

            u = torch.rand(self.num_points, 1, device=self.device)
            v = torch.rand(self.num_points, 1, device=self.device)
            w = torch.rand(self.num_points, 1, device=self.device)
            self.quats = torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w)
                ],
                -1
            )
            self.sh_degree = 0
        print(f"Initialized {self.num_points} gaussians")
        if logger is not None:
            logger.log_scalar("n_gaussians", self.num_points)

    def init_optimizer(self, optimizable_params: OptimizationParams):
        self.optimzable_params = optimizable_params
        
        optimize_tensors = {
            'sh_coeffs': (self.sh_coeffs, self.optimzable_params.sh_lr),
            'means': (self.means, self.optimzable_params.position_lr_init),
            'scales': (self.scales, self.optimzable_params.scaling_lr),
            'opacities': (self.opacities, self.optimzable_params.opacity_lr),
            'quats': (self.quats, self.optimzable_params.rotation_lr)
        }

        self.optimizer = Optimizer()
        self._update_tensors(self.optimizer.load_tensor_dict(optimize_tensors))
        self.optimizer.set_learning_rate_scheduler(
            self.optimzable_params.position_lr_init, 
            self.optimzable_params.position_lr_final, 
            self.optimzable_params.position_lr_delay_mult, 
            self.optimzable_params.position_lr_max_steps)

    def _update_tensors(self, new_tensors: dict[str, Tensor]):
        if "sh_coeffs" in new_tensors:
            self.sh_coeffs = new_tensors["sh_coeffs"]
        if "means" in new_tensors:
            self.means = new_tensors["means"]
        if "scales" in new_tensors:
            self.scales = new_tensors["scales"]
        if "opacities" in new_tensors:
            self.opacities = new_tensors["opacities"]
        if "quats" in new_tensors:
            self.quats = new_tensors["quats"]
        self.num_points = self.opacities.shape[0]

    def _is_refinement_iteration(self, itr: int) -> bool:
        return itr % self.optimzable_params.densification_interval == 0 and \
                itr >= self.optimzable_params.densify_from_iter and \
                itr <= self.optimzable_params.densify_until_iter

    def _is_reset_iteration(self, itr: int) -> bool:
        return itr % self.optimzable_params.opacity_reset_interval == 0 and \
                self.optimzable_params.iterations - itr >= 1000 and itr > 0

    def _add_sh_band(self, itr: int) -> bool:
        return itr % 1000 == 0 and self.sh_degree < 4 and itr > 0

    def train(self, data_list: DataList, data: Data, logger: Optional[CSVLogger] = None):
        n_data = len(data_list)
        iterations = self.optimzable_params.iterations

        for itr in range(iterations):
            average_image_loss = 0
            average_loss = 0
            data_itr = 0
            lr = self.optimizer.update_learning_rate(itr)
            if self._is_reset_iteration(itr):
                self._reset_opacity()
            if self._add_sh_band(itr):
                self.sh_degree += 1

            random.shuffle(data_list)
            for name in data_list:
                gt_view, camera, add_data = data[name]
                bg_depth = 20.0
                if "bg_depth" in add_data:
                    bg_depth = add_data["bg_depth"]
                
                nv_view, nv_alpha, nv_depth, t0, t1 = self.rasterize(camera, background_depth=bg_depth)
                loss, image_loss = self.loss(gt_view, nv_view, nv_alpha, nv_depth, add_data)
                t2 = self.optimizer.back_propagate_loss(loss)
                average_image_loss += image_loss
                average_loss += loss.item()

                self.times[0] += t0
                self.times[1] += t1
                self.times[2] += t2

                self.add_densification_states()

                print(f"Iteration {itr + 1}/{iterations}, Data: {data_itr + 1}/{n_data}, Loss: {loss.item()}")
                data_itr += 1
            
            if self._is_refinement_iteration(itr):
                culls, clones, splits = self._densify_and_prune(self.optimzable_params.densify_grad_threshold, 0.005, 2, lr)
                if logger is not None:
                    logger.log_scalar("culls", culls, itr)
                    logger.log_scalar("splits", splits, itr)
                    logger.log_scalar("clones", clones, itr)
                    logger.log_scalar("n_gaussians", self.num_points, itr)
            average_image_loss /= n_data
            if logger is not None:
                logger.log_scalar("average_image_loss", average_image_loss, itr)
                logger.log_scalar("average_loss", average_loss, itr)

    def save(self, save_path: Path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.sh_coeffs, f"{save_path}/sh.pt")
        torch.save(self.means, f"{save_path}/means.pt")
        torch.save(self.scales, f"{save_path}/scales.pt")
        torch.save(self.opacities, f"{save_path}/opacities.pt")
        torch.save(self.quats, f"{save_path}/quats.pt")
        torch.save(self.acc_grad, f"{save_path}/acc_grad.pt")
        torch.save(self.denom, f"{save_path}/denom.pt")

    def _calculate_sh_color(self, degrees_to_use: int, camera: Camera, sh_coeffs: Tensor) -> Tensor:
        view_dirs = camera.get_camera_position().repeat(self.num_points, 1) - self.means
        return spherical_harmonics(degrees_to_use, view_dirs, sh_coeffs)

    def rasterize(
            self,
            camera: Camera,
            size: float = 1.0,
            background_depth: float = 20.0,
            color: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, float, float]:
        view_mat, project_mat = camera.get_view_and_project_matrix()
        focalx, focaly = camera.get_focal()
        width, height = camera.get_size()
        cx, cy = camera.get_principal()

        start = time.time()  # get iteration start time

        if color is None:
            color = self.background

        xys, depths, radii, conics, compensation, num_tiles_hit, conv3d = _ProjectGaussians.apply(
            self.means,
            self.scales,
            size,
            self.quats,
            view_mat,
            project_mat,
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
            torch.sigmoid(self._calculate_sh_color(self.sh_degree, camera, self.sh_coeffs)),
            torch.sigmoid(self.opacities),
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

    def render(
            self,
            camera: Camera,
            size: float = 1.0,
            background_depth: float = 20.0,
            color: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, float, float]:
        with torch.no_grad():
            out_img, _, out_depth, t0, t1 = self.rasterize(camera, size, background_depth, color)
            return out_img, out_depth, t0, t1

    def _densify_and_prune(
            self,
            grad_threshold: float,
            opacity_threshold: float,
            size_threshold: float,
            extend: float
    ) -> tuple[int, int, int]:
        grads = self.acc_grad / self.denom
        grads[grads.isnan()] = 0.0
        
        clones = self._clone(grads, grad_threshold, size_threshold, extend)
        splits = self._split(grads, grad_threshold, size_threshold, extend)

        opacity_threshold = utils.inverse_sigmoid(opacity_threshold)
        mask = (self.opacities <= opacity_threshold).squeeze()
        cur_points = self.num_points
        self._update_tensors(self.optimizer.prune_optimizer(~mask))
        culls = cur_points - self.num_points
        self.culls += culls

        self.acc_grad = torch.zeros(self.num_points, 1, device=self.device)
        self.denom = torch.zeros(self.num_points, 1, dtype=torch.int32, device=self.device)

        torch.cuda.empty_cache()
        return culls, clones, splits

    def _split(self, grads: Tensor, grad_threshold: float, size_threshold: float, extend: float) -> int:
        # view_space_gradients = _RasterizeGaussians.getViewSpaceGradient()
        padded_grads = torch.zeros(self.num_points - grads.shape[0], grads.shape[1], device=self.device)
        padded_grads = torch.cat((grads, padded_grads))
        mask = torch.where(torch.norm(padded_grads, dim=-1) >= grad_threshold, True, False)
        mask = torch.logical_and(mask, torch.max(self.scales, dim=1).values > size_threshold)

        positional_gradient = _ProjectGaussians.getPositionalGradient()
        padded_positional_gradient = torch.zeros(self.num_points - positional_gradient.shape[0],
                                                 positional_gradient.shape[1], device=self.device)
        padded_positional_gradient = torch.cat((positional_gradient, padded_positional_gradient))
        mask_positional_gradient = padded_positional_gradient[mask]
        padded_grad = torch.cat((torch.zeros_like(mask_positional_gradient, device=self.device),
                                 mask_positional_gradient))

        new_means = ((self.means[mask]).repeat(2, 1)) + (padded_grad * extend)
        new_scales = ((self.scales[mask]).repeat(2, 1)) / 1.6
        new_quats = (self.quats[mask]).repeat(2, 1)
        new_sh_coeffs = (self.sh_coeffs[mask]).repeat(2, 1, 1)
        new_opacities = (self.opacities[mask]).repeat(2, 1)

        optimize_tensors = {
            'sh_coeffs': new_sh_coeffs,
            'means': new_means,
            'scales': new_scales,
            'opacities': new_opacities,
            'quats': new_quats
        }

        self.optimizer.prune_optimizer(~mask)
        self._update_tensors(self.optimizer.cat_optimizer_tensors(optimize_tensors))
        splits = int(new_opacities.shape[0]/2)
        self.splits += splits
        return splits

    def _clone(self, grads: Tensor,  grad_threshold: float, size_threshold: float, extend: float) -> int:
        # view_space_gradients = _RasterizeGaussians.getViewSpaceGradient()
        # mask = torch.where(torch.norm(view_space_gradients, dim=-1) > grad_threshold, True, False)
        mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        mask = torch.logical_and(mask, torch.max(self.scales, dim=1).values <= size_threshold)

        positional_gradient = _ProjectGaussians.getPositionalGradient()[mask]
        new_means = self.means[mask] + (positional_gradient * extend)
        new_scales = self.scales[mask]
        new_quats = self.quats[mask]
        new_sh_coeffs = self.sh_coeffs[mask]
        new_opacities = self.opacities[mask]

        optimize_tensors = {
            'sh_coeffs': new_sh_coeffs,
            'means': new_means,
            'scales': new_scales,
            'opacities': new_opacities,
            'quats': new_quats
        }
        self._update_tensors(self.optimizer.cat_optimizer_tensors(optimize_tensors))
        clones = new_opacities.shape[0]
        self.clones += clones
        return clones

    def add_densification_states(self):
        view_space_gradients = _RasterizeGaussians.getViewSpaceGradient()
        view_depth_gradients = _RasterizeGaussians.getViewDepthGradient()[:, None]
        view_gradients = torch.cat((view_space_gradients, view_depth_gradients), 1)

        mask = torch.where(torch.norm(view_gradients, dim=-1) > 0, True, False)
        self.acc_grad[mask] += torch.norm(view_gradients[mask, :3], dim=-1, keepdim=True)
        self.denom[mask] += 1

    def _reset_opacity(self):
        new_opacities = torch.min(self.opacities, utils.inverse_sigmoid_tensor(torch.ones_like(self.opacities) * 0.005))
        self._update_tensors(self.optimizer.replace_optimizer_tensor(new_opacities, "opacities"))

    def loss(self, gt_view: Tensor, nv_view: Tensor, nv_alpha: Tensor = None, nv_depth: Tensor = None, add_data: dict = None) -> tuple[Tensor, float]:
        loss = (1.0 - self.optimzable_params.l_ssim) * loss_functions.l1_loss(nv_view, gt_view) \
            + self.optimzable_params.l_ssim * (1.0 - loss_functions.ssim(nv_view, gt_view))
        img_loss = loss.item()
        if "depth" in add_data and nv_depth is not None:
            loss += self.optimzable_params.l_depth * loss_functions.l1_loss(nv_depth, add_data["depth"])
        if "edges" in add_data and nv_depth is not None:
            loss += self.optimzable_params.l_smooth * loss_functions.l_smooth(nv_depth, add_data["edges"])
        return loss, img_loss

    def verify(
            self,
            data_list: DataList,
            data: Data,
            save_path: Optional[Path] = None,
            logger: Optional[CSVLogger] = None
    ) -> float:
        average_loss = 0.0
        print(f"Number gaussians: {self.num_points}")
        print(f"Culls: {self.culls}, Splits: {self.splits}, Clones: {self.clones}")
        print(
            f"Total(s):\nProject: {self.times[0]:.3f}, Rasterize: {self.times[1]:.3f}, Backward: {self.times[2]:.3f}"
        )
        iterations = self.optimzable_params.iterations
        if iterations > 0 and len(data) > 0:
            print(
                f"Per step(s):\n"
                f"Project: {self.times[0] / (iterations * len(data)):.5f}, "
                f"Rasterize: {self.times[1] / (iterations * len(data)):.5f}, "
                f"Backward: {self.times[2] / (iterations * len(data)):.5f}"
            )

        if logger is not None:
            final_parameters = {
                "n_gaussians": self.num_points,
                "culls": self.culls,
                "splits": self.splits,
                "clones": self.clones,
                "project": self.times[0],
                "rasterize": self.times[1],
                "backwards": self.times[2]
            }
            logger.log_hparams(final_parameters)

        with torch.no_grad():
            final_loss = {}
            for name in data_list:
                gt_view, camera, add_data = data[name]
                bg_depth = 20.0
                if "bg_depth" in add_data:
                    bg_depth = add_data["bg_depth"]

                nv_view, nv_alpha, nv_depth, _, _ = self.rasterize(camera, background_depth=bg_depth)
                loss, img_loss = self.loss(gt_view, nv_view, nv_alpha, nv_depth, add_data)

                average_loss += img_loss
                print(f"Image: {name}, Loss:{loss.item()}")
                print(f"Image: {name}, Img_Loss:{img_loss}")
                if logger is not None:
                    final_loss[f"loss: {name}"] = loss.item()
                    final_loss[f"image loss: {name}"] = img_loss
                if save_path:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    image = Image.fromarray((nv_view.detach().cpu().numpy() * 255).astype(np.uint8))
                    image.save(f"{save_path}/{name}_render.png")

                    norm_depth = nv_depth - nv_depth.min()
                    m = norm_depth.max()
                    if m > 0.0:
                        norm_depth /= m
                    else:
                        norm_depth = norm_depth + 1.0

                    depth_image = Image.fromarray((norm_depth.detach().cpu().numpy() * 255).astype(np.uint8))
                    depth_image.save(f"{save_path}/{name}_depth.png")
            if logger is not None:
                logger.log_hparams(final_loss)
        return average_loss / len(data_list)
