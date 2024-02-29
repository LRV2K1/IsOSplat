import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .camera import Camera
from .optimizer import Optimizer
from .depth.sfm_types import PointCloud
import isosplat.loss_functions as loss_functions
import isosplat.utils as utils

import torch
from torch import Tensor, optim
from .project_gaussians import _ProjectGaussians
from .rasterize import _RasterizeGaussians
from gsplat import spherical_harmonics
from PIL import Image

BLOCK_WIDTH = 16


class GaussianSplatting:
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.background: Tensor = torch.zeros(3, device=self.device)
        self.frames: list = []

        self.num_points: int = 0
        self.means: Tensor
        self.scales: Tensor
        self.opacities: Tensor
        self.sh_coeffs: Tensor
        self.quats: Tensor
        self.sh_degree: int = 0

        self.optimizer: Optimizer

    def init_gaussians(self, splats: int, load_path: Optional[Path] = None, point_cloud: Optional[PointCloud] = None):
        if load_path:
            self.means = torch.load(load_path / "means.pt")
            self.scales = torch.load(load_path / "scales.pt")
            self.opacities = torch.load(load_path / "opacities.pt")
            self.sh_coeffs = torch.load(load_path / "sh.pt")
            self.quats = torch.load(load_path / "quats.pt")
            self.sh_degree = 4
            self.num_points = self.opacities.shape[0]
        elif point_cloud:
            print("sfm")
            # TODO sfm
        else:
            self.num_points = splats

            self.means = 2 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
            self.scales = torch.rand(self.num_points, 3, device=self.device)
            self.opacities = torch.ones((self.num_points, 1), device=self.device)
            self.sh_coeffs = torch.rand(self.num_points, 25, 3, device=self.device)

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

    def init_optimizer(self, lr: float):
        optimize_tensors = {
            'sh_coeffs': self.sh_coeffs,
            'means': self.means,
            'scales': self.scales,
            'opacities': self.opacities,
            'quats': self.quats
        }

        self.optimizer = Optimizer()
        self._update_tensors(self.optimizer.load_tensor_dict(optimize_tensors, lr))

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
        return itr % 100 == 0 and itr > 0

    def _is_reset_iteration(self, itr: int, iterations: int) -> bool:
        return (itr + 1) % 3000 == 0 and iterations - itr > 1000

    def _add_sh_band(self, itr: int) -> bool:
        return self.sh_degree < 4 and itr % 1000 == 0 and itr > 0

    def train(self, data: list[tuple[Tensor, Tensor, Camera, str]], iterations: int = 200):
        self.frames = []
        times = [0] * 3

        n_data = len(data)

        for itr in range(iterations):
            data_itr = 0
            if self._is_reset_iteration(itr, iterations):
                self._reset_opacity()
            if self._add_sh_band(itr):
                self.sh_degree += 1

            for gt_view, gt_alpha, camera, _ in data:
                gt_view = gt_view.to(device=self.device)
                gt_alpha = gt_alpha.to(device=self.device)
                nv_view, nv_alpha, t0, t1 = self.rasterize(camera)
                loss = self.loss(gt_view, nv_view, gt_alpha, nv_alpha)
                t2 = self.optimizer.back_propagate_loss(loss)

                times[0] += t0
                times[1] += t1
                times[2] += t2

                if self._is_refinement_iteration(itr):
                    self._densify_and_prune(0.0002, 0.005, 2, self.optimizer.get_learning_rate())

                print(f"Iteration {itr + 1}/{iterations}, Data: {data_itr + 1}/{n_data}, Loss: {loss.item()}")

                if data_itr == 0 and itr % 5 == 0:
                    self.frames.append((nv_view.detach().cpu().numpy() * 255).astype(np.uint8))
                data_itr += 1

        print(f"Number gaussians: {self.num_points}")
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0] / (iterations * len(data)):.5f}, Rasterize: {times[1] / (iterations * len(data)):.5f}, Backward: {times[2] / (iterations * len(data)):.5f}"
        )

    def save(self, save_path: Path):
        if len(self.frames) <= 0:
            return

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.sh_coeffs, f"{save_path}/sh.pt")  # sh_coeffs
        torch.save(self.means, f"{save_path}/means.pt")  # means
        torch.save(self.scales, f"{save_path}/scales.pt")  # scales
        torch.save(self.opacities, f"{save_path}/opacities.pt")  # opacities
        torch.save(self.quats, f"{save_path}/quats.pt")  # quats

        self.frames = [Image.fromarray(frame) for frame in self.frames]
        self.frames[0].save(
            f"{save_path}/training.gif",
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )
        self.frames[-1].save(
            f"{save_path}/training.png",
            save_all=True,
            optimize=False,
        )

    def _calculate_sh_color(self, degrees_to_use: int, camera: Camera, sh_coeffs: Tensor) -> Tensor:
        view_dirs = camera.get_camera_position().repeat(self.num_points, 1) - self.means
        return spherical_harmonics(degrees_to_use, view_dirs, sh_coeffs)

    def rasterize(self, camera: Camera, color: Optional[Tensor] = None) -> tuple[Tensor, Tensor, float, float]:
        view_mat, project_mat = camera.get_view_and_project_matrix()
        focalx, focaly = camera.get_focal()
        width, height = camera.get_size()

        start = time.time()  # get iteration start time

        if color == None:
            color = self.background

        xys, depths, radii, conics, compensation, num_tiles_hit, conv3d = _ProjectGaussians.apply(
            self.means,
            self.scales,
            1,
            self.quats,
            view_mat,
            project_mat,
            focalx,
            focaly,
            width / 2,
            height / 2,
            height,
            width,
            BLOCK_WIDTH
        )

        torch.cuda.synchronize()
        t0 = time.time() - start
        start = time.time()

        out_img, out_alpha = _RasterizeGaussians.apply(
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
            True
        )

        torch.cuda.synchronize()
        t1 = time.time() - start
        return out_img, out_alpha, t0, t1

    def render(self, camera: Camera, color: Optional[Tensor] = None) -> tuple[Tensor, float, float]:
        with torch.no_grad():
            out_img, _, t0, t1 = self.rasterize(camera, color)
            return out_img, t0, t1

    def _densify_and_prune(self, grad_threshold: float, opacity_threshold: float, size_threshold: float, extend: float):
        self._clone(grad_threshold, size_threshold, extend)
        self._split(grad_threshold, size_threshold, extend)

        opacity_threshold = utils.inverse_sigmoid(opacity_threshold)
        mask = (self.opacities <= opacity_threshold).squeeze()
        self._update_tensors(self.optimizer.prune_optimizer(~mask))

    def _split(self, grad_threshold: float, size_threshold: float, extend: float):
        view_space_gradients = _RasterizeGaussians.getViewSpaceGradient()
        padded_view_space_gradients = torch.zeros(self.num_points - view_space_gradients.shape[0],
                                                  view_space_gradients.shape[1], device=self.device)
        padded_view_space_gradients = torch.cat((view_space_gradients, padded_view_space_gradients))
        mask = torch.where(torch.norm(padded_view_space_gradients, dim=-1) > grad_threshold, True, False)
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

    def _clone(self, grad_threshold: float, size_threshold: float, extend: float):
        view_space_gradients = _RasterizeGaussians.getViewSpaceGradient()
        mask = torch.where(torch.norm(view_space_gradients, dim=-1) > grad_threshold, True, False)
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

    def _reset_opacity(self):
        new_opacities = torch.min(self.opacities, utils.inverse_sigmoid_tensor(torch.ones_like(self.opacities) * 0.005))
        self._update_tensors(self.optimizer.replace_optimizer_tensor(new_opacities, "opacities"))

    def loss(self, gt_view: Tensor, nv_view: Tensor, gt_alpha: Tensor, nv_alpha: Tensor) -> Tensor:
        return 0.8 * loss_functions.l1_loss(nv_view, gt_view) + (1.0 - loss_functions.ssim(nv_view, gt_view))

    def verify(self, data: list[tuple[Tensor, Tensor, Camera, str]], save_path: Optional[Path] = None):

        with torch.no_grad():
            for gt_view, gt_alpha, camera, name in data:
                gt_view = gt_view.to(device=self.device)
                gt_alpha = gt_alpha.to(device=self.device)
                nv_view, nv_alpha, _, _ = self.rasterize(camera)
                loss = self.loss(gt_view, nv_view, gt_alpha, nv_alpha)

                print(f"Image: {name}, Loss:{loss.item()}")
                if save_path:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    image = Image.fromarray((nv_view.detach().cpu().numpy() * 255).astype(np.uint8))
                    image.save(f"{save_path}/{name}_render.png")

    def orbit_render(self, camera: Camera, save_path: Path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frames = []
        anglh = 0.0
        anglv = 0.0
        dis = 0.0

        for itr in range(200):
            # set camera
            if itr < 100:
                anglh = (math.pi / 50) * itr
            else:
                anglh = (math.pi / 50) * (99 - (itr % 100))

            # if (itr < 50):
            #     anglv = (math.pi / 200) * itr
            # elif (itr < 150):
            #     anglv = (math.pi / 200) * (49 - (itr - 50))
            # else:
            #     anglv = (math.pi / 200) * (-50 + (itr % 50))

            if (itr % 100) < 50:
                dis += 1
            else:
                dis -= 1

            camera.orbit(0.0, 0.0, 0.0, 8.0 + (dis / 25.0), anglh, anglv)

            out_img, _, _ = self.render(camera)

            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(
            f"{save_path}/render.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )

    def zoom_render(self, camera: Camera, save_path: Path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frames = []
        dis = 0

        for itr in range(200):
            # set camera
            if itr < 100:
                dis += 1
            else:
                dis -= 1

            camera.distance(0.0, 0.0, 0.0, 7.0 + dis)

            out_img, _, _ = self.render(camera)

            frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(
            f"{save_path}/render.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )