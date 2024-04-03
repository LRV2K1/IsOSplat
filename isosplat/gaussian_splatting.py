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
from arguments import GroupParams
from .gaussian_model import GaussianModel
from utils.loss_utils import l1_loss, l2_loss, ssim, nearMean_map

from isosplat.utils import PointCloud, DataList, Data
from isosplat import render

from .project_gaussians import _ProjectGaussians
from .rasterize import _RasterizeGaussians


class GaussianSplatting:
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.background: Tensor = torch.zeros(3, device=self.device)

        self.gaussian_model: GaussianModel = GaussianModel(self.device)
        self.sh_degree: int = 0

        self.optimzable_params: Optional[GroupParams] = None
        self.optimizer: Optional[Optimizer] = None

        self.splits = 0
        self.clones = 0
        self.culls = 0
        self.times = [0] * 3

    @property
    def num_points(self):
        return self.gaussian_model.num_points

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
        scales = torch.ones(7, 3, device=self.device) * -2.0
        opacities = torch.ones((7, 1), device=self.device) * 10.0
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
        quats = torch.zeros(7, 4, device=self.device)
        quats[:, 0] = 1

        acc_grad = torch.zeros(7, 1, device=self.device)
        denom = torch.zeros(7, 1, dtype=torch.int32, device=self.device)
        max_radii2D = torch.zeros(7, device=self.device)
        tensors = {
            "means": means,
            "scales": scales,
            "opacities": opacities,
            "sh_coeffs": sh_coeffs,
            "quats": quats,
            "acc_grad": acc_grad,
            "denom": denom,
            "max_radii2D": max_radii2D
        }
        values = {
            "num_points": 7,
            "mean_lr": 0
        }
        if add_axis:
            self.gaussian_model.add_gaussians((tensors, values))
        else:
            self.gaussian_model.restore((tensors, values))

    def init_gaussians(
            self,
            splats: int,
            load_path: Optional[Path] = None,
            point_cloud: PointCloud = None,
            logger: Optional[CSVLogger] = None
    ):
        gaussians = 0
        if load_path:
            print("Loading existing gaussians")
            means = torch.load(load_path / "means.pt")
            scales = torch.load(load_path / "scales.pt")
            opacities = torch.load(load_path / "opacities.pt")
            sh_coeffs = torch.load(load_path / "sh.pt")
            quats = torch.load(load_path / "quats.pt")
            acc_grad = torch.load(load_path / "acc_grad.pt")
            denom = torch.load(load_path / "denom.pt")
            max_radii2D = torch.zeros(means.shape[0], device=self.device)

            tensors = {
                "means": means,
                "scales": scales,
                "opacities": opacities,
                "sh_coeffs": sh_coeffs,
                "quats": quats,
                "acc_grad": acc_grad,
                "denom": denom,
                "max_radii2D": max_radii2D
            }
            values = {
                "num_points": means.shape[0],
                "mean_lr": 0.0001
            }

            self.sh_degree = 4

            self.gaussian_model.restore((tensors, values))
            gaussians = means.shape[0]

        elif point_cloud:
            print("Creating gaussians from SFM point cloud")
            gaussians = self.gaussian_model.create_from_pcd(point_cloud, 0.0001)
            self.sh_degree = 0
        else:
            print("Randomly initialize gaussians")
            gaussians = self.gaussian_model.create_from_random(splats, 0.0001)
            self.sh_degree = 0
        print(f"Initialized {gaussians} gaussians")
        if logger is not None:
            logger.log_scalar("n_gaussians", gaussians)

    def init_optimizer(self, optimizable_params: GroupParams):
        self.optimzable_params = optimizable_params
        self.optimizer = self.gaussian_model.init_optimizer(optimizable_params)

    def train(self, data_list: DataList, data: Data, logger: Optional[CSVLogger] = None):
        n_data = len(data_list)
        iterations = self.optimzable_params.iterations

        for itr in range(1, iterations+1):
            average_image_loss = 0
            average_loss = 0
            data_itr = 0
            self.gaussian_model.mean_lr = self.optimizer.update_learning_rate(itr)

            if itr % 1000 == 0 and self.sh_degree < 4:
                self.sh_degree += 1

            bg = torch.rand(3, device=self.device) if self.optimzable_params.random_background else self.background

            random.shuffle(data_list)
            for name in data_list:
                gt_view, camera, add_data = data[name]

                bg_depth = 20.0
                if "bg_depth" in add_data:
                    bg_depth = add_data["bg_depth"]
                
                nv_view, nv_alpha, nv_depth, t0, t1 = render(camera, self.gaussian_model, sh_degree=self.sh_degree, background_depth=bg_depth, color=bg)

                loss, image_loss = self.loss(gt_view, nv_view, nv_alpha, nv_depth, add_data)
                t2 = self.optimizer.back_propagate_loss(loss)

                average_image_loss += image_loss
                average_loss += loss.item()

                self.times[0] += t0
                self.times[1] += t1
                self.times[2] += t2

                print(f"Iteration {itr}/{iterations}, Data: {data_itr + 1}/{n_data}, Loss: {loss.item()}")
                data_itr += 1

                with torch.no_grad():
                    if itr < self.optimzable_params.iterations:
                        self.optimizer.step_loss()

            with torch.no_grad():
                if itr < self.optimzable_params.densify_until_iter and itr <= iterations-100:
                    self.gaussian_model.add_densification_states(
                        _RasterizeGaussians.getViewSpaceGradient(),
                        _RasterizeGaussians.getViewDepthGradient(),
                        _ProjectGaussians.getRadii())

                    if itr >= self.optimzable_params.densify_from_iter and itr % self.optimzable_params.densification_interval == 0:
                        max_screen_size = 2000 if itr > self.optimzable_params.opacity_reset_interval else None
                        extent = 200  # todo check
                        culls, clones, splits = self.gaussian_model.densify_and_prune(
                            position_grads=_ProjectGaussians.getPositionalGradient(),
                            grad_threshold=self.optimzable_params.densify_grad_threshold,
                            opacity_threshold=0.005,
                            size_threshold=0.01 * extent,
                            extent=extent,
                            max_screen_size=max_screen_size)
                        self.culls += culls
                        self.clones += clones
                        self.splits += splits
                        if logger is not None:
                            logger.log_scalar("culls", culls, itr)
                            logger.log_scalar("splits", splits, itr)
                            logger.log_scalar("clones", clones, itr)

                    if itr % self.optimzable_params.opacity_reset_interval == 0:
                        self.gaussian_model.reset_opacity()

            average_image_loss /= n_data
            if logger is not None:
                logger.log_scalar("average_image_loss", average_image_loss, itr)
                logger.log_scalar("average_loss", average_loss, itr)

    def save(self, save_path: Path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tensors, values = self.gaussian_model.capture()
        for key in tensors:
            name = key
            if key == "sh_coeffs":
                name = "sh"
            torch.save(tensors[key], f"{save_path}/{name}.pt")

    def render(
            self,
            camera: Camera,
            size: float = 1.0,
            background_depth: float = 20.0,
            color: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, float, float]:
        with torch.no_grad():
            out_img, _, out_depth, t0, t1 = render(camera, self.gaussian_model, size, 4, background_depth, color)
            return out_img, out_depth, t0, t1

    def loss(self, gt_view: Tensor, nv_view: Tensor, nv_alpha: Tensor = None, nv_depth: Tensor = None, add_data: dict = None) -> tuple[Tensor, float]:
        loss = (1.0 - self.optimzable_params.l_ssim) * l1_loss(nv_view, gt_view) \
            + self.optimzable_params.l_ssim * (1.0 - ssim(nv_view, gt_view))
        img_loss = loss.item()
        if "depth" in add_data and nv_depth is not None:
            loss += self.optimzable_params.l_depth * l1_loss(nv_depth, add_data["depth"])
        if "edges" in add_data and nv_depth is not None:
            depth_mask = (nv_depth>0).detach()
            nearDepthMean_map = nearMean_map(nv_depth, add_data["edges"]*depth_mask, kernelsize=3)
            loss += self.optimzable_params.l_smooth * l2_loss(nearDepthMean_map, nv_depth*depth_mask)
        return loss, img_loss

    def verify(
            self,
            data_list: DataList,
            data: Data,
            save_path: Optional[Path] = None,
            logger: Optional[CSVLogger] = None
    ) -> float:
        average_loss = 0.0
        print(f"Number gaussians: {self.gaussian_model.num_points}")
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

                nv_view, nv_alpha, nv_depth, _, _ = render(camera, self.gaussian_model, sh_degree=self.sh_degree, background_depth=bg_depth)
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
