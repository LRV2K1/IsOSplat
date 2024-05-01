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
from scene.cameras import Camera
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss, l2_loss, ssim, nearMean_map

from utils.graphics_utils import BasicPointCloud
from utils.general_utils import image_rescale
from isosplat.utils import DataList, Data
from isosplat import render

from .project_gaussians import _ProjectGaussians
from .rasterize import _RasterizeGaussians


class GaussianSplatting:
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.background: Tensor = torch.zeros(3, device=self.device)

        self.gaussian_model: GaussianModel = GaussianModel(4, self.device)

        self.optimzable_params: Optional[GroupParams] = None

        self.splits = 0
        self.clones = 0
        self.culls = 0
        self.times = [0] * 3

    @property
    def num_points(self):
        return self.gaussian_model.get_xyz.shape[0]

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

        model_args = (
            means,
            sh_coeffs[:,0:1,:],
            sh_coeffs[:,1:,:],
            scales,
            quats,
            opacities,
            max_radii2D,
            acc_grad,
            denom,
            {},
            1.0
        )

        if add_axis:
            self.gaussian_model.append_gaussians(model_args)
        else:
            self.gaussian_model.partial_restore(model_args)

    def init_gaussians(
            self,
            splats: int,
            load_path: Optional[Path] = None,
            pcd: Optional[BasicPointCloud] = None,
            logger: Optional[CSVLogger] = None
    ):
        gaussians = 0
        if load_path:   # todo
            print("Loading existing gaussians")
            xyz = torch.load(load_path / "means.pt")
            scales = torch.load(load_path / "scales.pt")
            opacities = torch.load(load_path / "opacities.pt")
            features = torch.load(load_path / "sh.pt")
            rotation = torch.load(load_path / "quats.pt")
            acc_grad = torch.load(load_path / "acc_grad.pt")
            denom = torch.load(load_path / "denom.pt")
            max_radii2D = torch.zeros(xyz.shape[0], device=self.device)


            features_dc = features[:,0:1,:]
            features_rest = features[:,1:,]

            self.gaussian_model.partial_restore(
                (
                    4,
                    xyz,
                    features_dc,
                    features_rest,
                    scales,
                    rotation,
                    opacities,
                    max_radii2D,
                    acc_grad,
                    denom,
                    {},
                    1.0
                )
            )

            gaussians = xyz.shape[0]

        elif pcd:
            print("Creating gaussians from SFM point cloud")
            gaussians = self.gaussian_model.create_from_pcd(pcd, 1.0)
        else:
            print("Randomly initialize gaussians")
            gaussians = self.gaussian_model.create_from_random(splats, 10.0)
        print(f"Initialized {gaussians} gaussians")
        if logger is not None:
            logger.log_scalar("n_gaussians", gaussians)

    def init_optimizer(self, optimizable_params: GroupParams):
        self.optimzable_params = optimizable_params
        self.gaussian_model.training_setup(optimizable_params)

    def rescale_data(self, data: Data, scale: float) -> Data:
        use_data = data.copy()
        for name in use_data:
            nv_view, cam, _ = use_data[name]
            new_add_data = {}
            new_view = image_rescale(nv_view, scale, self.device)
            use_data[name] = new_view, cam, new_add_data
        return use_data

    def train(self, data_list: DataList, data: Data, logger: Optional[CSVLogger] = None):
        n_data = len(data_list)
        iterations = self.optimzable_params.iterations
        image_scale = 0.25

        use_data = self.rescale_data(data, image_scale)

        data_queue = []

        for itr in range(1, iterations+1):
            average_image_loss = 0
            average_loss = 0
            data_itr = 0
            lr = self.gaussian_model.update_learning_rate(itr)
            
            if itr == 250:
                print("rescale")
                image_scale = 0.5
                use_data = self.rescale_data(data, image_scale)
            if itr == 500:
                print("rescale")
                image_scale = 1
                use_data = data

            if itr % 1000 == 0:
                self.gaussian_model.oneupSHdegree()

            bg = torch.rand(3, device=self.device) if self.optimzable_params.random_background else self.background

            random.shuffle(data_list)
            for name in data_list:
            # if len(data_queue) == 0:
            #     data_queue = data_list.copy()
            #     random.shuffle(data_queue)
            # name = data_queue.pop()
            
                gt_view, camera, add_data = use_data[name]

                bg_depth = 20.0
                if "bg_depth" in add_data:
                    bg_depth = add_data["bg_depth"]
                
                nv_view, nv_alpha, nv_depth, t0, t1 = render(camera, self.gaussian_model, background_depth=bg_depth, color=bg, image_scale=image_scale)

                loss, image_loss = self.loss(gt_view, nv_view, nv_alpha, nv_depth, add_data)
                # t2 = self.optimizer.back_propagate_loss(loss)
                start = time.time()
                loss.backward()
                t2 = time.time() - start

                average_image_loss += image_loss
                average_loss += loss.item()

                self.times[0] += t0
                self.times[1] += t1
                self.times[2] += t2

                print(f"Iteration {itr}/{iterations}, Data: {data_itr + 1}/{n_data}, Loss: {loss.item()}")
                # print(f"Iteration {itr}/{iterations}, Loss: {loss.item()}")
                data_itr += 1

                with torch.no_grad():
                    if itr < self.optimzable_params.iterations:
                        # self.optimizer.step_loss()
                        self.gaussian_model.optimizer.step()
                        self.gaussian_model.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                if itr < self.optimzable_params.densify_until_iter and itr <= iterations-100:
                    self.gaussian_model.add_densification_stats(
                        _RasterizeGaussians.getViewSpaceGradient(),
                        _RasterizeGaussians.getViewDepthGradient(),
                        _ProjectGaussians.getRadii())

                    if itr >= self.optimzable_params.densify_from_iter and itr % self.optimzable_params.densification_interval == 0:
                        max_screen_size = 2000 if itr > self.optimzable_params.opacity_reset_interval else None
                        extent = 200  # todo check
                        culls, clones, splits = self.gaussian_model.densify_and_prune(
                            position_grads=_ProjectGaussians.getPositionalGradient(),
                            max_grad=self.optimzable_params.densify_grad_threshold,
                            min_opacity=0.005,
                            lr=lr,
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

        (
            sh_degree,
            xyz,
            features_dc,
            features_rest,
            scale,
            rotation,
            opacity,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            spatial_lr_scale
        ) = self.gaussian_model.capture()

        features = torch.cat((features_dc, features_rest), dim=1)

        torch.save(xyz_gradient_accum, f"{save_path}/acc_grad.pt")
        torch.save(denom, f"{save_path}/denom.pt")
        torch.save(max_radii2D, f"{save_path}/max_radii2D.pt")
        torch.save(xyz, f"{save_path}/means.pt")
        torch.save(opacity, f"{save_path}/opacities.pt")
        torch.save(rotation, f"{save_path}/quats.pt")
        torch.save(scale, f"{save_path}/scales.pt")
        torch.save(features, f"{save_path}/sh.pt")

        self.gaussian_model.save_ply(save_path / "model.ply")

    def render(
            self,
            camera: Camera,
            size: float = 1.0,
            background_depth: float = 20.0,
            color: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, float, float]:
        with torch.no_grad():
            out_img, _, out_depth, t0, t1 = render(camera, self.gaussian_model, size, background_depth, color)
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

                nv_view, nv_alpha, nv_depth, _, _ = render(camera, self.gaussian_model, background_depth=bg_depth)
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
