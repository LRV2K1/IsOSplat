# import depth.sfm
# from depth.sfm import SFM
# from depth.sfm_types import *

from pathlib import Path
import os
from typing import Optional
from enum import Enum
# from PIL import Image

import tyro
# import numpy as np

# import torch
# from torch import Tensor

# from isosplat.camera import Camera
# from isosplat.gaussian_splatting import GaussianSplatting


class Initialize(Enum):
    Random = 0
    SFM = 1


class CamModel(Enum):
    CamFile = 0
    SFM = 1

#
# def image_path_to_tensor(image_path: Path) -> tuple[Tensor, Tensor]:
#     import torchvision.transforms as transforms
#
#     img = Image.open(image_path)
#     transform = transforms.ToTensor()
#     img_transform = transform(img)
#     img_tensor = img_transform.permute(1, 2, 0)[..., :3]
#     if (img_transform.shape[0] > 3):
#         img_alpha_tensor = img_transform.permute(1, 2, 0)[..., 3]
#     else:
#         img_alpha_tensor = torch.ones(img_tensor.shape[0], img_tensor.shape[0]) * 1.0
#     return img_tensor, img_alpha_tensor
#
#
# def create_camera_from_cam_file(width: int, height: int, cam_path: Path, device: torch.device) -> Camera:
#     with open(cam_path) as cam:
#         lines = cam.readlines()
#         focal = lines[0].split(',')
#         principal = lines[1].split(',')
#         pos = lines[2].split(',')
#         dir = lines[3].split(',')
#         camera = Camera(width, height,
#                         float(focal[0]), float(focal[1]),
#                         float(principal[0]), float(principal[1]),
#                         device)
#         camera.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
#         if len(lines) > 4:
#             top = lines[4].split(',')
#             camera.look_at_top(float(dir[0]), float(dir[1]), float(dir[2]), float(top[0]), float(top[1]), float(top[2]))
#         else:
#             camera.look_at(float(dir[0]), float(dir[1]), float(dir[2]))
#     return camera
#
#
# def create_camera_from_sfm_data(cam_data: CameraData,
#                                 pos: tuple[float, float, float],
#                                 dir: tuple[float, float, float],
#                                 device: torch.device) -> Camera:
#     width, height, focalx, focaly, cx, cy = cam_data
#     x, y, z = pos
#     dx, dy, dz = dir
#     camera = Camera(width, height, focalx, focaly, cx, cy, device)
#     camera.set_position(x, y, z)
#     camera.set_view_direction(dx, dy, dz)
#     return camera


def main(
        height: int = 256,
        width: int = 256,
        img_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        load_path: Optional[Path] = None,
        iterations: int = 1000,
        lr: float = 0.01,
        splats: int = 100000,
        initialize: Initialize = Initialize.Random,
        cam_model: CamModel = CamModel.CamFile,
        clean: bool = False
) -> None:
    if initialize == Initialize.SFM or cam_model == CamModel.SFM:
        from depth.sfm import SFM
        sfm = SFM(img_path)
        sfm.sfm(clean)
        camera_data = sfm.get_camera_data()
        image_data = sfm.get_image_data()

    import torch
    device = torch.device("cuda:0")
    data = []

    if img_path:
        match cam_model:
            case CamModel.CamFile:
                for file in os.listdir(img_path):
                    filename = os.fsdecode(file)
                    if filename.endswith(".cam"):
                        from isosplat.camera_constructor import image_path_to_tensor, create_camera_from_cam_file
                        name = filename.split('.')[0]
                        gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png")
                        width, height = gt_image.shape[0], gt_image.shape[1]
                        camera = create_camera_from_cam_file(width, height, img_path / f"{name}.cam", device)
                        data.append((gt_image, gt_alpha, camera, name))
            case CamModel.SFM:
                from isosplat.camera_constructor import image_path_to_tensor, create_camera_from_sfm_data
                for name, cam_id, pos, dir in image_data:
                    name = name.split('.')[0]
                    camera = create_camera_from_sfm_data(camera_data[cam_id], pos, dir, device)
                    gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png")
                    data.append((gt_image, gt_alpha, camera, name))
    else:
        from isosplat.camera import Camera
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])

        gt_alpha = torch.ones((height, width)) * 1.0

        camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
        data = [(gt_image, gt_alpha, camera, "test")]

    from isosplat.gaussian_splatting import GaussianSplatting
    trainer = GaussianSplatting(device)
    trainer.init_gaussians(splats, load_path, None)
    trainer.init_optimizer(lr)
    if iterations > 0 and len(data) > 0:
        trainer.train(
            data=data,
            iterations=iterations,
        )
    trainer.verify(data, save_path)
    if save_path:
        trainer.save(save_path)


if __name__ == '__main__':
    tyro.cli(main)
