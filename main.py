from pathlib import Path
import os
from typing import Optional
from enum import Enum

import tyro
import torch
from torch import Tensor

from isosplat.camera_constructor import image_path_to_tensor, create_camera_from_cam_file, create_camera_from_sfm_data
from isosplat.camera import Camera
from isosplat.gaussian_splatting import GaussianSplatting

import preprocess.colmap_loader as prep


class Initialize(Enum):
    Random = 0
    SFM = 1


class CamModel(Enum):
    CamFile = 0
    SFM = 1


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
    device = torch.device("cuda:0")
    data = []
    point_cloud = None

    if img_path:
        match cam_model:
            case CamModel.CamFile:
                for file in os.listdir(img_path):
                    filename = os.fsdecode(file)
                    if filename.endswith(".cam"):
                        name = filename.split('.')[0]
                        gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png")
                        width, height = gt_image.shape[0], gt_image.shape[1]
                        camera = create_camera_from_cam_file(width, height, img_path / f"{name}.cam", device)
                        data.append((gt_image, gt_alpha, camera, name))
            case CamModel.SFM:
                prep_point_cloud = prep.read_points3D_binary(img_path / "sfm" / "0" / "points3D.bin")
                prep_images = prep.read_extrinsics_binary(img_path / "sfm" / "0" / "images.bin")
                prep_cameras = prep.read_intrinsics_binary(img_path / "sfm" / "0" / "cameras.bin")
                for prep_image in prep_images.values():
                    name = prep_image.name.split('.')[0]
                    prep_camera = prep_cameras[prep_image.camera_id]
                    camera = create_camera_from_sfm_data(prep_camera, prep_image, device)
                    gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png")
                    data.append((gt_image, gt_alpha, camera, name))
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])

        gt_alpha = torch.ones((height, width)) * 1.0

        camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
        data = [(gt_image, gt_alpha, camera, "test")]

    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, prep_point_cloud)
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
