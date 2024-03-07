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
        no_depth: bool = False
) -> None:
    device = torch.device("cuda:0")
    data_list = []
    data = {}
    point_cloud = None

    if img_path:
        match cam_model:
            case CamModel.CamFile:
                for file in os.listdir(img_path):
                    filename = os.fsdecode(file)
                    if filename.endswith(".cam"):
                        name = filename.split('.')[0]
                        gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png", device)
                        width, height = gt_image.shape[0], gt_image.shape[1]
                        camera = create_camera_from_cam_file(width, height, img_path / f"{name}.cam", device)
                        
                        data[name] = gt_image, camera, {}
                        data_list.append(name)
            case CamModel.SFM:
                if initialize == Initialize.SFM:
                    point_cloud = prep.read_points3D_binary(img_path / "sfm" / "0" / "points3D.bin")
                images = prep.read_extrinsics_binary(img_path / "sfm" / "0" / "images.bin")
                cameras = prep.read_intrinsics_binary(img_path / "sfm" / "0" / "cameras.bin")
                for image in images.values():
                    name = image.name.split('.')[0]
                    camera_data = cameras[image.camera_id]
                    camera = create_camera_from_sfm_data(camera_data, image, device)
                    gt_image, gt_alpha = image_path_to_tensor(img_path / f"{name}.png", device)
                    
                    data[name] = gt_image, camera, {}
                    data_list.append(name)
    else:
        gt_image = torch.ones((height, width, 3), device=device) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=device)
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0], device=device)

        add_data = {}

        gt_alpha = torch.ones((height, width), device=device) * 1.0
        add_data["alpha"] = gt_alpha
        if not no_depth:
            gt_depth = torch.ones((height, width), device=device) * 7.0
            add_data["depth"] = gt_depth

        camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
        data_list = ["test"]
        data = {"test": (gt_image, camera, add_data)}

    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud)
    trainer.init_optimizer(lr)
    if iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            iterations=iterations,
        )
    trainer.verify(data_list, data, save_path)
    if save_path:
        trainer.save(save_path)


if __name__ == '__main__':
    tyro.cli(main)
