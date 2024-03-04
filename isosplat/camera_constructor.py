from pathlib import Path

import torch
from torch import Tensor
from isosplat.camera import Camera

from PIL import Image


def create_camera_from_cam_file(width: int, height: int, cam_path: Path, device: torch.device) -> Camera:
    with open(cam_path) as cam:
        lines = cam.readlines()
        focal = lines[0].split(',')
        principal = lines[1].split(',')
        pos = lines[2].split(',')
        dir = lines[3].split(',')
        camera = Camera(width, height,
                        float(focal[0]), float(focal[1]),
                        float(principal[0]), float(principal[1]),
                        device)
        camera.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        # camera data is in left handed coordinate system, use right handed coordinate system
        if len(lines) > 4:
            top = lines[4].split(',')
            camera.look_at_top(float(dir[0]), float(dir[1]), -float(dir[2]), float(top[0]), float(top[1]), float(top[2]))
        else:
            camera.look_at(float(dir[0]), float(dir[1]), -float(dir[2]))
    return camera


def create_camera_from_sfm_data(cam_data: tuple[int, int, float, float, float, float],
                                pos: tuple[float, float, float],
                                dir: tuple[float, float, float],
                                device: torch.device) -> Camera:
    width, height, focalx, focaly, cx, cy = cam_data
    x, y, z = pos
    dx, dy, dz = dir
    camera = Camera(width, height, float(focalx), float(focaly), float(cx), float(cy), device)
    camera.set_position(float(x), float(y), float(z))
    camera.set_view_direction(float(dx), float(dy), float(dz))
    return camera


def image_path_to_tensor(image_path: Path) -> tuple[Tensor, Tensor]:
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_transform = transform(img)
    img_tensor = img_transform.permute(1, 2, 0)[..., :3]
    if img_transform.shape[0] > 3:
        img_alpha_tensor = img_transform.permute(1, 2, 0)[..., 3]
    else:
        img_alpha_tensor = torch.ones(img_tensor.shape[0], img_tensor.shape[0]) * 1.0
    return img_tensor, img_alpha_tensor
