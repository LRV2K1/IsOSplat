from pathlib import Path

import torch
from torch import Tensor
from isosplat.camera import Camera

from PIL import Image
import numpy as np

import preprocess.colmap_loader as prep


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
        if len(lines) > 4:
            top = lines[4].split(',')
            camera.look_at_top(float(dir[0]), float(dir[1]), float(dir[2]), float(top[0]), float(top[1]), float(top[2]))
        else:
            camera.look_at(float(dir[0]), float(dir[1]), float(dir[2]))
    return camera


def create_camera_from_sfm_data(cam_data: tuple[int, int, float, float, float, float],
                                pos: tuple[float, float, float],
                                dir: tuple[float, float, float],
                                device: torch.device) -> Camera:
    width, height, focalx, focaly, cx, cy = cam_data
    x, y, z = pos
    dx, dy, dz = dir
    # camera data is in right-handed coordinate system with y down,
    # use right-handed coordinate system with y up
    camera = Camera(width, height, float(focalx), float(focaly), float(cx), float(cy), device)
    camera.set_position(float(x), -float(y), -float(z))
    camera.set_view_direction(float(dx), -float(dy), -float(dz))
    return camera


def create_camera_from_prep_sfm(cam_data: prep.Camera, img_data: prep.Image, device: torch.device) -> Camera:
    if img_data.camera_id is not cam_data.id:
        print("wrong")
        return None
    
    print(img_data.name)
    R = np.transpose(img_data.qvec2rotmat())
    T = np.array(img_data.tvec)
   
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]

    # rotate camera by 180 degrees
    rot = np.zeros((3, 3))
    rot[0,0] = -1
    rot[1,1] = -1
    rot[2,2] = 1
    nr = np.matmul(rot, R.transpose())
    Rtt = np.zeros((4, 4))
    Rtt[:3, :3] = nr
    Rtt[3, 3] = 1.0

    view = torch.tensor(np.float32(Rtt), device=device)
    camera = Camera(cam_data.width, cam_data.height, float(cam_data.params[0]), float(cam_data.params[0]), float(cam_data.params[1]), float(cam_data.params[2]), device)
    camera.set_position(float(cam_center[0]), float(cam_center[1]), float(cam_center[2]))
    camera.set_rotation(view)
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
