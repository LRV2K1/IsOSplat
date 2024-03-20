from pathlib import Path
import numpy as np

import torch
from isosplat.camera import Camera

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


def create_camera_from_sfm_data(cam_data: prep.Camera, img_data: prep.Image, device: torch.device) -> Camera:
    if img_data.camera_id is not cam_data.id:
        raise Exception(f"image_data.camera_id: {img_data.camera_id}, and cam_data.id: {cam_data.id} are not the same")
    
    rot = np.transpose(img_data.qvec2rotmat())
    rot = rot.transpose()
    flip = np.zeros((3, 3))
    flip[0, 0] = 1
    flip[1, 1] = -1
    flip[2, 2] = -1
    rot = np.matmul(rot, flip)
    trans = np.array(img_data.tvec)
   
    tr = np.zeros((4, 4))
    tr[:3, :3] = rot
    tr[:3, 3] = trans
    tr[3, 3] = 1.0
    cam2world = np.linalg.inv(tr)
    cam_center = cam2world[:3, 3]

    # rotate camera by 180 degrees
    k = np.zeros((3, 3))
    k[0, 0] = -1
    k[1, 1] = -1
    k[2, 2] = 1
    nr = np.matmul(k, rot)
    rot = np.zeros((4, 4))
    rot[:3, :3] = nr
    rot[3, 3] = 1.0

    view = torch.tensor(np.float32(rot), device=device)

    focalx = float(cam_data.params[0])
    focaly = float(cam_data.params[0])
    cx = float(cam_data.params[1])
    cy = float(cam_data.params[2])
    camera = Camera(cam_data.width, cam_data.height, focalx, focaly, cx, cy, device)
    camera.set_position(float(cam_center[0]), float(cam_center[1]), float(cam_center[2]))
    camera.set_rotation(view)
    return camera
