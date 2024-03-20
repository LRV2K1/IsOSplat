from pathlib import Path
import os
from enum import Enum
from typing import Optional

import torch
from torch import Tensor
from torchrl.record import CSVLogger

from isosplat.camera import Camera
from .depth_map_normilizer import PointCloud, DepthMapNormalizer
from .edge_detector import CannyEdgeDetector, CV2CannyEdgeDetector
from .colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from .image_loader import image_path_to_tensor, save_img_from_tensor
from .camera_constructor import create_camera_from_cam_file, create_camera_from_sfm_data
from .marigold_loader import load_depth_map


class Initialize(Enum):
    Random = 0
    SFM = 1


class CamModel(Enum):
    NoCam = 0
    CamFile = 1
    SFM = 2

class DepthModel(Enum):
    NoDepth = 0
    DepthFile = 1
    SFM = 2


class PreProcessor:
    def __init__(self, data_path: Path):
        self.img_path = None
        self.sfm_path = None
        self.depth_path = None
        self.cams_path = None
        self.depth_file_path = None
        
        if data_path:
            self.img_path = data_path / "images"
            self.sfm_path = data_path / "sfm" / "0"
            self.depth_path = data_path / "depth_npy"
            self.cams_path = data_path / "cams"
            self.depth_file_path = self.depth_path / "st.json"

        self.has_img = self.img_path != None and os.path.exists(self.img_path)
        self.has_sfm = self.sfm_path != None and os.path.exists(self.sfm_path)
        self.has_depth = self.depth_path != None and os.path.exists(self.depth_path)
        self.has_cams = self.cams_path != None and os.path.exists(self.cams_path)
        self.has_depth_file = self.depth_file_path != None and os.path.exists(self.depth_file_path)

    def preprocess_data(
            self, device: torch.device,
            initialize: Initialize = Initialize.Random, cam_model: CamModel = CamModel.NoCam,
            depth_model: DepthModel = DepthModel.NoDepth, no_alpha: bool = True,
            edge_low: float = 0.5, edge_high: float = 0.8,
            logger: Optional[CSVLogger] = None
                        ) -> tuple[list[str], dict[str, tuple[Tensor, Camera, dict[str, Tensor]]], PointCloud]:
        
        if not self.has_img:
            print(f"No image file found or given, generating dummy data")
            return self._dummy_data(device, depth_model, no_alpha)
        if initialize == Initialize.SFM and not self.has_sfm:
            raise Exception(f"Cannot initialize gaussians with SFM, sfm path not found: {self.sfm_path}")
        if cam_model == CamModel.CamFile and not self.has_cams:
            raise Exception(f"Cannot create cameras with cam files, cams path not found: {self.cams_path}")
        if cam_model == CamModel.SFM and not self.has_sfm:
            raise Exception(f"Cannot create cameras with SFM, sfm path not found: {self.sfm_path}")
        if not depth_model == DepthModel.NoDepth and not self.depth_path:
            raise Exception(f"Cannot create depth maps, depth_npy path not found: {self.depth_path}")
        if depth_model == DepthModel.SFM and not self.has_sfm:
            raise Exception(f"Cannot normalize depth maps with SFM, sfm path not found: {self.sfm_path}")
        if depth_model == DepthModel.DepthFile and not self.has_depth_file:
            raise Exception(f"Cannot nomralize depth maps with depth file, depth_file path not found: {self.depth_file_path}")

        pid, point_cloud = None, None
        sfm_images = None
        sfm_cameras = None
        if initialize == Initialize.SFM or cam_model == CamModel.SFM:
            pid, point_cloud = read_points3D_binary(self.sfm_path / "points3D.bin")
            point_cloud = self._flip_point_cloud(point_cloud)
            sfm_images = read_extrinsics_binary(self.sfm_path / "images.bin")
            sfm_cameras = read_intrinsics_binary(self.sfm_path / "cameras.bin")

        data = {}
        data_list = [] 

        match cam_model:
            case CamModel.CamFile:
                for file in os.listdir(self.cams_path):
                    filename = os.fsdecode(file)
                    if not filename.endswith(".cam"):
                        continue
                    name = filename.split('.')[0]
                    add_data = {}
                    gt_image, gt_alpha = image_path_to_tensor(self.img_path / f"{name}.png", device)
                    if not no_alpha:
                        add_data["alpha"] = gt_alpha

                    height, width = gt_image.shape[0], gt_image.shape[1]
                    camera = create_camera_from_cam_file(width, height, self.cams_path / f"{name}.cam", device)

                    data[name] = gt_image, camera, add_data
                    data_list.append(name)
            case CamModel.SFM:
                for sfm_image in sfm_images.values():
                    name = sfm_image.name.split('.')[0]
                    add_data = {}
                    gt_image, gt_alpha = image_path_to_tensor(self.img_path / f"{name}.png", device)
                    if not no_alpha:
                        add_data["alpha"] = gt_alpha

                    sfm_camera = sfm_cameras[sfm_image.camera_id]
                    camera = create_camera_from_sfm_data(sfm_camera, sfm_image, device)

                    data[name] = gt_image, camera, add_data
                    data_list.append(name)
            case CamModel.NoCam:
                for file in os.listdir(self.img_path):
                    filename = os.fsdecode(file)
                    if not filename.endswith(".png"):
                        continue
                    name = filename.split('.')[0]
                    add_data = {}
                    gt_image, gt_alpha = image_path_to_tensor(self.img_path / f"{name}.png", device)
                    if not no_alpha:
                        add_data["alpha"] = gt_alpha

                    width, height = gt_image.shape[0], gt_image.shape[1]

                    camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
                    camera.set_position(0, 0, 8.0)
                    camera.look_at(0, 0, 0)

                    data[name] = gt_image, camera, add_data
                    data_list.append(name)

        depth_map_normalizer = DepthMapNormalizer()
        # edge_detector = CannyEdgeDetector(device)
        edge_detector = CV2CannyEdgeDetector(edge_low, edge_high)

        if not depth_model == DepthModel.NoDepth:
            for name in data_list:
                gt_image, camera, add_data = data[name]
                edge_map = edge_detector.calculate_edge_map(name, gt_image, device)
                add_data["edges"] = edge_map
                data[name] = gt_image, camera, add_data
                # save_img_from_tensor(edge_map, "edges", name)

        depth_parameters = {}
        if depth_model == DepthModel.SFM:
            for sfm_image in sfm_images.values():
                name = sfm_image.name.split('.')[0]
                gt_image, camera, add_data = data[name]

                npy_depth_map = load_depth_map(self.depth_path / f"{name}_pred.npy")
                depth_map, s, t = depth_map_normalizer.normalize_depth_map(name, npy_depth_map,
                    pid, point_cloud, sfm_image, camera, device)
                depth_parameters[f"depth scale (s): {name}"] = s
                depth_parameters[f"depth offset (t): {name}"] = t
                
                add_data["depth"] = depth_map
                add_data["bg_depth"] = torch.max(depth_map).item() + 10
                data[name] = gt_image, camera, add_data
        if depth_model == DepthModel.DepthFile:
            for name in data_list:
                gt_image, camera, add_data = data[name]

                npy_depth_map = load_depth_map(self.depth_path / f"{name}_pred.npy")
                depth_map, s, t = depth_map_normalizer.normalize_depth_map_file(name, npy_depth_map, self.depth_file_path, device)
                depth_parameters[f"depth scale (s): {name}"] = s
                depth_parameters[f"depth offset (t): {name}"] = t

                add_data["depth"] = depth_map
                add_data["bg_depth"] = torch.max(depth_map).item() + 10
                data[name] = gt_image, camera, add_data

        if logger is not None:
            logger.log_hparams(depth_parameters)

        if initialize == Initialize.Random:
            return data_list, data, None
        else:
            return data_list, data, point_cloud
        
    def _dummy_data(self, device: torch.device, depth_model: DepthModel, no_alpha: bool) -> tuple[list[str], dict[str, tuple[Tensor, Camera, dict[str, any]]], None]:
        height = 256
        width = 256
        
        gt_image = torch.ones((height, width, 3), device=device) * 1.0
        gt_image = torch.ones((height, width, 3), device=device) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=device)
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0], device=device)

        add_data = {}        
        if not no_alpha:
            gt_alpha = torch.ones((height, width), device=device) * 1.0
            add_data["alpha"] = gt_alpha
        if not depth_model == DepthModel.NoDepth:
            gt_depth = (torch.ones((height, width), device=device) * 7.0)
            add_data["depth"] = gt_depth
            add_data["bg_depth"] = 15

        camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
        camera.set_position(0, 0, 8.0)
        camera.look_at(0, 0, 0)
        data_list = ["test"]
        data = {"test": (gt_image, camera, add_data)}

        return data_list, data, None

    def _flip_point_cloud(self, point_cloud: PointCloud) -> PointCloud:
        xyzs, rgbs, errors = point_cloud

        xyzs[:, 1] = xyzs[:, 1] * -1.0
        xyzs[:, 2] = xyzs[:, 2] * -1.0

        return PointCloud((xyzs, rgbs, errors))
