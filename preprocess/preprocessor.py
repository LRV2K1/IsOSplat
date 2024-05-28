from pathlib import Path
import os
from enum import Enum
from typing import Optional
import numpy as np

import torch
from torchrl.record import CSVLogger

from isosplat.camera import Camera
from utils.graphics_utils import BasicPointCloud
from isosplat.utils import Data, DataList
from .depth_map_normilizer import DepthMapNormalizer
from .edge_detector import CannyEdgeDetector, CV2CannyEdgeDetector
from .colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from .image_loader import image_path_to_tensor, save_img_from_tensor
from .camera_constructor import create_camera_from_cam_file, create_camera_from_sfm_data
from .marigold_loader import load_depth_map
from arguments import GroupParams, InitModel, CamModel, DepthModel
from .object_segmenter import ObjectSegmenter
from .segmentation import segment_object, create_objects, select_object


class PreProcessor:
    def __init__(self, data_path: Path):
        self.img_path = None
        self.sfm_path = None
        self.depth_path = None
        self.cams_path = None
        self.depth_file_path = None
        self.segment_path = None
        
        if data_path:
            self.img_path = data_path / "images"
            self.sfm_path = data_path / "sparse" / "0"
            self.depth_path = data_path / "depth_npy"
            self.cams_path = data_path / "cams"
            self.depth_file_path = self.depth_path / "st.json"
            self.segment_path = data_path / "segments"

        self.has_img = self.img_path is not None and os.path.exists(self.img_path)
        self.has_sfm = self.sfm_path is not None and os.path.exists(self.sfm_path)
        self.has_depth = self.depth_path is not None and os.path.exists(self.depth_path)
        self.has_cams = self.cams_path is not None and os.path.exists(self.cams_path)
        self.has_depth_file = self.depth_file_path is not None and os.path.exists(self.depth_file_path)

    def preprocess_data(
            self, device: torch.device,
            preprocess_params: GroupParams,
            logger: Optional[CSVLogger] = None
    ) -> tuple[DataList, Data, Optional[BasicPointCloud]]:
        
        if not self.has_img:
            print(f"No image file found or given, generating dummy data")
            return self._dummy_data(device, preprocess_params.depth_model, preprocess_params.no_alpha)
        if InitModel[preprocess_params.init_model] == InitModel.SFM and not self.has_sfm:
            raise Exception(f"Cannot initialize gaussians with SFM, sfm path not found: {self.sfm_path}")
        if CamModel[preprocess_params.cam_model] == CamModel.CamFile and not self.has_cams:
            raise Exception(f"Cannot create cameras with cam files, cams path not found: {self.cams_path}")
        if CamModel[preprocess_params.cam_model] == CamModel.SFM and not self.has_sfm:
            raise Exception(f"Cannot create cameras with SFM, sfm path not found: {self.sfm_path}")
        if not DepthModel[preprocess_params.depth_model] == DepthModel.NoDepth and not self.depth_path:
            raise Exception(f"Cannot create depth maps, depth_npy path not found: {self.depth_path}")
        if DepthModel[preprocess_params.depth_model] == DepthModel.SFM and not self.has_sfm:
            raise Exception(f"Cannot normalize depth maps with SFM, sfm path not found: {self.sfm_path}")
        if DepthModel[preprocess_params.depth_model] == DepthModel.DepthFile and not self.has_depth_file:
            raise Exception(f"Cannot normalize depth maps with depth file, depth_file path not found: {self.depth_file_path}")

        pid, point_cloud = None, None
        sfm_images = None
        sfm_cameras = None
        if InitModel[preprocess_params.init_model] == InitModel.SFM or CamModel[preprocess_params.cam_model] == CamModel.SFM:
            pid, pc = read_points3D_binary(self.sfm_path / "points3D.bin")
            pc = self._flip_point_cloud(pc)
            xyzs, colors, errors = pc
            point_cloud = BasicPointCloud(xyzs, colors, np.zeros_like(xyzs), errors)
            sfm_images = read_extrinsics_binary(self.sfm_path / "images.bin")
            sfm_cameras = read_intrinsics_binary(self.sfm_path / "cameras.bin")


        data = Data({})
        data_list = DataList([])

        match CamModel[preprocess_params.cam_model]:
            case CamModel.CamFile:
                for file in os.listdir(self.cams_path):
                    filename = os.fsdecode(file)
                    if not filename.endswith(".cam"):
                        continue
                    name = filename.split('.')[0]
                    add_data = {}
                    gt_image, gt_alpha = image_path_to_tensor(self.img_path / f"{name}.png", device)
                    if not preprocess_params.no_alpha:
                        add_data["alpha"] = gt_alpha

                    height, width = gt_image.shape[0], gt_image.shape[1]
                    camera = create_camera_from_cam_file(width, height, self.cams_path / f"{name}.cam", device)

                    data[name] = gt_image, camera, add_data
                    data_list.append(name)
            case CamModel.SFM:
                for sfm_image in sfm_images.values():
                    name = sfm_image.name.split('.')[0]
                    add_data = {}
                    gt_image, gt_alpha = image_path_to_tensor(self.img_path / f"{name}.jpg", device)
                    if not preprocess_params.no_alpha:
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
                    if not preprocess_params.no_alpha:
                        add_data["alpha"] = gt_alpha

                    width, height = gt_image.shape[0], gt_image.shape[1]

                    camera = Camera(width, height, width/2, height/2, width/2, height/2, device)
                    camera.set_position(0, 0, 8.0)
                    camera.look_at(0, 0, 0)

                    data[name] = gt_image, camera, add_data
                    data_list.append(name)
            case _:
                raise Exception(f"No cam model found: {CamModel[preprocess_params.cam_model]}")

        depth_map_normalizer = DepthMapNormalizer()
        # edge_detector = CannyEdgeDetector(device)
        edge_detector = CV2CannyEdgeDetector(preprocess_params.edge_low, preprocess_params.edge_high)

        if not DepthModel[preprocess_params.depth_model] == DepthModel.NoDepth:
            for name in data_list:
                gt_image, camera, add_data = data[name]
                edge_map = edge_detector.calculate_edge_map(name, gt_image, device)
                add_data["edges"] = edge_map
                data[name] = gt_image, camera, add_data
                # save_img_from_tensor(edge_map, "edges", name)

        depth_parameters = {}
        if DepthModel[preprocess_params.depth_model] == DepthModel.SFM:
            for sfm_image in sfm_images.values():
                name = sfm_image.name.split('.')[0]
                gt_image, camera, add_data = data[name]

                npy_depth_map = load_depth_map(self.depth_path / f"{name}_pred.npy")
                depth_map, s, t = depth_map_normalizer.normalize_depth_map(
                    name=name,
                    depth_map=npy_depth_map,
                    point_cloud_id=pid,
                    point_cloud=point_cloud,
                    image=sfm_image,
                    camera=camera,
                    device=device)
                depth_parameters[f"depth scale (s): {name}"] = s
                depth_parameters[f"depth offset (t): {name}"] = t
                
                add_data["depth"] = depth_map
                add_data["bg_depth"] = torch.max(depth_map).item() + 10
                data[name] = gt_image, camera, add_data
        if DepthModel[preprocess_params.depth_model] == DepthModel.DepthFile:
            for name in data_list:
                gt_image, camera, add_data = data[name]

                npy_depth_map = load_depth_map(self.depth_path / f"{name}_pred.npy")
                depth_map, s, t = depth_map_normalizer.normalize_depth_map_file(
                    name=name,
                    depth_map=npy_depth_map,
                    depth_file_path=self.depth_file_path,
                    device=device)
                depth_parameters[f"depth scale (s): {name}"] = s
                depth_parameters[f"depth offset (t): {name}"] = t

                add_data["depth"] = depth_map
                add_data["bg_depth"] = torch.max(depth_map).item() + 10
                data[name] = gt_image, camera, add_data

        if logger is not None:
            logger.log_hparams(depth_parameters)

        if sfm_images is not None:
            # object_segmenter = ObjectSegmenter(self.segment_path)
            # object_segmenter.segment_objects(data, pid, point_cloud, sfm_images, device)
            # object_segmenter.segment_objects_new(data, pid, point_cloud, sfm_images, device)
            segments_map, features_map, segments_mask_map = segment_object(self.segment_path, sfm_images, sfm_cameras, device, 0.00, 0.1)
            objects_map, obj_segments_dict, _, _, _ = create_objects(segments_map, features_map, segments_mask_map, sfm_images, sfm_cameras, device, 0.75, Path("segments/fortress"))
            img_masks, points = select_object(objects_map, obj_segments_dict, sfm_images, sfm_cameras, device)
            for img_id in img_masks:
                mask = img_masks[img_id]
                name = sfm_images[img_id].name
                name = name.split('.')[0]
                image, cam, dict = data[name]
                dict["mask"] = mask
                data[name] = image, cam, dict

            point_ids = []
            for point_id in points:
                point_ids.append(pid[point_id])
            bp_points = point_cloud.points[point_ids]
            bp_colors = point_cloud.colors[point_ids]
            bp_normals = point_cloud.normals[point_ids]
            bp_errors = point_cloud.errors[point_ids]
            point_cloud = BasicPointCloud(bp_points, bp_colors, bp_normals, bp_errors)
                    
        if InitModel[preprocess_params.init_model] == InitModel.Random:
            return data_list, data, None
        else:
            return data_list, data, point_cloud

    @staticmethod
    def _dummy_data(
            device: torch.device,
            depth_model: DepthModel,
            no_alpha: bool
    ) -> tuple[DataList, Data, None]:
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
        data_list = DataList(["test"])
        data = Data({"test": (gt_image, camera, add_data)})

        return data_list, data, None

    @staticmethod
    def _flip_point_cloud(point_cloud: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyzs, rgbs, errors = point_cloud

        xyzs[:, 1] = xyzs[:, 1] * -1.0
        xyzs[:, 2] = xyzs[:, 2] * -1.0

        return (xyzs, rgbs, errors)
