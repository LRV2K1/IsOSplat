from pathlib import Path
import numpy as np
import json

import torch
from torch import Tensor, optim

from isosplat.camera import Camera
from isosplat.utils import PointCloud
from preprocess.colmap_loader import Image


class DepthMapNormalizer:
    def normalize_depth_map(
            self,
            name: str,
            depth_map: np.ndarray,
            point_cloud_id: dict[int, int],
            point_cloud: PointCloud,
            image: Image,
            camera: Camera,
            device: torch.device
    ) -> tuple[Tensor, float, float]:
        print(f"Normalizing depth map {name}")
        depth_map = torch.tensor(np.float32(depth_map), device=device)
        
        xzys, _, errors = point_cloud

        ipids = torch.tensor(image.point3D_ids, device=device)
        ipids_mask = torch.where(ipids >= 0, True, False)
        ipids = ipids[ipids_mask]

        xys = np.int32(image.xys)[ipids_mask.tolist()]

        means = torch.tensor(np.float32(xzys), device=device)
        means_mask = []
        ipids_list = ipids.tolist()
        for i in ipids_list:
            means_mask.append(point_cloud_id[i])
        means = means[means_mask]
        depth_errors = torch.tensor(np.float32(errors), device=device)
        depth_errors = depth_errors[means_mask]
        depth_errors = torch.where(depth_errors > 1.0, depth_errors, 1.0)

        view, _ = camera.get_view_and_project_matrix()
        sparse_depths = (_get_depths(view, means) * (1.0 / depth_errors[:, 0]))

        dense_list = []
        for [y, x] in xys:
            dense_list.append(depth_map[int(x), int(y)])

        dense_depths = torch.tensor(dense_list, device=device)

        s, t = self._argmin(sparse_depths, dense_depths, device)
        return s * depth_map + t, s, t

    @staticmethod
    def _argmin_old(sparse_depths: Tensor, dense_depths: Tensor, device: torch.device) -> tuple[float, float]:
        sparse_depths.requires_grad = False
        dense_depths.requires_grad = False

        s = torch.rand(1, device=device)
        t = torch.rand(1, device=device)
        s.requires_grad = True
        t.requires_grad = True

        last_loss = 0.0
        current_loss = 1.0
        itr = 0
        optimizer = optim.SGD([s, t], 0.1)
        while last_loss != current_loss and itr < 5000:
            loss = torch.abs((sparse_depths - (s * dense_depths + t)) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_loss = current_loss
            current_loss = loss.item()

            itr += 1
        return s.item(), t.item()

    @staticmethod
    def _argmin(sparse_depths: Tensor, dense_depths: Tensor, device: torch.device) -> tuple[float, float]:
        sparse_depths = sparse_depths[:, None]
        dense_depths = dense_depths[:, None]
        dense_depths = torch.cat((dense_depths, torch.ones_like(dense_depths, device=device)), 1)
        dense_depths_t = torch.transpose(dense_depths, 0, 1)

        dense_depths_p = torch.linalg.inv(torch.matmul(dense_depths_t, dense_depths))
        dense_depths_p = torch.matmul(dense_depths_p, dense_depths_t)
        x = torch.matmul(dense_depths_p, sparse_depths)

        return x[0, 0].item(), x[1, 0].item()

    @staticmethod
    def normalize_depth_map_file(
            name: str,
            depth_map: np.ndarray,
            depth_file_path: Path,
            device: torch.device
    ) -> tuple[Tensor, float, float]:
        print(f"Normalizing depth map {name}")
        depth_map = torch.tensor(np.float32(depth_map), device=device)

        with open(depth_file_path) as f:
            data = f.read()

        scale_offsets = json.loads(data)
        s, t = scale_offsets[name]

        return s * depth_map + t, s, t


def _get_depths(view_matrix: Tensor, means: Tensor) -> Tensor:
    return view_matrix[2, 0] * means[:, 0] + \
        view_matrix[2, 1] * means[:, 1] + \
        view_matrix[2, 2] * means[:, 2] + \
        view_matrix[2, 3]
