from pathlib import Path
import numpy as np
import os

import torch
from torch import Tensor

from isosplat.utils import Data, DataList
from utils.graphics_utils import BasicPointCloud
from preprocess.colmap_loader import Image
from preprocess.image_loader import image_path_to_tensor, save_img_from_tensor


class ObjectSegmenter:
    def __init__(self, segment_path: Path):
        self.segment_path = segment_path

    def segment_objects(self, data: Data, point_cloud_id: dict[int, int], point_cloud: BasicPointCloud, sfm_images: dict[int, Image], device: torch.device):
        # xzys, _, errors = point_cloud

        image_segment_masks = {}

        offset = 0
        for sfm_image in sfm_images.values():
            name = sfm_image.name.split('.')[0]
            image_id = sfm_image.id
            print(f"Extracting segments {name}")

            gt_view, _, _ = data[name]

            segment_id = 1
            segments_mask = torch.zeros_like(gt_view[:,:,0], dtype=torch.int16, device=device)
            image_segment_path = self.segment_path / name
            for file in os.listdir(image_segment_path):
                # print(f"segment: {segment_id}")
                filename = os.fsdecode(file)
                if not filename.endswith(".png"):
                    continue
                segment, _ = image_path_to_tensor(image_segment_path / filename, device)
                segment = segment[:,:,0]
                segments_mask = self.add_to_segment_mask(segment_id, segments_mask, segment)
                
                segment_id += 1
            offset = max(segment_id, offset)

            image_segment_masks[image_id] = segments_mask

        point_to_segments: dict[int, list[int]] = {}
        segment_to_points: dict[int, list[int]] = {}

        for image_id in image_segment_masks:
            segments_mask = image_segment_masks[image_id]
            sfm_image = sfm_images[image_id]
            name = sfm_image.name.split('.')[0]
            ipids = sfm_image.point3D_ids
            ipids_mask = ipids >= 0
            ipids = ipids[ipids_mask]
            xys = np.int32(sfm_image.xys)[ipids_mask.tolist()]

            segment_offset = image_id * offset

            print(f"{name}, points: {xys.shape}/{sfm_image.xys.shape}")
                        
            segments_mask_list = segments_mask.tolist()

            for i in range(xys.shape[0]):
                x, y = xys[i]
                sid = segments_mask_list[int(y)][int(x)] + segment_offset
                pid = ipids[i]
                if sid not in segment_to_points:
                    segment_to_points[sid] = []
                segment_to_points[sid].append(pid)
                if pid not in point_to_segments:
                    point_to_segments[pid] = []
                point_to_segments[pid].append(sid)

        objects = self.combine_objects(point_to_segments, segment_to_points)
        print(len(objects))      

        new_image_segment_masks = {}
        for image_id in image_segment_masks:
            new_image_segment_masks[image_id] = torch.zeros_like(image_segment_masks[image_id], device=device) #, dtype=torch.int16

        object_id = 1
        for o in objects:
            for sid in o:
                image_id = sid//offset
                s = sid % offset
                segment_mask = image_segment_masks[image_id]
                new_segment_mask = new_image_segment_masks[image_id]
                new_segment_mask[segment_mask==s] = object_id
            object_id += 1  
        

    @staticmethod
    def add_to_segment_mask(segment_id: int, segments_mask: Tensor, segment: Tensor) -> Tensor:
        mask = segment > 0
        masked_segments_mask = segments_mask[mask]

        zero_mask = masked_segments_mask == 0
        masked_segments_mask[zero_mask] = segment_id     # set zero fields
        while masked_segments_mask[~zero_mask].shape[0] != 0:
            s_id = masked_segments_mask[~zero_mask][0].item()   # get id
            s_id_mask = masked_segments_mask == s_id            # get items in mask with id
            sm_id_mask = segments_mask == s_id                  # get all items with id
            # if all items with id whitin mask
            if (segments_mask[sm_id_mask].shape[0] == masked_segments_mask[s_id_mask].shape[0]):
                masked_segments_mask[s_id_mask] = segment_id    # completely over object
            # if all items within mask are id
            if (masked_segments_mask[~zero_mask].shape[0] == masked_segments_mask[s_id_mask].shape[0]):
                break   # completely inside object
            zero_mask = torch.logical_or(zero_mask, s_id_mask)

        segments_mask[mask] = masked_segments_mask
        
        return segments_mask
    
    @staticmethod
    def combine_objects(point_to_segments: dict[int, list[int]], segment_to_points: dict[int, list[int]]) -> list[list[int]]:
        combined_segments: dict[int, list[int]] = {}
        for sid in segment_to_points:
            pids = segment_to_points[sid]
            combined_segments[sid] = []
            for pid in pids:
                sids = point_to_segments[pid]
                combined_segments[sid] += sids
                combined_segments[sid] = remove_duplicates(combined_segments[sid])

        visited: dict[int, bool] = {}
        objects: list[list[int]] = []

        for sid in combined_segments:
            if sid in visited:
                continue
            local_object, visited = ObjectSegmenter._combine_object(sid, combined_segments, visited)
            objects.append(local_object)
        return objects
    
    @staticmethod
    def _combine_object(sid: int, combined_segments: dict[int, list[int]], visited: dict[int, bool]) -> tuple[list[int], dict[int, bool]]:
        if sid in visited:
            return [], visited
        visited[sid] = True
        local_object = [sid]
        for sid2 in combined_segments[sid]:
            new_object, visited = ObjectSegmenter._combine_object(sid2, combined_segments, visited)
            local_object += new_object
        return local_object, visited


def remove_duplicates(l: list[any]) -> list[any]:
    return list(dict.fromkeys(l))
                
        

