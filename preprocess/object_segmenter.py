from pathlib import Path
import numpy as np
import os
from typing import Optional

import torch
from torch import Tensor

from isosplat.utils import Data, DataList
import isosplat.cuda as _C
from utils.graphics_utils import BasicPointCloud
from preprocess.colmap_loader import Image
from preprocess.image_loader import image_path_to_tensor, save_img_from_tensor


class ObjectSegmenter:
    def __init__(self, segment_path: Path):
        self.segment_path = segment_path

    def segment_objects_new(self, data: Data, point_cloud_id: dict[int, int], point_cloud: BasicPointCloud, sfm_images: dict[int, Image], device: torch.device):
        # closing_kernel = torch.tensor(
        #     [
        #         [0, 0, 1, 1, 1, 0, 0],
        #         [0, 1, 1, 1, 1, 1, 0],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1],
        #         [0, 1, 1, 1, 1, 1, 0],
        #         [0, 0, 1, 1, 1, 0, 0]
        #     ],
        #     dtype=torch.bool,
        #     device=device
        # )
        self.closing_kernel = torch.tensor(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0]
            ],
            dtype=torch.bool,
            device=device
        )
        self.growing_kernel = torch.tensor(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ],
            dtype=torch.bool,
            device=device
        )

        # image_id -> {segment_id -> [feature_id]}
        segments_map: dict[int, dict[int, list[int]]] = {}
        # feature_id -> [(image_id, segment_id)]
        features_map: dict[int, list[tuple[int, int]]] = {}
        final_masks: dict[int, dict[int, Tensor]] = {}

        for sfm_image in sfm_images.values():
            name = sfm_image.name.split('.')[0]
            image_id = sfm_image.id
            print(f"Extracting segments {name}")

            # get feature points
            ipids = sfm_image.point3D_ids
            ipids_mask = ipids >= 0
            ipids = ipids[ipids_mask]
            xys = torch.tensor(np.int32(sfm_image.xys)[ipids_mask.tolist()], device=device)
            
            n_segments = 0
            n_discarded = 0
            segment_dict = {}
            discarded_masks = []
            image_segment_path = self.segment_path / name

            gt_view, _, _ = data[name]
            total_mask = torch.zeros_like(gt_view[:,:,0], dtype=torch.bool, device=device)
            image_final_mask = {}

            for file in os.listdir(image_segment_path):     # for every segment in the image
                filename = os.fsdecode(file)
                if not filename.endswith(".png"):
                    continue
                segment_id = int(filename.split('.')[0])
                segment, _ = image_path_to_tensor(image_segment_path / filename, device)
                segment_mask = segment[:,:,0] > 0
                xy_mask = _C.extract_segment_features(segment_mask, xys)
                total_mask[segment_mask] = True

                feature_list = []
                segment_ipids = ipids[xy_mask.cpu()]
                for pid in segment_ipids:
                    feature_list.append(pid)
                    if pid not in features_map:
                        features_map[pid] = []
                    features_map[pid].append((image_id, segment_id))
                if len(feature_list) > 0:
                    # total_mask[segment_mask] = True
                    segment_dict[segment_id] = feature_list
                    n_segments += 1
                    image_final_mask[segment_id] = segment_mask
                else:
                    n_discarded += 1
                    discarded_masks.append(segment_mask)
            
            # empty masks
            new_total_mask = _C.closing(self.closing_kernel, total_mask)
            torch.cuda.synchronize()
            total_masks = self.region_mapping(new_total_mask, device)
            segment_id = -1
            for segment_mask in total_masks:
                xy_mask = _C.extract_segment_features(segment_mask, xys)
                total_mask[segment_mask] = True

                feature_list = []
                segment_ipids = ipids[xy_mask.cpu()]
                for pid in segment_ipids:
                    feature_list.append(pid)
                    if pid not in features_map:
                        features_map[pid] = []
                    features_map[pid].append((image_id, segment_id))
                if len(feature_list) > 0:
                    # total_mask[segment_mask] = True
                    segment_dict[segment_id] = feature_list
                    n_segments += 1
                    image_final_mask[segment_id] = segment_mask
                else:
                    n_discarded += 1
                    discarded_masks.append(segment_mask)
                segment_id -= 1

            image_final_mask, n_combined = self.add_descarded_segments(image_final_mask, discarded_masks)
            final_masks[image_id] = image_final_mask
            segments_map[image_id] = segment_dict
            print(f"{n_segments} segments extracted")
            print(f"{n_combined} segments combined")
            print(f"{n_discarded} segments discarded")

            filePath = f"segments/total6"
            save_img_from_tensor(total_mask, filePath, f"{image_id}")
 
        # [([(image_id, segment_id)], [feature_id])]
        # objects = self.combine_segments(segments_map, features_map, 5)
        objects = self.combine_segments_percentage(segments_map, features_map, 0.80)

        n_dis_objects = 0
        for masks, points in objects:
            if len(points) < 5:
                objects.remove((masks, points))
                n_dis_objects += 1
        print(f"{len(objects)} objects found")
        print(f"{n_dis_objects} objects discarded")

        object_masks: list[dict[int, Tensor]] = self.create_object_masks(objects, data, sfm_images, final_masks, device, None)

        id = 0
        for masks, points in object_masks:
            for image_id in masks:
                filePath = f"segments/new_segments8"
                save_img_from_tensor(masks[image_id], filePath, f"{id}-{image_id}")
            id += 1
        

    @staticmethod                                                                                                                                       # [([(image_id, segment_id)], [feature_id])]
    def combine_segments(segments_map: dict[int, dict[int, list[int]]], features_map: dict[int, list[tuple[int, int]]], number_feature_threshold: int) -> list[tuple[list[tuple[int, int]], list[int]]]:
        print("Combining segments")
        done_segments: dict[tuple[int, int], bool] = {}
        
        c_segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
        combined_segments: list[tuple[list[tuple[int, int]], list[int]]] = []

        # create segments queue
        segment_queue: list[tuple[int, int]] = []
        for image_key in segments_map:
            for segment_key in segments_map[image_key]:
                segment_queue.append((image_key, segment_key))
        print(f"Initial segment queue lenght: {len(segment_queue)}")

        while len(segment_queue) > 0:
            image_key, segment_key = segment_queue.pop()        # pick a segment
            if (image_key, segment_key) in done_segments:
                continue
            if (image_key, segment_key) not in c_segments:
                c_segments[(image_key, segment_key)] = [(image_key, segment_key)]

            other_segments: dict[tuple[int, int], int] = {}
            for pid in segments_map[image_key][segment_key]:    # for every feature
                p_segments = features_map[pid]
                for p_segment in p_segments:                    # go over every segment that feature is in
                    if p_segment in done_segments or \
                        p_segment == (image_key, segment_key):  # if the segment is done, skip it
                        continue
                    if p_segment in other_segments:             # if segment already seen before
                        other_segments[p_segment] += 1          # add point
                    else:                                       # if segment not seen before
                        other_segments[p_segment] = 1           # initialize segment
            
            added = False
            for segment in other_segments:                              # go over all other segments
                if other_segments[segment] >= number_feature_threshold: # check feature threshold
                    done_segments[segment] = True
                    added = True
                    o_image_key, o_segment_key = segment
                    pids = segments_map[image_key][segment_key]
                    o_pids = segments_map[o_image_key][o_segment_key]
                    n_pids = remove_duplicates(pids + o_pids)
                    segments_map[image_key][segment_key] = n_pids       # add points to segment
                    for pid in n_pids:                                  # add segment to points
                        features_map[pid].append((image_key, segment_key))
                        features_map[pid] = remove_duplicates(features_map[pid])
                    # store combined segments
                    new_segments = [segment]
                    if segment in c_segments:
                        new_segments = c_segments[segment]
                    c_segments[(image_key, segment_key)] = remove_duplicates(c_segments[(image_key, segment_key)] + new_segments)
            if added:
                segment_queue.append((image_key, segment_key))          # segment added to queue
            else:
                done_segments[(image_key, segment_key)] = True          # segment done
                print(f"done: {len(segments_map[image_key][segment_key])}")
                if len(segments_map[image_key][segment_key]) > 0:
                    combined_segments.append((c_segments[(image_key, segment_key)], segments_map[image_key][segment_key]))
                
        return combined_segments


    def create_object_masks(self, objects: list[tuple[list[tuple[int, int]], list[int]]], data: Data, sfm_images: dict[int, Image], final_masks: dict[int, dict[int, Tensor]], device: torch.device, point_padd: Optional[int] = None) -> list[tuple[dict[int, Tensor]], list[int]]:   
        print("Creating object masks")
        object_masks: list[dict[int, Tensor]] = []
        object_id = 0

        distribution = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]], device=device)
        kernel = distribution * distribution.transpose(0,1)

        for segments, points in objects:
            print(f"Creating mask {object_id}")
            masks = {}
            for image_id, segment_id in segments:
                sfm_image = sfm_images[image_id]
                name = sfm_image.name.split('.')[0]

                if image_id not in masks:
                    gt_view, _, _ = data[name]
                    mask = torch.zeros_like(gt_view[:,:,0], device=device)
                    masks[image_id] = mask
                segment_mask = final_masks[image_id][segment_id]
                masks[image_id][segment_mask] = 1

            if point_padd:
                print("Padd mask with feature points")
                np_map = np.vectorize(lambda a : True if a in points else False)
                for sfm_image in sfm_images.values():
                    image_id = sfm_image.id
                    if image_id in masks:
                        ipids = sfm_image.point3D_ids
                        ipids_mask = np_map(ipids)
                        xys = torch.tensor(np.int32(sfm_image.xys)[ipids_mask.tolist()], device=device)
                        masks[image_id] = _C.padd_features(kernel, masks[image_id], xys)
            
            # remove small objects
            for image_id in masks:
                masks[image_id] = _C.opening(self.closing_kernel, _C.closing(self.growing_kernel, masks[image_id] > 0))

            print(f"{len(masks)} masks created for object {object_id}")
            object_id += 1
            object_masks.append((masks, points))

        return object_masks


    @staticmethod                                                                                                                                       # [([(image_id, segment_id)], [feature_id])]
    def combine_segments_percentage(segments_map: dict[int, dict[int, list[int]]], features_map: dict[int, list[tuple[int, int]]], feature_threshold: float) -> list[tuple[list[tuple[int, int]], list[int]]]:
        print("Combining segments")
        done_segments: dict[tuple[int, int], bool] = {}
        
        c_segments: dict[tuple[int, int], list[tuple[int, int]]] = {}
        combined_segments: list[tuple[list[tuple[int, int]], list[int]]] = []

        # create segments queue
        segment_queue: list[tuple[int, int]] = []
        for image_key in segments_map:
            for segment_key in segments_map[image_key]:
                segment_queue.append((image_key, segment_key))
        print(f"Initial segment queue lenght: {len(segment_queue)}")

        while len(segment_queue) > 0:
            image_key, segment_key = segment_queue.pop()        # pick a segment
            if (image_key, segment_key) in done_segments:
                continue
            if (image_key, segment_key) not in c_segments:
                c_segments[(image_key, segment_key)] = [(image_key, segment_key)]

            other_segments: dict[tuple[int, int], int] = {}
            for pid in segments_map[image_key][segment_key]:    # for every feature
                p_segments = features_map[pid]
                for p_segment in p_segments:                    # go over every segment that feature is in
                    if p_segment in done_segments or \
                        p_segment == (image_key, segment_key):  # if the segment is done, skip it
                        continue
                    if p_segment in other_segments:             # if segment already seen before
                        other_segments[p_segment] += 1          # add point
                    else:                                       # if segment not seen before
                        other_segments[p_segment] = 1           # initialize segment
            
            added = False
            for segment in other_segments:                              # go over all other segments
                o_image_key, o_segment_key = segment
                other_percentage = float(other_segments[segment]) / len(segments_map[o_image_key][o_segment_key])
                percentage = float(other_segments[segment]) / len(segments_map[image_key][segment_key])
                if other_percentage >= feature_threshold or percentage >= feature_threshold: # check feature threshold
                    done_segments[segment] = True
                    added = True
                    pids = segments_map[image_key][segment_key]
                    o_pids = segments_map[o_image_key][o_segment_key]
                    n_pids = remove_duplicates(pids + o_pids)
                    segments_map[image_key][segment_key] = n_pids       # add points to segment
                    for pid in n_pids:                                  # add segment to points
                        features_map[pid].append((image_key, segment_key))
                        features_map[pid] = remove_duplicates(features_map[pid])
                    # store combined segments
                    new_segments = [segment]
                    if segment in c_segments:
                        new_segments = c_segments[segment]
                    c_segments[(image_key, segment_key)] = remove_duplicates(c_segments[(image_key, segment_key)] + new_segments)
            if added:
                segment_queue.append((image_key, segment_key))          # segment added to queue
            else:
                done_segments[(image_key, segment_key)] = True          # segment done
                print(f"done: {len(segments_map[image_key][segment_key])}")
                if len(segments_map[image_key][segment_key]) > 0:
                    combined_segments.append((c_segments[(image_key, segment_key)], segments_map[image_key][segment_key]))
                
        return combined_segments

    @staticmethod
    def region_mapping(mask: Tensor, device: torch.device) -> list[Tensor]:
        print("Extracting missing regions")

        neighbour_list = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]  # todo check positions, (0,0) should be top left

        height, width = mask.shape
        mask_list = mask.cpu()
        label_list = torch.zeros(height, width, dtype=torch.int32, device=mask_list.device)
        collision_list: list[tuple[int, int]] = []

        # assign initial labels
        label = 1
        for y in range(0, height):                          # go over all pixels
            for x in range(0, width):
                if not mask_list[y][x]:                     # if active pixel
                    neighbours = []
                    for ny, nx, in neighbour_list:          # check all neighbours
                        if y + ny >= 0 and 0 <= x + nx < width:
                            if not mask_list[y+ny, x+nx]:   # if neighbour active
                                neighbours.append(label_list[y+ny, x+nx].item())
                    neighbours = remove_duplicates(neighbours)
                    if len(neighbours) <= 0:                # new label
                        label_list[y, x] = label
                        label += 1
                    else:
                        label_list[y, x] = neighbours[0]    # reuse label
                        if len(neighbours) > 1:
                            for i in range(1, len(neighbours)):     # record collisions
                                collision_list.append((neighbours[0], neighbours[i]))

        collision_list = remove_duplicates(collision_list)

        # resolve collisions
        label_set: list[set[int]] = []
        label_dict: dict[int, int] = {}
        for i in range(1, label):
            label_set.append({i})
            label_dict[i] = i-1
        for ca, cb in collision_list:
            ra = label_dict[ca]
            rb = label_dict[cb]
            if ra != rb:
                label_set[ra] |= label_set[rb]
                for p in label_set[rb]:
                    label_dict[p] = ra
                label_set[rb] = set()

        # extract masks
        final_masks: list[Tensor] = []
        for l_set in label_set:
            if len(l_set) <= 0:
                continue
            mask_list = torch.zeros_like(mask_list, dtype=torch.bool, device=mask_list.device)
            for label in l_set:
                label_mask = label_list == label
                mask_list[label_mask] = True
            final_masks.append(mask_list.to(dtype=torch.bool, device=device))
        print(f"{len(final_masks)} missing regions extracted")
        return final_masks

    def add_descarded_segments(self, final_masks: dict[int, Tensor], discarded_masks: list[Tensor]) -> tuple[dict[int, Tensor], int]:
        overlaps: list[tuple[int, Tensor]] = []

        for dsegment in discarded_masks:
            max_overlap: Optional[tuple[int, int]] = None
            dsegment = _C.dilation(self.growing_kernel, dsegment)
            for segment_id in final_masks:
                segment = final_masks[segment_id]
                overlap = torch.logical_and(dsegment, segment)
                # todo change to percentage
                n_overlap = torch.count_nonzero(overlap).item()
                m_overlap = 0 if max_overlap is None else max_overlap[1]
                if n_overlap > m_overlap:
                    max_overlap = segment_id, m_overlap
            if max_overlap is not None:
                segment_id, _ = max_overlap
                overlaps.append((segment_id, dsegment))
        
        for overlap in overlaps:
            segment_id, dsegment = overlap
            segment = final_masks[segment_id]
            segment = torch.logical_or(segment, dsegment)
            final_masks[segment_id] = segment

        return final_masks, len(overlaps)


# deprecated
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
                
        

