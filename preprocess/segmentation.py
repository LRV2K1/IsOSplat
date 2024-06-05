from pathlib import Path
import numpy as np
import os
from typing import Optional
import math

import torch
from torch import Tensor

from isosplat.utils import Data, DataList
import isosplat.cuda as _C
from utils.graphics_utils import BasicPointCloud
from preprocess.colmap_loader import Image, Camera
from preprocess.image_loader import image_path_to_tensor, save_img_from_tensor

def get_colors(device:torch.device) -> Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ],
        device=device
    )

# def get_filter(device:torch.device) -> Tensor:
#     filter = torch.tensor(
#         [
#             [0.105, 0.197, 0.287, 0.325, 0.287, 0.197, 0.105],
#             [0.197, 0.368, 0.535, 0.607, 0.535, 0.368, 0.197],
#             [0.287, 0.535, 0.779, 0.882, 0.779, 0.535, 0.287],
#             [0.325, 0.607, 0.882, 1.000, 0.882, 0.607, 0.325],
#             [0.287, 0.535, 0.779, 0.882, 0.779, 0.535, 0.287],
#             [0.197, 0.368, 0.535, 0.607, 0.535, 0.368, 0.197],
#             [0.105, 0.197, 0.287, 0.325, 0.287, 0.197, 0.105]
#         ],
#         device=device
#     )
#     return filter / 3

def gaussian(sigma, x, y):
    return math.e ** (-((x**2 + y**2)/(2*sigma**2)))

def get_filter(device:torch.device, sigma: float = 1.0) -> Tensor:
    filter = torch.zeros(5, 5, device=device)
    for x in range(filter.shape[0]):
        for y in range(filter.shape[1]):
            filter[x,y] = gaussian(sigma, x-2, y-2)
    return filter


def generate_segmentation_images(path: Path, name: str, width: int, height: int, segments: dict[int, Tensor], device: torch.device, indexed: bool = False, xys: Optional[Tensor] = None):
        image = torch.zeros(height, width, 3, device=device)
        edges = torch.zeros(height, width, 3, device=device)
        colors = get_colors(device)
        i = 0
        for segment in segments:
            if indexed:
                i = segment % colors.shape[0]
            image[segments[segment]] = colors[i] * 0.5
            new_edge = torch.logical_xor(segments[segment], _C.erosion(growing_kernel(device), segments[segment]))
            edges[new_edge] = colors[i]
            i = (i + 1) % colors.shape[0]

        edge_map = edges > 0
        image[edge_map] = edges[edge_map]

        if xys is not None:
            kernel = torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0]
                ],
                device=device
            )
            points = torch.zeros(height, width, device=device)
            points = _C.padd_features(kernel, points, xys)[:,:,None]
            points = points.expand(-1,-1,3)
            image[points > 0] = points[points > 0]

        save_img_from_tensor(image, path, name) 


def closing_kernel(device: torch.device) -> Tensor:
    return torch.tensor(
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


def growing_kernel(device: torch.device) -> Tensor:
    return torch.tensor(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        dtype=torch.bool,
        device=device
    )


def segment_object(
        segment_path: Path, 
        sfm_images: dict[int, Image], 
        sfm_cameras: dict[int, Camera], 
        device: torch.device,

        descarded_segments_th: float = 0.1,
        local_feature_th: float = 0.1) -> tuple[dict[int, dict[int, list[int]]], dict[int, list[tuple[int, int]]], dict[int, dict[int, Tensor]]]:
    print("Segmenting objects")

    features_map: dict[int, list[tuple[int, int]]] = {}
    segments_map: dict[int, dict[int, list[int]]] = {}
    segments_mask_map: dict[int, dict[int, Tensor]] = {}

#   for every image
    for sfm_image in sfm_images.values():
        name = sfm_image.name.split('.')[0]
        image_id = sfm_image.id
        print()
        print(f"Processing segments {name}")

        # get feature points
        ipids = sfm_image.point3D_ids
        ipids_mask = ipids >= 0
        ipids = ipids[ipids_mask]
        xys = torch.tensor(np.int32(sfm_image.xys)[ipids_mask.tolist()], device=device)

        camera = sfm_cameras[sfm_image.camera_id]
        total_mask = torch.zeros(camera.height, camera.width, dtype=torch.bool, device=device)

        segments: dict[int, Tensor] = {}

#       for every segment
        image_segment_path = segment_path / name
        for file in os.listdir(image_segment_path):
            filename = os.fsdecode(file)
            if not filename.endswith(".png"):
                continue
            segment_id = int(filename.split('.')[0])
            segment, _ = image_path_to_tensor(image_segment_path / filename, device)

#           add segment to combined segments
            segment_mask = segment[:,:,0] > 0
            segments[segment_id] = segment_mask
            total_mask[segment_mask] = True

        initial_seg = len(segments)
        print(f"{initial_seg}: segments extracted")

#       missing segments <- generate missing segments
        closed_total_mask = _C.closing(closing_kernel(device), total_mask)
        torch.cuda.synchronize()
        missing_segments = region_mapping(closed_total_mask, device)

        missing_seg = len(missing_segments)
        print(f"{missing_seg} missing regions extracted")

#       for every missing segment
        segment_id = 0
        for missing_segment in missing_segments:
            segment_id -= 1
            segments[segment_id] = missing_segment

        total_seg = len(segments)
        print(f"{total_seg}: combined segments ({missing_seg} + {initial_seg})")

        segment_dict, discarded_segments, local_features, n_segments = configure_segment(xys, segments, image_id, ipids, features_map)

        created_seg = len(segment_dict)
        discarded_seg = len(discarded_segments)
        print(f"{created_seg}/{total_seg}: segments with features found and created")
        print(f"{discarded_seg}/{total_seg}: segments discarded")

#       some way of adding segments without features
        added_seg = add_descarded_segments(segments, discarded_segments, descarded_segments_th, device)

        print(f"{added_seg}/{discarded_seg}: discarded segments added")

#       combine segments in image
        combined_seg = combine_image_segments(image_id, segments, segment_dict, local_features, features_map, local_feature_th)

        print(f"{combined_seg}: segments combined")
        print(f"{len(segment_dict)}: final total segments")

        segments_map[image_id] = segment_dict
        segments_mask_map[image_id] = segments

    print()

    return segments_map, features_map, segments_mask_map


def create_objects(
        segments_map: dict[int, dict[int, list[int]]], 
        features_map: dict[int, list[tuple[int, int]]], 
        segments_mask_map: dict[int, dict[int, Tensor]], 
        sfm_images: dict[int, Image], 
        sfm_cameras: dict[int, Camera],
        device: torch.device, 
        feature_th: float = 0.1,
        file_path: Optional[Path] = None
    ) -> tuple[dict[int, tuple[list[int], list[tuple[int, int]]]], dict[int, dict[int, Tensor]], int, int, int]:
    objects_map_1, objects_feature_map_1 = create_objects_1(segments_map, features_map, feature_th)
    objects_map_2, objects_feature_map_2 = create_objects_2(segments_map, features_map, feature_th)
    objects_map_3, objects_feature_map_3 = create_objects_3(segments_map, features_map, feature_th)
    
    print(f"{len(objects_map_1)}: objects_1 created")
    print(f"{len(objects_map_2)}: objects_2 created")
    print(f"{len(objects_map_3)}: objects_3 created")
    print()

    final_obj_segments_dict: dict[int, dict[int, Tensor]] = {}
    final_objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}

    # final images
    for i in range(3):
        if i == 0:
            c_objects_map = objects_map_1
        elif i == 1:
            c_objects_map = objects_map_2
        else:
            c_objects_map = objects_map_3

        objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}
        obj_segments_dict: dict[int, dict[int, Tensor]] = {}
        obj_i = 0
        for obj_id in c_objects_map:
            _, obj_seg = c_objects_map[obj_id]
            objects_map[obj_i] = c_objects_map[obj_id]
            for img, seg in obj_seg:
                if img not in obj_segments_dict:
                    obj_segments_dict[img] = {}
                if obj_i not in obj_segments_dict[img]:
                    obj_segments_dict[img][obj_i] = segments_mask_map[img][seg]
                obj_segments_dict[img][obj_i] = torch.logical_or(obj_segments_dict[img][obj_i], segments_mask_map[img][seg])
            obj_i += 1
        if i == 0:
            final_objects_map = objects_map
            final_obj_segments_dict = obj_segments_dict
        if file_path is not None:
            img_path = file_path / f"object_{i+1}"
            for img in obj_segments_dict:
                camera = sfm_cameras[sfm_images[img].camera_id]
                generate_segmentation_images(img_path, f"{img}", camera.width, camera.height, obj_segments_dict[img], device, True)

    return final_objects_map, final_obj_segments_dict, len(objects_map_1), len(objects_map_2), len(objects_map_3)


def select_object(
        objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]],
        obj_segments_dict: dict[int, dict[int, Tensor]],
        sfm_images: dict[int, Image], 
        sfm_cameras: dict[int, Camera],
        device: torch.device
    ) -> tuple[dict[int, Tensor], list[int]]:


    points = []
    img_masks = {}

    gathered_objects = set()
    images = set()

    # get highest scoring obj
    while len(images) < len(obj_segments_dict):     # as long as not all images have a segment
        centre_scores = {}
        # calculate scores
        for img_id in obj_segments_dict:
            if img_id in images:
                continue
            for obj_id in obj_segments_dict[img_id]:
                if obj_id in gathered_objects:
                    continue
                camera = sfm_cameras[sfm_images[img_id].camera_id]
                p_x, p_y = camera.width / 2, camera.height / 2
                mask = obj_segments_dict[img_id][obj_id]
                if mask[int(p_y),int(p_x)].item():
                    if obj_id not in centre_scores:
                        centre_scores[obj_id] = 0
                    centre_scores[obj_id] += 1

        # get highest score
        h_obj = 0
        h_score = 0
        for obj_id in centre_scores:
            if centre_scores[obj_id] > h_score:
                h_obj = obj_id
                h_score = centre_scores[obj_id]

        # add highest rated obj
        gathered_objects.add(h_obj)
        obj_points, _ = objects_map[h_obj]
        points = remove_duplicates(points + obj_points)

        # get size
        c_size = 0
        size_scores = {}
        for img_id in obj_segments_dict:
            if h_obj in obj_segments_dict[img_id]:
                mask = obj_segments_dict[img_id][h_obj]
                size = mask.count_nonzero()
                size_scores[img_id] = size
                c_size += size
        f_size = c_size / len(size_scores)       

        # record images
        for img_id in obj_segments_dict:
            if img_id in images:
                continue
            if h_obj in obj_segments_dict[img_id]:
                if size_scores[img_id] > f_size / 2:
                    images.add(img_id)

    # get masks    
    filter = get_filter(device)
    for img_id in obj_segments_dict:
        for obj_id in gathered_objects:
            if obj_id not in obj_segments_dict[img_id]:
                continue
            if img_id not in img_masks:
                img_masks[img_id] = torch.zeros_like(mask, device=device)
            mask = obj_segments_dict[img_id][obj_id]
            img_masks[img_id] = torch.logical_or(img_masks[img_id], mask)
        img_masks[img_id] = _C.filter(filter, img_masks[img_id])
        image = img_masks[img_id]
        save_img_from_tensor(image, f"masks", f"{img_id}")    

    return img_masks, points


def region_mapping(mask: Tensor, device: torch.device) -> list[Tensor]:
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
    return final_masks


def configure_segment(
    xys: Tensor, 
    masks: dict[int, Tensor], 
    image_id: int, 
    ipids: np.ndarray, 
    features_map: dict[int, list[tuple[int, int]]]
) -> tuple[dict[int, list[int]], dict[int, Tensor], dict[int, list[int]], int]:
    n_segments = 0
    segment_dict = {}
    discarded_segments = {}
    local_features = {}
    
    for segment_id in masks:
        mask = masks[segment_id]
        xy_mask = _C.extract_segment_features(mask, xys)
        segment_ipids = ipids[xy_mask.cpu()]

        # if segment_id == 1:
        #     img = mask[:,:,None].repeat(1,1,3)
        #     red = torch.tensor([1.0,0.0,0.0], device=img.device)
        #     green = torch.tensor([0.0,1.0,0.0], device=img.device)

        #     print(xys.shape)
        #     xy_list = xys.tolist()
        #     for x, y in xy_list:
        #         if y < img.shape[0] and x < img.shape[1]:
        #             img[int(y), int(x)] = green

        #     xy_list2 = xys[xy_mask.cpu()].tolist()
        #     for x, y in xy_list2:
        #         if y < img.shape[0] and x < img.shape[1]:
        #             img[int(y), int(x)] = red

        #     save_img_from_tensor(img, "masks_1", f"mask{segment_id}")

        #     raise Exception("test")
    

        feature_list = []
        for pid in segment_ipids:
            feature_list.append(pid)
            if pid not in features_map:
                features_map[pid] = []
            features_map[pid].append((image_id, segment_id))
            features_map[pid] = remove_duplicates(features_map[pid])    # should not be needed
            if pid not in local_features:
                local_features[pid] = []
            local_features[pid].append(segment_id)
            local_features[pid] = remove_duplicates(local_features[pid])    # should not be needed
        if len(feature_list) > 0:
            n_segments += 1
            segment_dict[segment_id] = remove_duplicates(feature_list)
        else:
            # masks.pop(segment_id)
            discarded_segments[segment_id] = mask
    
    for segment_id in discarded_segments:
        masks.pop(segment_id)
    return segment_dict, discarded_segments, local_features, n_segments


def add_descarded_segments(segments: dict[int, Tensor], discarded_segments: list[int], threshold: float, device: torch.device) -> int:
    overlaps: list[tuple[int, Tensor]] = []

    for dsegment_id in discarded_segments:
        max_overlap: Optional[tuple[int, int]] = None
        dsegment = _C.dilation(growing_kernel(device), discarded_segments[dsegment_id])
        n_dsegment = torch.count_nonzero(dsegment).item()
        for segment_id in segments:
            segment = segments[segment_id]
            overlap = torch.logical_and(dsegment, segment)
            n_overlap = torch.count_nonzero(overlap).item()
            m_overlap = 0 if max_overlap is None else max_overlap[1]
            if n_overlap > m_overlap:
                max_overlap = segment_id, m_overlap
        if max_overlap is not None and float(max_overlap[1])/n_dsegment >= threshold:
            segment_id, _ = max_overlap
            overlaps.append((segment_id, dsegment))
    
    # add overlaps
    for overlap in overlaps:
        segment_id, dsegment = overlap
        segment = segments[segment_id]
        segments[segment_id] = torch.logical_or(segment, dsegment)

    return len(overlaps)


def combine_image_segments(
        img_id: int,
        segments: dict[int, Tensor], 
        segment_dict: dict[int, list[int]], 
        local_features: dict[int, list[int]], 
        features_map: dict[int, list[tuple[int, int]]], 
        feature_th: float) -> int:
    checked_segments = set()
    combined_segments = set()

    # determine order
    segment_queue = []
    for segment in segment_dict:
        segment_queue.append((len(segment_dict[segment]), segment))
    segment_queue.sort()    # largest first

    # while not all segments done:
    while len(segment_queue) > 0:
        # segment_queue.sort()
    #   pick first segment
        size_1, seg_id_1 = segment_queue.pop()
        if seg_id_1 in combined_segments:    # should not be in checked, because those should not be in the queue
            continue
        features_1 = segment_dict[seg_id_1]

    #   count matching features
        matching_features = {}
        for feature in features_1:
            for seg_id_2 in local_features[feature]:
                if seg_id_1 == seg_id_2 or seg_id_2 in checked_segments:    # should not be in combined, because when combined, they should be removed from the feature list
                    continue
                if seg_id_2 not in matching_features:
                    matching_features[seg_id_2] = 0
                matching_features[seg_id_2] += 1

    #   calculate matching feature score
        matching_feature_scores = []
        for seg_id_2 in matching_features:
            size_2 = len(segment_dict[seg_id_2])
            size_o = float(matching_features[seg_id_2])
            # intersection over union
            i_score = size_o / ((size_1 + size_2) - size_o)
            # new score
            n_score = ((size_o / size_1) ** 2) + ((size_o /  size_2) ** 2)
            matching_feature_scores.append((n_score, seg_id_2))
        matching_feature_scores.sort()

    #   if highest score >= threshold:
        if len(matching_feature_scores) > 0 and matching_feature_scores[-1][0] >= feature_th:
    #       combine segments
            score, seg_id_2 = matching_feature_scores[-1]
            features_2 = segment_dict[seg_id_2]
            for feature in features_2:
                local_features[feature].remove(seg_id_2)
                if seg_id_1 not in local_features[feature]:
                    local_features[feature].append(seg_id_1)
            segments[seg_id_1] = torch.logical_or(segments[seg_id_1], segments.pop(seg_id_2))
            combine_segments(img_id, seg_id_1, seg_id_2, segment_dict, features_map)

            segment_queue.append((len(segment_dict[seg_id_1]), seg_id_1))   # updated, put back in queue
            combined_segments.add(seg_id_2)

    #   else: segment done
        else:
            checked_segments.add(seg_id_1)

    return len(combined_segments)


# destroys segment_2
def combine_segments(img_id: int, seg_id_1: int, seg_id_2: int, segment_dict: dict[int, list[int]], features_map: dict[int, list[tuple[int, int]]]):
    features_1 = segment_dict[seg_id_1]
    features_2 = segment_dict.pop(seg_id_2)
    segment_dict[seg_id_1] = remove_duplicates(features_1 + features_2)

    for feature in features_2:
        features_map[feature].remove((img_id, seg_id_2))
        if (img_id, seg_id_1) not in features_map[feature]:
            features_map[feature].append((img_id, seg_id_1))


def create_objects_1(
        segments_map: dict[int, dict[int, list[int]]], 
        features_map: dict[int, list[tuple[int, int]]], 
        feature_th: float
        ) -> tuple[dict[int, tuple[list[int], list[tuple[int, int]]]], dict[int, list[int]]]:
    # create objects
    object_segment_map: dict[tuple[int, int], int] = {}
    object_features_map: dict[int, list[int]] = {}
    objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}

    obj_i = 0
    for image in segments_map:
        for segment in segments_map[image]:
            object_segment_map[(image, segment)] = obj_i
            objects_map[obj_i] = (segments_map[image][segment], [(image, segment)])
            obj_i += 1
    for feature in features_map:
        object_list = []
        for image_segment in features_map[feature]:
            object_list.append(object_segment_map[image_segment])
        object_features_map[feature] = object_list

    checked_segments = set()
    combined_segments = set()

    # determine order
    objects_queue: list[tuple[int, int, int, int]] = []
    for obj_id in objects_map:
        objects_queue.append((len(objects_map[obj_id][0]), obj_id))
    objects_queue.sort()

    # while not all segments done
    obj_i = 0
    while len(objects_queue) > 0:
        # segment_queue.sort()
    #   pick first object
        size_1, obj_id_1 = objects_queue.pop()
        if obj_id_1 in combined_segments:
            continue
        features_1, segments_1 = objects_map[obj_id_1]

    #   count matching features
        matching_features = {}
        for feature in features_1:
            for obj_id_2 in object_features_map[feature]:
                if obj_id_1 == obj_id_2 or obj_id_2 in checked_segments:    # should not be in combined, because when combined, they should be removed from the feature list
                    continue
                if obj_id_2 not in matching_features:
                    matching_features[obj_id_2] = 0
                matching_features[obj_id_2] += 1

    #   calculate matching feature score
        matching_feature_scores = []
        for obj_id_2 in matching_features:
            size_2 = len(objects_map[obj_id_2][0])
            size_o = float(matching_features[obj_id_2])
            # intersection over union
            i_score = size_o / ((size_1 + size_2) - size_o)
            # new score
            n_score = ((size_o / size_1) ** 2) + ((size_o /  size_2) ** 2)
            matching_feature_scores.append((n_score, obj_id_2))
        matching_feature_scores.sort()

    #   if highest score >= threshold:
        if len(matching_feature_scores) > 0 and matching_feature_scores[-1][0] >= feature_th:
    #       combine objects
            score, obj_id_2 = matching_feature_scores[-1]
            features_2, segments_2 = objects_map.pop(obj_id_2)
            for feature in features_2:
                object_features_map[feature].remove(obj_id_2)
                if obj_id_1 not in object_features_map[feature]:
                    object_features_map[feature].append(obj_id_1)
            objects_map[obj_id_1] = (remove_duplicates(features_1 + features_2), segments_1 + segments_2)

            objects_queue.append((len(objects_map[obj_id_1][0]), obj_id_1))   # updated, put back in queue
            combined_segments.add(obj_id_2)

    #   else: segment done
        else:
            checked_segments.add(obj_id_1)

    return objects_map, object_features_map


def create_objects_2(
        segments_map: dict[int, dict[int, list[int]]], 
        features_map: dict[int, list[tuple[int, int]]], 
        feature_th: float
        ) -> tuple[dict[int, tuple[list[int], list[tuple[int, int]]]], dict[int, list[int]]]:
    # create objects
    object_segment_map: dict[tuple[int, int], int] = {}
    object_features_map: dict[int, list[int]] = {}
    objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}

    obj_i = 0
    for image in segments_map:
        for segment in segments_map[image]:
            object_segment_map[(image, segment)] = obj_i
            objects_map[obj_i] = (segments_map[image][segment], [(image, segment)])
            obj_i += 1
    for feature in features_map:
        object_list = []
        for image_segment in features_map[feature]:
            object_list.append(object_segment_map[image_segment])
        object_features_map[feature] = object_list

    checked_segments = set()

    # determine order
    objects_queue: list[tuple[int, int]] = []
    for obj_id in objects_map:
        features, segments = objects_map[obj_id]
        objects_queue.append((len(features), segments[0][0], segments[0][1]))
    objects_queue.sort()

    # while not all segments done
    obj_i = 0
    while len(objects_queue) > 0:
        # segment_queue.sort()
    #   pick first object
        sort_score, img_id_1, seg_id_1 = objects_queue.pop()
        obj_id_1 = object_segment_map[(img_id_1, seg_id_1)]
        features_1 = segments_map[img_id_1][seg_id_1]
        size_1 = len(features_1)

    #   count matching features
        matching_features = {}
        for feature in features_1:
            for img_id_2, seg_id_2 in features_map[feature]:
                obj_id_2 = object_segment_map[(img_id_2, seg_id_2)]
                if img_id_1 == img_id_2 or obj_id_1 == obj_id_2 or (img_id_2, seg_id_2) in checked_segments:    # should not be in combined, because when combined, they should be removed from the feature list
                    continue
                if (img_id_2, seg_id_2) not in matching_features:
                    matching_features[(img_id_2, seg_id_2)] = 0
                matching_features[(img_id_2, seg_id_2)] += 1

    #   calculate matching feature score
        matching_feature_scores = []
        for (img_id_2, seg_id_2) in matching_features:
            size_2 = len(segments_map[img_id_2][seg_id_2])
            size_o = float(matching_features[(img_id_2, seg_id_2)])
            # intersection over union
            i_score = size_o / ((size_1 + size_2) - size_o)
            # new score
            n_score = ((size_o / size_1) ** 2) + ((size_o /  size_2) ** 2)
            matching_feature_scores.append((n_score, (img_id_2, seg_id_2)))
        matching_feature_scores.sort()

    #   if highest score >= threshold:
        if len(matching_feature_scores) > 0 and matching_feature_scores[-1][0] >= feature_th:
    #       combine objects
            score, (img_id_2, seg_id_2) = matching_feature_scores[-1]
            obj_id_2 = object_segment_map[(img_id_2, seg_id_2)]
            o_features_1, o_segments_1 = objects_map[obj_id_1]
            o_features_2, o_segments_2 = objects_map.pop(obj_id_2)
            for feature in o_features_2:
                object_features_map[feature].remove(obj_id_2)
                if obj_id_1 not in object_features_map[feature]:
                    object_features_map[feature].append(obj_id_1)
            objects_map[obj_id_1] = (remove_duplicates(o_features_1 + o_features_2), o_segments_1 + o_segments_2)

            objects_queue.append((sort_score, img_id_1, seg_id_1))   # updated, put back in queue
            for (img_id, seg_id) in o_segments_2:
                object_segment_map[(img_id, seg_id)] = obj_id_1

    #   else: segment done
        else:
            checked_segments.add(obj_id_1)

    return objects_map, object_features_map


def create_objects_3(
        segments_map: dict[int, dict[int, list[int]]], 
        features_map: dict[int, list[tuple[int, int]]], 
        feature_th: float
        ) -> tuple[dict[int, tuple[list[int], list[tuple[int, int]]]], dict[int, list[int]]]:
    # create objects
    object_features_map: dict[int, list[int]] = {}
    objects_map: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}

    finished_segments = set()
    combined_segments = {}

    # determine order
    objects_queue: list[tuple[int, int]] = []
    for img_id in segments_map:
        for seg_id in segments_map[img_id]:
            features = segments_map[img_id][seg_id]
            objects_queue.append((len(features), img_id, seg_id))
    objects_queue.sort()

    # while not all segments done
    obj_i = 0
    while len(objects_queue) > 0:
        # segment_queue.sort()
    #   pick first object
        size_1, img_id_1, seg_id_1 = objects_queue.pop()
        if (img_id_1, seg_id_1) in finished_segments:
            continue
        if (img_id_1, seg_id_1) not in combined_segments:
            combined_segments[(img_id_1, seg_id_1)] = obj_i
            objects_map[obj_i] = (segments_map[img_id_1][seg_id_1], [(img_id_1, seg_id_1)])
            features_1 = segments_map[img_id_1][seg_id_1]
            for feature in features_1:
                if feature not in object_features_map:
                    object_features_map[feature] = []
                object_features_map[feature].append(obj_i)
            obj_i += 1
        obj_id_1 = combined_segments[(img_id_1, seg_id_1)]
        features_1 = segments_map[img_id_1][seg_id_1]

    #   count matching features
        matching_features = {}
        for feature in features_1:
            for img_id_2, seg_id_2 in features_map[feature]:
                if (img_id_2, seg_id_2) in combined_segments:
                    continue
                if (img_id_2, seg_id_2) not in matching_features:
                    matching_features[(img_id_2, seg_id_2)] = 0
                matching_features[(img_id_2, seg_id_2)] += 1

    #   calculate matching feature score
        matching_feature_scores = []
        for (img_id_2, seg_id_2) in matching_features:
            size_2 = len(segments_map[img_id_2][seg_id_2])
            size_o = float(matching_features[(img_id_2, seg_id_2)])
            # intersection over union
            i_score = size_o / ((size_1 + size_2) - size_o)
            # new score
            n_score = ((size_o / size_1) ** 2) + ((size_o /  size_2) ** 2)
            matching_feature_scores.append((n_score, (img_id_2, seg_id_2)))
        matching_feature_scores.sort()

    #   if highest score >= threshold:
        if len(matching_feature_scores) > 0 and matching_feature_scores[-1][0] >= feature_th:
    #       combine objects
            score, (img_id_2, seg_id_2) = matching_feature_scores[-1]
            o_features_1, o_segments_1 = objects_map[obj_id_1]
            features_2 = segments_map[img_id_2][seg_id_2]
            o_segments_1.append((img_id_2, seg_id_2))
            objects_map[obj_id_1] = (remove_duplicates(o_features_1 + features_2), o_segments_1)
            for feature in features_2:
                if feature not in object_features_map:
                    object_features_map[feature] = []
                if obj_id_1 not in object_features_map[feature]:
                    object_features_map[feature].append(obj_id_1)

            combined_segments[(img_id_2, seg_id_2)] = obj_id_1
            new_queue = []
            for img, seg in o_segments_1:
                size = len(segments_map[img][seg])
                new_queue.append((size, img, seg))
            new_queue.sort()
            objects_queue = objects_queue + new_queue

    #   else: segment done
        else:
            finished_segments.add((img_id_1, seg_id_1))

    return objects_map, object_features_map


def remove_duplicates(l: list[any]) -> list[any]:
    return list(dict.fromkeys(l))
