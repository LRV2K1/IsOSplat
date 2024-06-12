from pathlib import Path
from typing import Optional
import torch
import csv
import os


from preprocess.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from preprocess.preprocessor import segment_object, create_objects, select_object


iterations = 2
data = ["fountainD", "fernD", "roomD", "fortressD", "leavesD", "flowerD", "orchidsD", "trexD", "hornsD"]


def main():
    device = torch.device("cuda:0")

    g_record_path = Path(f"train_segments")
    
    if not os.path.exists(g_record_path):
        os.makedirs(g_record_path)

    with open(g_record_path / "g_records.csv", "a", newline='') as record_file:
        writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["name", "score", "initial_seg", "missing_seg", "non_cl_missing_seg", "total_seg", "created_seg", "discarded_seg", "added_seg", "non_cl_added_seg", "combined_seg", "final_seg"])

    a_score = False
    for i in range(2):
        score_type = "IoU"
        if not a_score:
            score_type = "SQI"
        for d in data:
            print(d)

            sfm_path = Path(f"train_data/{d}/sparse/0")
            segment_path = Path(f"train_data/{d}/segments")
            record_path = Path(f"train_segments/{d}-{score_type}")
            mask_save_path = record_path / "masks"
            obj_save_path = record_path/ "objects"

            if not os.path.exists(record_path):
                os.makedirs(record_path)
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            if not os.path.exists(obj_save_path):
                os.makedirs(obj_save_path)


            with open(record_path / "records.csv", "a", newline='') as record_file:
                writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["n_objects_1", "n_objects_2", "s_objects_1", "s_objects_2", "threshold", "iteration"])

            with open(record_path / "s_records.csv", "a", newline='') as record_file:
                writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["img_id", "img_name", "initial_seg", "missing_seg", "non_cl_missing_seg", "total_seg", "created_seg", "discarded_seg", "added_seg", "non_cl_added_seg", "combined_seg", "final_seg"])

            sfm_images = read_extrinsics_binary(sfm_path / "images.bin")
            sfm_cameras = read_intrinsics_binary(sfm_path / "cameras.bin")
            segments_map, features_map, segments_mask_map, seg_data = segment_object(segment_path, sfm_images, sfm_cameras, device, 0.0, 0.1, a_score, mask_save_path)

            t_initial_seg = 0
            t_missing_seg = 0
            t_non_cl_missing_seg = 0
            t_total_seg = 0
            t_created_seg = 0
            t_discarded_seg = 0
            t_added_seg = 0
            t_non_cl_added_seg = 0
            t_combined_seg = 0
            t_final_seg = 0

            with open(record_path / "s_records.csv", "a", newline='') as record_file:
                writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for  (name, img_id, initial_seg, missing_seg, non_cl_missing_seg, total_seg, created_seg, discarded_seg, added_seg, non_cl_added_seg, combined_seg, final_seg) in seg_data:
                    writer.writerow([img_id, name, initial_seg, missing_seg, non_cl_missing_seg, total_seg, created_seg, discarded_seg, added_seg, non_cl_added_seg, combined_seg, final_seg])
                    t_initial_seg += initial_seg
                    t_missing_seg += missing_seg
                    t_non_cl_missing_seg += non_cl_missing_seg
                    t_total_seg += total_seg
                    t_created_seg += created_seg
                    t_discarded_seg += discarded_seg
                    t_added_seg += added_seg
                    t_non_cl_added_seg += non_cl_added_seg
                    t_combined_seg += combined_seg
                    t_final_seg += final_seg

            with open(g_record_path / "g_records.csv", "a", newline='') as record_file:
                writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([d, score_type, t_initial_seg, t_missing_seg, t_non_cl_missing_seg, t_total_seg, t_created_seg, t_discarded_seg, t_added_seg, t_non_cl_added_seg, t_combined_seg, t_final_seg])

            # [n_objects, threshold, iteration]
            gathered_data_1: list[tuple[int, float, int]] = []
            gathered_data_2: list[tuple[int, float, int]] = []
            
            threshold = 0.0
            max_threshold = 1.0
            if not a_score:
                max_threshold = 2.0

            for i in range(iterations + 1):
                threshold =  (max_threshold / iterations) * i
                # if len(gathered_data_1) >= 2:
                #     g_data = gathered_data_1
                #     # if i % 3 == 1:
                #     #     g_data = gathered_data_2
                #     # elif i % 3 == 2:
                #     #     g_data = gathered_data_3

                #     max_distance = 0
                #     g_data.sort()
                #     # print(g_data)
                #     distances = []
                #     for j in range(len(g_data) - 1):
                #         n_1, th_1, i_1 = g_data[j]
                #         n_2, th_2, i_2 = g_data[j+1]
                #         distances.append(n_2 - n_1)
                #         if n_2 - n_1 > max_distance:
                #             max_distance = n_2 - n_1
                #             threshold = (th_1 + th_2) * 0.5
                #     # print(distances)
                #     # print(max_distance, threshold)

                print(threshold)
                image_path = obj_save_path / f"{i}_{threshold}"
                image_path2 = obj_save_path / f"{i}_{threshold}_selected"
                image_path3 = obj_save_path / f"{i}_{threshold}_final"

                om_1, os_1, om_2, os_2, om_3, os_3, no_1, no_2, no_3 = create_objects(segments_map, features_map, segments_mask_map, sfm_images, sfm_cameras, device, threshold, a_score, image_path)

                _, _, ob_1 = select_object(om_1, os_1, sfm_images, sfm_cameras, device, image_path2 / "objects_1", image_path3 / "objects_1")
                print(f"{ob_1}: objects selected")
                _, _, ob_2 = select_object(om_2, os_2, sfm_images, sfm_cameras, device, image_path2 / "objects_2", image_path3 / "objects_2")
                print(f"{ob_2}: objects selected")
                # _, _, ob_3 = select_object(om_3, os_3, sfm_images, sfm_cameras, device, image_path2 / "objects_3", image_path3 / "objects_3")
                # print(f"{ob_3}: objects selected")
                print()

                with open(record_path / "records.csv", "a", newline='') as record_file:
                    writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([no_1, no_2, ob_1, ob_2, threshold, i])

        a_score = True

if __name__ == '__main__':
    main()
