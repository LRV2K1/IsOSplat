from pathlib import Path
from typing import Optional
import torch
import csv
import os


from preprocess.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from preprocess.preprocessor import segment_object, create_objects


iterations = 20
data = ["fountainD"]#, "fern", "horns"]


def main():
    device = torch.device("cuda:0")

    for d in data:
        print(d)
        sfm_path = Path(f"data/{d}/sparse/0")
        segment_path = Path(f"data/{d}/segments")
        record_path = Path(f"segments_data_2/{d}")

        if not os.path.exists(record_path):
            os.makedirs(record_path)

        with open(record_path / "records.csv", "a", newline='') as record_file:
            writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["data_1", "data_2", "data_3", "", ""])
            writer.writerow(["n_objects_1", "n_objects_2", "n_objects_3", "threshold", "iteration"])

        sfm_images = read_extrinsics_binary(sfm_path / "images.bin")
        sfm_cameras = read_intrinsics_binary(sfm_path / "cameras.bin")
        segments_map, features_map, segments_mask_map = segment_object(segment_path, sfm_images, sfm_cameras, device, 0.0, 0.1)

        # [n_objects, threshold, iteration]
        gathered_data_1: list[tuple[int, float, int]] = []
        gathered_data_2: list[tuple[int, float, int]] = []
        gathered_data_3: list[tuple[int, float, int]] = []
        
        threshold = 2.0
        for i in range(iterations):
            if len(gathered_data_1) >= 2:
                g_data = gathered_data_1
                # if i % 3 == 1:
                #     g_data = gathered_data_2
                # elif i % 3 == 2:
                #     g_data = gathered_data_3

                max_distance = 0
                g_data.sort()
                print(g_data)
                distances = []
                for j in range(len(g_data) - 1):
                    n_1, th_1, i_1 = g_data[j]
                    n_2, th_2, i_2 = g_data[j+1]
                    distances.append(n_2 - n_1)
                    if n_2 - n_1 > max_distance:
                        max_distance = n_2 - n_1
                        threshold = (th_1 + th_2) * 0.5
                print(distances)
                print(max_distance, threshold)

            print(threshold)
            image_path = record_path / f"{threshold}_{i}"

            no_1, no_2, no_3 = create_objects(segments_map, features_map, segments_mask_map, sfm_images, sfm_cameras, device, threshold, image_path)
            gathered_data_1.append((no_1, threshold, i))
            gathered_data_2.append((no_2, threshold, i))
            gathered_data_3.append((no_3, threshold, i))

            
            with open(record_path / "records.csv", "a", newline='') as record_file:
                writer = csv.writer(record_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([no_1, no_2, no_3, threshold, i])

            threshold = 0.0

if __name__ == '__main__':
    main()
