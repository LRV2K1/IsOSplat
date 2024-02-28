import shutil
import urllib.request
import zipfile
from pathlib import Path
# from typing import Optional, NewType
import os

import tyro

import enlighten

import pycolmap
from pycolmap import logging

# from .types import PointCloud

class SFM():
    def __init__(self, img_path: Path):
        self.img_path: Path = img_path
        self.sfm_path: Path = img_path / "sfm"
        self.db_path: Path = self.sfm_path / "database.db"

        self.reconstruction: pycolmap.Reconstruction

    # def get_point_cloud(self) -> PointCloud:
    #     print("pc")

    def sfm(self, clean: bool = False):
        self._sfm(clean)

        try:
            self.reconstruction = pycolmap.Reconstruction(self.sfm_path / "0")
        except:
            raise Exception("SFM could not reconstruct the data")
        for image_id, image in self.reconstruction.images.items():
            print(image_id, image, image.cam_from_world)

        # for point3D_id, point3D in reconstruction.points3D.items():
        #     print(point3D_id, point3D)

        for camera_id, camera in self.reconstruction.cameras.items():
            print(camera_id, camera)


    def _sfm(self, clean: bool = False):
        if not os.path.exists(self.sfm_path):
            os.makedirs(self.sfm_path)
        if self.db_path.exists() and not clean:
            return
        
        logging.set_log_destination(logging.INFO, self.sfm_path / "INFO.log.")  # + time
        
        if self.db_path.exists():
            self.db_path.unlink()
        if self.sfm_path.exists():
            shutil.rmtree(self.sfm_path)
        self.sfm_path.mkdir(exist_ok=True)

        pycolmap.extract_features(self.db_path, self.img_path)
        pycolmap.match_exhaustive(self.db_path)
        num_images = pycolmap.Database(self.db_path).num_images

        with enlighten.Manager() as manager:
            with manager.counter(total=num_images, desc="Images registered:") as pbar:
                pbar.update(0, force=True)
                recs = pycolmap.incremental_mapping(
                    self.db_path,
                    self.img_path,
                    self.sfm_path,
                    initial_image_pair_callback=lambda: pbar.update(2),
                    next_image_callback=lambda: pbar.update(1),
                )
        for idx, rec in recs.items():
            logging.info(f"#{idx} {rec.summary()}")


def main(
        img_path: Path,
        clean: bool = False
) -> None:
    sfm = SFM(img_path)
    sfm.sfm(clean)

if __name__ == "__main__":
    tyro.cli(main)
