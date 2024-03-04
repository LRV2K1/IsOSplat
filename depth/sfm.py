import shutil
from pathlib import Path
import os

import tyro
import numpy as np

import enlighten

import pycolmap
from pycolmap import logging
from PIL import Image
from .sfm_types import *

class SFM:
    def __init__(self, img_path: Path):
        self.img_path: Path = img_path
        self.sfm_path: Path = img_path / "sfm"
        self.db_path: Path = self.sfm_path / "database.db"

        self.reconstruction: pycolmap.Reconstruction = None

    def get_point_cloud(self) -> PointCloud:
        print("Finding point cloud")
        if self.reconstruction is None:
            print("Point cloud has been blown away")
            return []
        points = []
        for point3D_id, point3D in self.reconstruction.points3D.items():
            point = (point3D.xyz, point3D.color)
            points.append(point)
        return points

    def get_camera_data(self) -> dict[int, CameraData]:
        print("Acquiring cameras")
        if self.reconstruction is None:
            print("No camera data found")
            return {}
        cameras = {}
        for camera_id, camera in self.reconstruction.cameras.items():
            cam = (camera.width, camera.height,
                   camera.focal_length_x, camera.focal_length_y,
                   camera.principal_point_x, camera.principal_point_y)
            cameras[camera_id] = cam
            print(camera_id, camera)
        return cameras

    def get_image_data(self) -> list[ImageData]:
        print("Gathering image data")
        if self.reconstruction is None:
            print("No image data found")
            return []
        images = []
        for image_id, image in self.reconstruction.images.items():
            img = (image.name, image.camera_id, image.projection_center(), image.viewing_direction())
            images.append(img)
            print(image_id, image, f"p={image.projection_center()}", f"d={image.viewing_direction()}")
        return images

    def sfm(self, clean: bool = False):
        self._sfm(clean)

        try:
            self.reconstruction = pycolmap.Reconstruction(self.sfm_path / "0")
        except:
            raise Exception("SFM could not reconstruct the data")

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

    def get_image(self):
        print("Generating images")
        if self.reconstruction is None:
            print("No data found")
            return
        images = self.reconstruction.images.items()
        points = self.reconstruction.points3D.items()
        cameras = self.reconstruction.cameras.items()

        cam: pycolmap.Camera
        point: pycolmap.point3D
        img: pycolmap.Image

        p = []
        c = []

        for point3D_id, point3D in points:
            # print(point3D_id, point3D)
            point = point3D
            p.append(point.xyz)
            c.append(point.color)

        for camera_id, camera in cameras:
            cam = camera
            break

        for image_id, image in images:
            img = image
            # cam is voor camera transform data zoals focus, img is voor de wereld transformatie
            # uv = cam.img_from_cam(img.cam_from_world * p)
            points2 = img.points2D
            uv = [pon.xy for pon in points2]

            arr = np.ones((cam.height, cam.width, 3), dtype=np.uint8) * 255
            # print(len(uv))
            col = np.zeros(3, dtype=np.uint8)
            # print(col)
            for i in range(len(uv)):
                x, y = uv[i]
                # col = c[i]
                # col = np.zeros(3)

                ix = int(x)
                iy = int(y)

                if 0 <= ix < cam.width and 0 <= iy < cam.height:
                    arr[iy, ix] = col
                if 0 <= ix - 1 < cam.width and 0 <= iy < cam.height:
                    arr[iy, ix - 1] = col
                if 0 <= ix + 1 < cam.width and 0 <= iy < cam.height:
                    arr[iy, ix + 1] = col
                if 0 <= ix < cam.width and 0 <= iy - 1 < cam.height:
                    arr[iy - 1, ix] = col
                if 0 <= ix < cam.width and 0 <= iy + 1 < cam.height:
                    arr[iy + 1, ix] = col

                if 0 <= ix - 2 < cam.width and 0 <= iy < cam.height:
                    arr[iy, ix - 2] = col
                if 0 <= ix - 1 < cam.width and 0 <= iy - 1 < cam.height:
                    arr[iy - 1, ix - 1] = col
                if 0 <= ix < cam.width and 0 <= iy - 2 < cam.height:
                    arr[iy - 2, ix] = col
                if 0 <= ix + 1 < cam.width and 0 <= iy - 1 < cam.height:
                    arr[iy - 1, ix + 1] = col
                if 0 <= ix + 2 < cam.width and 0 <= iy < cam.height:
                    arr[iy, ix + 2] = col
                if 0 <= ix + 1 < cam.width and 0 <= iy + 1 < cam.height:
                    arr[iy + 1, ix + 1] = col
                if 0 <= ix < cam.width and 0 <= iy + 2 < cam.height:
                    arr[iy + 2, ix] = col
                if 0 <= ix - 1 < cam.width and 0 <= iy + 1 < cam.height:
                    arr[iy + 1, ix - 1] = col

            save_image = Image.fromarray(arr)
            save_image.save(f"{self.sfm_path}/{image.name}")


def main(
        img_path: Path,
        clean: bool = False
) -> None:
    sfm = SFM(img_path)
    sfm.sfm(clean)
    sfm.get_point_cloud()
    sfm.get_camera_data()
    sfm.get_image_data()
    sfm.get_image()


if __name__ == "__main__":
    tyro.cli(main)
