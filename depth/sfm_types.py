from typing import NewType


PointCloud = NewType('PointCloud', list[tuple[tuple[float, float, float], tuple[int, int, int]]])

CameraData = NewType('CameraData', tuple[int, int, float, float, float, float])
ImageData = NewType('ImageData', tuple[str, int, tuple[float, float, float], tuple[float, float, float]])
