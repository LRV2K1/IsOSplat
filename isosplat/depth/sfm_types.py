from typing import NewType


Point = NewType('Point', (float, float, float))
PointCloud = NewType('PointCloud', Point)

