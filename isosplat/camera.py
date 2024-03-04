import math
import torch
from torch import Tensor


class Camera:

    def __init__(self,
                 width: int, height: int,
                 focalx: float, focaly: float,
                 cx: float, cy: float,
                 # near: float, far: float,
                 device: torch.device):
        self.device = device

        self.width = width
        self.height = height

        self.focalx = focalx
        self.focaly = focaly

        self.cx = cx
        self.cy = cy

        fov_width = (2 * self.focalx) / width
        fov_height = (2 * self.focaly) / height
        far = 10
        near = 1
        a = -(far + near) / (far - near)
        b = -(2 * far * near) / (far - near)

        self.perspective_project_mat = torch.tensor(
            [
                [fov_width, 0.0, 0.0, 0.0],
                [0.0, fov_height, 0.0, 0.0],
                [0.0, 0.0, a, b],
                [0.0, 0.0, -1.0, 0.0]
            ],
            device=self.device
        )

        self.x = 0.0
        self.y = 0.0
        self.z = 8.0

        self.look_at(0.0, 0.0, 0.0)

        self.viewMatrixUpdate = True

        self.rotation_mat.requires_grad = False
        self.perspective_project_mat.requires_grad = False

        self.project_matrix: Tensor
        self.view_matrix: Tensor

    def set_perspective_matrix(self, mat: Tensor):
        self.perspective_project_mat = mat
        self.perspective_project_mat.requires_grad = False

        self.viewMatrixUpdate = True

    def _update_view_and_projection_matrix(self):
        if self.viewMatrixUpdate:
            translation_mat = torch.tensor(
                [
                    [1.0, 0.0, 0.0, -self.x],
                    [0.0, 1.0, 0.0, -self.y],
                    [0.0, 0.0, 1.0, -self.z],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                device=self.device
            )
            self.model_view_mat = torch.matmul(self.rotation_mat, translation_mat)
            self.model_view_mat.requires_grad = False

            self.project_matrix = torch.matmul(self.perspective_project_mat, self.model_view_mat)
            self.project_matrix.requires_grad = False

            self.viewMatrixUpdate = False

    def get_view_and_project_matrix(self) -> tuple[Tensor, Tensor]:
        self._update_view_and_projection_matrix()
        return self.model_view_mat, self.project_matrix

    def get_view_direction(self) -> Tensor:
        direction = torch.tensor(
            [self.rotation_mat[0, 2],
             self.rotation_mat[1, 2],
             self.rotation_mat[2, 2]],
            device=self.device
        )
        return direction

    def get_camera_position(self) -> Tensor:
        pos = torch.tensor(
            [
                self.x,
                self.y,
                self.z
            ],
            device=self.device
        )
        return pos

    def get_focal(self) -> tuple[float, float]:
        return self.focalx, self.focaly

    def get_principal(self) -> tuple[float, float]:
        return self.cx, self.cy

    def get_size(self) -> tuple[int, int]:
        return self.width, self.height

    def set_position(self, posx: float, posy: float, posz: float):
        self.x = posx
        self.y = posy
        self.z = posz
        self.viewMatrixUpdate = True

    def translate(self, posx: float, posy: float, posz: float):
        self.x += posx
        self.y += posy
        self.z += posz
        self.viewMatrixUpdate = True

    def set_view_direction(self, dirx: float, diry: float, dirz: float):
        self.look_at(self.x+dirx, self.y+diry, self.z+dirz)

    def distance(self, posx: float, posy: float, posz: float, dis: float):
        self._distance(self.x, self.y, self.z, posx, posy, posz, dis)

    def _distance(self, camx: float, camy: float, camz: float, posx: float, posy: float, posz: float, dis: float):
        dir = torch.tensor([camx - posx, camy - posy, camz - posz], device=self.device)
        dir = dir / torch.linalg.vector_norm(dir)

        self.x = posx + dir[0] * dis
        self.y = posy + dir[1] * dis
        self.z = posz + dir[2] * dis
        self.viewMatrixUpdate = True

    def look_at(self, posx: float, posy: float, posz: float):
        # direction, left, and up vectors of the camera
        d = torch.tensor([posx - self.x, posy - self.y, posz - self.z], device=self.device)
        d = d / torch.linalg.vector_norm(d)
        ut = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        l = torch.linalg.cross(ut, d)
        l = l / torch.linalg.vector_norm(l)
        u = torch.linalg.cross(d, l)
        u = u / torch.linalg.vector_norm(u)

        self.rotation_mat = torch.tensor(
            [
                [l[0], l[1], l[2], 0.0],
                [u[0], u[1], u[2], 0.0],
                [d[0], d[1], d[2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )
        self.viewMatrixUpdate = True

    def look_at_top(self, posx: float, posy: float, posz: float, topx: float, topy: float, topz: float):
        d = torch.tensor([posx - self.x, posy - self.y, posz - self.z], device=self.device)
        d = d / torch.linalg.vector_norm(d)
        ut = torch.tensor([topx, topy, topz], device=self.device)
        l = torch.linalg.cross(ut, d)
        l = l / torch.linalg.vector_norm(l)
        u = torch.linalg.cross(d, l)
        u = u / torch.linalg.vector_norm(u)

        self.rotation_mat = torch.tensor(
            [
                [l[0], l[1], l[2], 0.0],
                [u[0], u[1], u[2], 0.0],
                [d[0], d[1], d[2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )
        self.viewMatrixUpdate = True

    def orbit(self, posx: float, posy: float, posz: float, dis: float, anglh: float, anglv: float):
        anglh = anglh - math.pi / 2
        tempx = math.cos(anglh)
        tempz = -math.sin(anglh)

        y = math.sin(anglv) + posy
        xz = math.cos(anglv)
        x = tempx * xz + posx
        z = tempz * xz + posz

        self._distance(x, y, z, posx, posy, posz, dis)
        self.look_at(posx, posy, posz)
