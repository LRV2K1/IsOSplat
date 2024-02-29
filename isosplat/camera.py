import math
import torch
from torch import Tensor


class Camera:

    def __init__(self,
                 width: int, height: int,
                 focalx: float, focaly: float,
                 near: float, far: float,
                 device: torch.device):
        self.device = device

        self.width = width
        self.height = height

        self.focalx = focalx
        self.focaly = focaly

        fov_width = (2 * self.focalx) / width
        fov_height = (2 * self.focaly) / height
        a = (far + near) / (far - near)
        b = -(far * near) / (far - near)

        self.perspective_project_mat = torch.tensor(
            [
                [fov_width, 0.0, 0.0, 0.0],
                [0.0, fov_height, 0.0, 0.0],
                [0.0, 0.0, a, b],
                [0.0, 0.0, 1.0, 0.0]
            ],
            device=self.device
        )

        self.translation_mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -8.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )

        self.rotation_mat = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )

        self.viewMatrixUpdate = True

        self.translation_mat.requires_grad = False
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
            self.view_matrix = torch.matmul(self.rotation_mat, self.translation_mat)
            self.view_matrix.requires_grad = False

            self.project_matrix = torch.matmul(self.perspective_project_mat, self.view_matrix)
            self.project_matrix.requires_grad = False

            self.viewMatrixUpdate = False

    def get_view_and_project_matrix(self) -> tuple[Tensor, Tensor]:
        self._update_view_and_projection_matrix()
        return self.view_matrix, self.project_matrix

    def get_view_direction(self) -> Tensor:
        direction = torch.tensor(
            [self.rotation_mat[2, 0],
             self.rotation_mat[2, 1],
             self.rotation_mat[2, 2]],
            device=self.device
        )
        return direction

    def get_camera_position(self) -> Tensor:
        pos = torch.tensor(
            [
                self.translation_mat[0, 3],
                self.translation_mat[1, 3],
                self.translation_mat[2, 3]
            ],
            device=self.device
        )
        return pos

    def get_focal(self) -> tuple[float, float]:
        return self.focalx, self.focaly

    def get_size(self) -> tuple[int, int]:
        return self.width, self.height

    def set_position(self, posx: float, posy: float, posz: float):
        self.translation_mat = torch.transpose(torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [posx, posy, posz, 1.0]
            ],
            device=self.device
        ), 0, 1)
        self.viewMatrixUpdate = True

    def set_view_direction(self, dirx, diry, dirz):
        self.look_at(-dirx, -diry, -dirz)

    def distance(self, posx: float, posy: float, posz: float, dis: float):
        camx = self.translation_mat[0, 3]
        camy = self.translation_mat[1, 3]
        camz = self.translation_mat[2, 3]
        self._distance(camx, camy, camz, posx, posy, posz, dis)

    def _distance(self, camx: float, camy: float, camz: float, posx: float, posy: float, posz: float, dis: float):
        dir = torch.tensor([camx - posx, camy - posy, camz - posz], device=self.device)
        dir = dir / torch.linalg.vector_norm(dir)

        self.translation_mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, dir[0] * dis],
                [0.0, 1.0, 0.0, dir[1] * dis],
                [0.0, 0.0, 1.0, dir[2] * dis],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )
        self.viewMatrixUpdate = True

    def look_at(self, posx: float, posy: float, posz: float):
        camx = self.translation_mat[0, 3]
        camy = self.translation_mat[1, 3]
        camz = self.translation_mat[2, 3]

        # direction, right, and up vectors of the camera
        d = torch.tensor([camx - posx, camy - posy, camz - posz], device=self.device)
        d = d / torch.linalg.vector_norm(d)
        r = torch.tensor([camz - posz, 0.0, -camx - posx], device=self.device)
        r = r / torch.linalg.vector_norm(r)
        u = torch.linalg.cross(d, r)

        self.rotation_mat = torch.tensor(
            [
                [r[0], r[1], r[2], 0.0],
                [u[0], u[1], u[2], 0.0],
                [d[0], d[1], d[2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            device=self.device
        )
        self.viewMatrixUpdate = True

    def look_at_top(self, posx: float, posy: float, posz: float, topx: float, topy: float, topz: float):
        camx = self.translation_mat[0, 3]
        camy = self.translation_mat[1, 3]
        camz = self.translation_mat[2, 3]

        d = torch.tensor([camx - posx, camy - posy, camz - posz], device=self.device)
        d = d / torch.linalg.vector_norm(d)
        u = torch.tensor([topx, topy, topz], device=self.device)
        u = u / torch.linalg.vector_norm(d)
        r = torch.linalg.cross(d, u)

        self.rotation_mat = torch.tensor(
            [
                [r[0], r[1], r[2], 0.0],
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
        tempy = math.sin(anglv)
        tempz = math.sin(anglh)

        self._distance(tempx, tempy, tempz, posx, posy, posz, dis)
        self.look_at(posx, posy, posz)
