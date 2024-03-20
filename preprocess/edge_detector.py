import numpy as np
import cv2 as cv

import torch
from torch import Tensor, nn


class CannyEdgeDetector:
    def __init__(
            self,
            device: torch.device,
            threshold_low: float = 0.3,
            threshold_high: float = 0.8,
            sigma: float = 1,
            kernel_size: int = 3
    ):
        gaussian_kernel = get_gaussian_kernel(k=kernel_size, sigma=sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=kernel_size,
                                         padding=kernel_size // 2,
                                         device=device,
                                         bias=False)
        gaussian_weight_tensor = torch.zeros_like(self.gaussian_filter.weight, device=device)
        gaussian_weight_tensor[0, 0] = torch.from_numpy(gaussian_kernel)
        self.gaussian_filter.weight = torch.nn.Parameter(gaussian_weight_tensor)

        sobel_2d = torch.tensor([
            [-0.5, 0, 0.5],
            [-1, 0, 1],
            [-0.5, 0, 0.5]
        ], device=device)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=3 // 2,
                                        device=device,
                                        bias=False)
        sobel_weight_tensor = torch.zeros_like(self.sobel_filter_x.weight, device=device)
        sobel_weight_tensor[0, 0] = sobel_2d
        self.sobel_filter_x.weight = torch.nn.Parameter(torch.transpose(sobel_weight_tensor, 2, 3))

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=3,
                                        padding=3 // 2,
                                        device=device,
                                        bias=False)
        self.sobel_filter_y.weight = torch.nn.Parameter(sobel_weight_tensor)

        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def calculate_edge_map(self, name: str, img: Tensor, device: torch.device) -> Tensor:
        print(f"Calculate edge map {name}")
        height, width, colors = img.shape

        blurred = torch.zeros(colors, height, width, device=device)

        a = torch.zeros(1, height, width, device=device)
        b = torch.zeros(1, height, width, device=device)
        c = torch.zeros(1, height, width, device=device)
        for i in range(colors):
            blurred[i, :, :] = img[:, :, i]
            blurred[i:i+1, :, :] = self.gaussian_filter(blurred[i:i+1, :, :])

            grad_x = self.sobel_filter_x(blurred[i:i+1, :, :])
            grad_y = self.sobel_filter_y(blurred[i:i+1, :, :])
            a = a + grad_x ** 2
            b = b + grad_y ** 2
            c = c + grad_x * grad_y

        d = ((a - b) ** 2 + 4 * c ** 2) ** 0.5
        grad_magnitude = (0.5 * (a + b + d)) ** 0.5
        e_x = a - b + d
        e_y = 2 * c

        rot = np.pi / 8
        orientation_x = e_x * np.cos(rot) - e_y * np.sin(rot)
        orientation_y = e_x * np.sin(rot) - e_y * np.cos(rot)
        mirror_mask = orientation_y < 0
        orientation_x[mirror_mask] *= -1
        orientation_y[mirror_mask] *= -1

        # create orientations
        q0_mask = torch.logical_and(orientation_x >= 0, orientation_x >= orientation_y)
        q1_mask = torch.logical_and(orientation_x >= 0, orientation_x < orientation_y)
        q2_mask = torch.logical_and(orientation_x < 0, -orientation_x < orientation_y)
        q3_mask = torch.logical_and(orientation_x < 0, -orientation_x >= orientation_y)
        orientation_masks = [q0_mask, q1_mask, q2_mask, q3_mask]

        local_max_mask = torch.zeros(1, height, width, dtype=torch.bool, device=device)
        m_c = grad_magnitude[:, 1:height - 1, 1:width - 1]
        m_l = torch.zeros_like(m_c, device=device)
        m_r = torch.zeros_like(m_c, device=device)
        local_max_mask[:, 1:height-1, 1:width-1] = m_c >= self.threshold_low

        u_v = [(1, 0), (1, 1), (0, 1), (1, -1)]
        for i in range(4):
            u, v = u_v[i]
            orientation_mask = (orientation_masks[i])[:, 1:height - 1, 1:width - 1]
            m_l_local = grad_magnitude[:, (1-u):((height - 1)-u), (1-v):((width - 1)-v)]
            m_l[orientation_mask] = m_l_local[orientation_mask]
            m_r_local = grad_magnitude[:, (1+u):((height - 1)+u), (1+v):((width - 1)+v)]
            m_r[orientation_mask] = m_r_local[orientation_mask]
        local_max_mask[:, 1:height-1, 1:width-1] = torch.logical_and(local_max_mask[:, 1:height-1, 1:width-1],
                                                                     torch.logical_and(m_l <= m_c, m_c >= m_r))
        local_max = torch.zeros_like(grad_magnitude, device=device)
        local_max[local_max_mask] = grad_magnitude[local_max_mask]

        e_bin = torch.zeros(height, width, dtype=torch.bool, device=device)

        local_max_list = local_max[0].tolist()
        e_bin_list = e_bin.tolist()

        for u in range(1, height-2):
            for v in range(1, width - 2):
                if (local_max_list[u][v] >= self.threshold_high) and not e_bin_list[u][v]:
                    e_bin_list = self.trace_and_threshold(local_max_list, e_bin_list, u, v, height, width)

        e_bin = torch.tensor(e_bin_list, dtype=torch.bool, device=device)
        return e_bin

    def trace_and_threshold(self, local_max: list, e_bin: list, u, v, height, width) -> Tensor:
        e_bin[u][v] = True
        ut = max(u-1, 0)
        ub = min(u+1, height - 1)
        vl = max(v-1, 0)
        vr = min(v+1, width-1)
        for un in range(ut, ub+1):
            for vn in range(vl, vr+1):
                if (local_max[un][vn] >= self.threshold_low) and not e_bin[un][vn]:
                    e_bin = self.trace_and_threshold(local_max, e_bin, un, vn, height, width)

        return e_bin


def get_gaussian_kernel(k: int = 3, mu: float = 0, sigma: float = 1, normalize: bool = True) -> np.ndarray:
    # compute 1 dimension gaussian
    gaussian_1d = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1d, gaussian_1d)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2d = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2d = gaussian_2d / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
    return gaussian_2d


class CV2CannyEdgeDetector:
    def __init__(self, threshold_low: float = 0.3, threshold_high: float = 0.8):
        self.threshold_low = threshold_low * 255
        self.threshold_high = threshold_high * 255
    
    def calculate_edge_map(self, name: str, img: Tensor, device: torch.device) -> Tensor:
        print(f"Calculate edge map {name}")
        image = (img.detach().cpu().numpy() * 255).astype(np.uint8)
        edges = cv.Canny(image, self.threshold_low, self.threshold_high)

        edge_map = torch.tensor(edges, dtype=torch.bool, device=device)
        return edge_map
