from math import exp

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(out_img: Tensor, gt: Tensor) -> Tensor:
    return torch.abs((out_img - gt)).mean()


def l2_loss(out_img: Tensor, gt: Tensor) -> Tensor:
    return torch.abs((out_img - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


uv_s = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def l_smooth(out_depth: Tensor, edge_map: Tensor) -> Tensor:
    height, width = out_depth.shape
    edge_map_l = edge_map[1:height-1, 1:width-1]
    final_smooth = torch.zeros(height-2, width-2, device=out_depth.device)
    d_i = out_depth[1:height-1, 1:width-1]
    for uv in uv_s:     # go over all adjacent
        u, v = uv
        d_j = out_depth[(1+u):((height-1)+u), (1+v):((width-1)+v)]  # shift to adjacent pixel

        if v > 0 or (v == 0 and u > 0):
            d_j[edge_map_l] = d_i[edge_map_l]  # set in edge
        else:
            d_i[edge_map_l] = d_j[edge_map_l]  # set in edge

        l2 = l2_loss(d_i, d_j)
        final_smooth += l2  # add smooth
    return final_smooth.mean()
