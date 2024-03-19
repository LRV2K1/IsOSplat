from pathlib import Path
from typing import Optional

import tyro
import torch
from torch import Tensor

from isosplat.gaussian_splatting import GaussianSplatting
from preprocess.preprocessor import Initialize, CamModel, DepthModel, PreProcessor, PointCloud


def main(
        data_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        load_path: Optional[Path] = None,
        iterations: int = 1000,
        lr: float = 0.01,
        splats: int = 100000,
        initialize: Initialize = Initialize.Random,
        cam_model: CamModel = CamModel.CamFile,
        depth_model: DepthModel = DepthModel.NoDepth,
        no_alpha: bool = True,
        l_ssim: float = 0.2,
        l_depth: float = 0.1,
        l_smooth: float = 0.1,
        edge_low: float = 0.5,
        edge_high: float = 0.8
) -> float:
    device = torch.device("cuda:0")

    data_list, data, point_cloud = preprocess(
        data_path=data_path, 
        cam_model=cam_model, 
        initialize=initialize, 
        depth_model=depth_model, 
        no_alpha=no_alpha, 
        edge_low=edge_low,
        edge_high=edge_high,
        device=device)
    
    trainer = train(
        data_list=data_list, 
        data=data, 
        point_cloud=point_cloud, 
        load_path=load_path,
        lr=lr,
        l_ssim=l_ssim,
        l_depth=l_depth,
        l_smooth=l_smooth,
        iterations=iterations,
        splats=splats,
        device=device)
        
    loss = verify(
        trainer=trainer,
        save_path=save_path,
        data_list=data_list,
        data=data)

    return loss


def preprocess(data_path: Optional[Path], cam_model: CamModel, initialize: Initialize, depth_model: DepthModel, no_alpha: bool, edge_low: float, edge_high: float, device: torch.device
               ) -> tuple[list[str], dict[str, any], PointCloud]:
    preprocessor = PreProcessor(data_path)
    data_list, data, point_cloud = preprocessor.preprocess_data(
        device=device,
        cam_model=cam_model,
        edge_low=edge_low,
        edge_high=edge_high,
        initialize=initialize,
        depth_model=depth_model,
        no_alpha=no_alpha
    )
    return data_list, data, point_cloud


def train(data_list: list[str], data: dict[str, any], point_cloud: PointCloud, load_path: Optional[Path], lr: float, l_ssim: float, l_depth: float, l_smooth: float, iterations: int, splats: int, device: torch.device
          ) -> GaussianSplatting:
    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud)
    trainer.init_optimizer(lr, l_ssim, l_depth, l_smooth)
    if iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            iterations=iterations,
        )
    return trainer


def verify(trainer: GaussianSplatting, save_path: Optional[Path], data_list: list[str], data: dict[str, any]) -> float:
    loss = trainer.verify(data_list, data, save_path)
    if save_path:
        trainer.save(save_path)
    return loss


if __name__ == '__main__':
    tyro.cli(main)
