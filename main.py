from pathlib import Path
from typing import Optional
import time
import os

import tyro
import torch
from torchrl.record import CSVLogger

from isosplat.gaussian_splatting import GaussianSplatting
from preprocess.preprocessor import Initialize, CamModel, DepthModel, PreProcessor, PointCloud


def main(
        data_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        load_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
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
        edge_low: float = 0.3,
        edge_high: float = 0.8
) -> float:
    logger = None
    if log_path is not None:
        start_time = time.time()
        path_name = "test"
        if data_path is not None:
            path_name = os.path.basename(data_path)
        log_name = f"{path_name} - {iterations} - {start_time}"
        logger = CSVLogger(log_dir=log_path, exp_name=log_name)
        hyperparameters = {
            "data_path": data_path,
            "save_path": save_path,
            "load_path": load_path,
            "iterations": iterations,
            "lr": lr,
            "splats": splats,
            "initialize": initialize,
            "cam_model": cam_model,
            "depth_model": depth_model,
            "no_alpha": no_alpha,
            "l_ssim": l_ssim,
            "l_depth": l_depth,
            "l_smooth": l_smooth,
            "edge_low": edge_low,
            "edge_high": edge_high
        }
        logger.log_hparams(hyperparameters)

    device = torch.device("cuda:0")

    data_list, data, point_cloud = preprocess(
        data_path=data_path, 
        cam_model=cam_model, 
        initialize=initialize, 
        depth_model=depth_model, 
        no_alpha=no_alpha, 
        edge_low=edge_low,
        edge_high=edge_high,
        device=device,
        logger=logger)
    
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
        device=device,
        logger=logger)
        
    loss = verify(
        trainer=trainer,
        save_path=save_path,
        data_list=data_list,
        data=data,
        iterations=iterations,
        logger=logger)

    return loss


def preprocess(
        data_path: Optional[Path],
        cam_model: CamModel,
        initialize: Initialize,
        depth_model: DepthModel,
        no_alpha: bool,
        edge_low: float,
        edge_high: float,
        device: torch.device,
        logger: Optional[CSVLogger] = None
) -> tuple[list[str], dict[str, any], PointCloud]:
    preprocessor = PreProcessor(data_path)
    data_list, data, point_cloud = preprocessor.preprocess_data(
        device=device,
        cam_model=cam_model,
        edge_low=edge_low,
        edge_high=edge_high,
        initialize=initialize,
        depth_model=depth_model,
        no_alpha=no_alpha,
        logger=logger
    )
    return data_list, data, point_cloud


def train(
        data_list: list[str],
        data: dict[str, any],
        point_cloud: PointCloud,
        load_path: Optional[Path],
        lr: float, l_ssim: float,
        l_depth: float,
        l_smooth: float,
        iterations: int,
        splats: int,
        device: torch.device,
        logger: Optional[CSVLogger] = None
) -> GaussianSplatting:
    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud, logger)
    trainer.init_optimizer(lr, l_ssim, l_depth, l_smooth)
    if iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            iterations=iterations,
            logger=logger
        )
    return trainer


def verify(
        trainer: GaussianSplatting,
        save_path: Optional[Path],
        data_list: list[str],
        data: dict[str, any],
        iterations: int,
        logger: Optional[CSVLogger] = None
) -> float:
    loss = trainer.verify(
        data_list=data_list,
        data=data,
        iterations=iterations,
        save_path=save_path,
        logger=logger)
    if save_path:
        trainer.save(save_path)
    return loss


if __name__ == '__main__':
    tyro.cli(main)
