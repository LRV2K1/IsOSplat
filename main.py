from pathlib import Path
from typing import Optional
import time
import os

import tyro
import torch
from torchrl.record import CSVLogger

from isosplat.gaussian_splatting import GaussianSplatting
from isosplat.utils import PointCloud, Data, DataList
from isosplat.optimization_params import OptimizationParams
from preprocess.preprocessor import PreProcessor
from preprocess.preprocess_params import InitModel, CamModel, DepthModel, PreProcessParams


def main(
        data_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        load_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
        opt_param_path: Optional[Path] = None,
        pre_param_path: Optional[Path] = None,

        # optimization params
        iterations: Optional[int] = None,
        position_lr_init: Optional[float] = None,
        position_lr_final: Optional[float] = None,
        position_lr_delay_mult: Optional[float] = None,
        position_lr_max_steps: Optional[int] = None,
        sh_lr: Optional[float] = None,
        opacity_lr: Optional[float] = None,
        scaling_lr: Optional[float] = None,
        rotation_lr: Optional[float] = None,
        
        l_ssim: Optional[float] = None,
        l_depth: Optional[float] = None,
        l_smooth: Optional[float] = None,

        densification_interval: Optional[int] = None,
        opacity_reset_interval: Optional[int] = None,

        densify_from_iter: Optional[int] = None,
        densify_until_iter: Optional[int] = None,
        densify_grad_threshold: Optional[float] = None,

        random_background: Optional[bool] = None,

        # preprocess params
        splats: Optional[int] = None,

        init_model: Optional[InitModel] = None,
        cam_model: Optional[CamModel] = None,
        depth_model: Optional[DepthModel] = None,

        no_alpha: Optional[bool] = None,

        edge_low: Optional[float] = None,
        edge_high: Optional[float] = None
) -> float:
    optimization_params = OptimizationParams()
    opt_parameters = {
        "iterations": iterations,
        "position_lr_init": position_lr_init,
        "position_lr_final": position_lr_final,
        "position_lr_delay_mult": position_lr_delay_mult,
        "position_lr_max_steps": position_lr_max_steps,
        "sh_lr": sh_lr,
        "opacity_lr": opacity_lr,
        "scaling_lr": scaling_lr,
        "rotation_lr": rotation_lr,

        "l_ssim": l_ssim,
        "l_depth": l_depth,
        "l_smooth": l_smooth,

        "densification_interval": densification_interval,
        "opacity_reset_interval": opacity_reset_interval,

        "densify_from_iter": densify_from_iter,
        "densify_until_iter": densify_until_iter,
        "densify_grad_threshold": densify_grad_threshold,

        "random_background": random_background,
        }
    if opt_param_path is not None:
        optimization_params.load_json(opt_param_path)
    optimization_params.load_dictionary(opt_parameters)
    
    preprocess_params = PreProcessParams()
    pre_parameters = {
        "splats": splats,

        "init_model": init_model,
        "cam_model": cam_model,
        "depth_model": depth_model,

        "no_alpha": no_alpha,

        "edge_low": edge_low,
        "edge_high": edge_high       
    }
    if pre_param_path is not None:
        preprocess_params.load_json(pre_param_path)
    preprocess_params.load_dictionary(pre_parameters)

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
        }
        logger.log_hparams(hyperparameters)
        optimization_params.log_params(logger)
        preprocess_params.log_params(logger)

    device = torch.device("cuda:0")

    data_list, data, point_cloud = preprocess(
        data_path=data_path, 
        preprocess_params=preprocess_params,
        device=device,
        logger=logger)
    
    trainer = train(
        data_list=data_list, 
        data=data, 
        point_cloud=point_cloud, 
        load_path=load_path,
        optimization_params=optimization_params,
        splats=preprocess_params.splats,
        device=device,
        logger=logger)
        
    loss = verify(
        trainer=trainer,
        save_path=save_path,
        data_list=data_list,
        data=data,
        logger=logger)

    return loss


def preprocess(
        data_path: Optional[Path],
        preprocess_params: PreProcessParams,
        device: torch.device,
        logger: Optional[CSVLogger] = None
) -> tuple[DataList, Data, Optional[PointCloud]]:
    preprocessor = PreProcessor(data_path)
    data_list, data, point_cloud = preprocessor.preprocess_data(
        device=device,
        preprocess_params=preprocess_params,
        logger=logger
    )
    return data_list, data, point_cloud


def train(
        data_list: DataList,
        data: Data,
        point_cloud: Optional[PointCloud],
        load_path: Optional[Path],
        optimization_params: OptimizationParams,
        splats: int,
        device: torch.device,
        logger: Optional[CSVLogger] = None
) -> GaussianSplatting:
    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud, logger)
    trainer.init_optimizer(optimization_params)
    if optimization_params.iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            logger=logger
        )
    return trainer


def verify(
        trainer: GaussianSplatting,
        save_path: Optional[Path],
        data_list: list[str],
        data: dict[str, any],
        logger: Optional[CSVLogger] = None
) -> float:
    loss = trainer.verify(
        data_list=data_list,
        data=data,
        save_path=save_path,
        logger=logger)
    if save_path:
        trainer.save(save_path)
    return loss


if __name__ == '__main__':
    tyro.cli(main)
