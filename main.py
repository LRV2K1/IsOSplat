from pathlib import Path
from typing import Optional
import time
import os

# import tyro
import torch
from torchrl.record import CSVLogger

from isosplat.gaussian_splatting import GaussianSplatting
from utils.graphics_utils import BasicPointCloud
from isosplat.utils import Data, DataList
from preprocess.preprocessor import PreProcessor
from arguments import ModelParams, PipelineParams, PreProcessParams, OptimizationParams, get_combined_args, GroupParams
from argparse import ArgumentParser


def main(
        optimization_params: GroupParams,
        preprocess_params: GroupParams,

        data_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        load_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
) -> float:
    logger = None
    if log_path is not None:
        start_time = time.time()
        path_name = "test"
        if data_path is not None:
            path_name = os.path.basename(data_path)
        log_name = f"{path_name} - {optimization_params.iterations} - {start_time}"
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
        logger=logger,
        save_path=save_path)
        
    loss = verify(
        trainer=trainer,
        save_path=save_path,
        data_list=data_list,
        data=data,
        logger=logger)

    return loss


def preprocess(
        data_path: Optional[Path],
        preprocess_params: GroupParams,
        device: torch.device,
        logger: Optional[CSVLogger] = None
) -> tuple[DataList, Data, Optional[BasicPointCloud]]:
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
        point_cloud: Optional[BasicPointCloud],
        load_path: Optional[Path],
        optimization_params: GroupParams,
        splats: int,
        device: torch.device,
        logger: Optional[CSVLogger] = None,
        save_path: Optional[Path] = None
) -> GaussianSplatting:
    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud, logger)
    trainer.init_optimizer(optimization_params)
    if optimization_params.iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            logger=logger,
            save_path=save_path
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
    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--save_path", default=None, type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--log_path", default=None, type=Path)
    parser.add_argument("--opt_param_path", default=None, type=Path)
    parser.add_argument("--pre_param_path", default=None, type=Path)

    optimization_args = OptimizationParams(parser)
    preprocess_args = PreProcessParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    args = get_combined_args(parser)

    optimization_args.load_json(args.opt_param_path)
    preprocess_args.load_json(args.pre_param_path)

    optimization_params = optimization_args.extract(args)
    preprocess_params = preprocess_args.extract(args)

    # tyro.cli(main)
    main(optimization_params, preprocess_params, args.data_path, args.save_path, args.load_path, args.log_path)
