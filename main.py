from pathlib import Path
from typing import Optional

import tyro
import torch

from isosplat.gaussian_splatting import GaussianSplatting
from preprocess.preprocessor import Initialize, CamModel, DepthModel, PreProcessor


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
        l_depth: float = 0.1
) -> None:
    device = torch.device("cuda:0")

    preprocessor = PreProcessor(data_path)
    data_list, data, point_cloud = preprocessor.preprocess_data(
        device=device,
        cam_model=cam_model,
        initialize=initialize,
        depth_model=depth_model,
        no_alpha=no_alpha
    )

    trainer = GaussianSplatting(device)

    trainer.init_gaussians(splats, load_path, point_cloud)
    trainer.init_optimizer(lr, l_ssim, l_depth)
    if iterations > 0 and len(data_list) > 0:
        trainer.train(
            data_list=data_list,
            data=data,
            iterations=iterations,
        )
    trainer.verify(data_list, data, save_path)
    if save_path:
        trainer.save(save_path)


if __name__ == '__main__':
    tyro.cli(main)
