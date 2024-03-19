from pathlib import Path
from typing import Optional

import main
import tyro
import optuna
import torch

from preprocess.preprocessor import Initialize, CamModel, DepthModel, PreProcessor, PointCloud



class OptunaStudy:   
    def __init__(self, data_path: Path, storage_path: Path, iterations):
        self.iterations = iterations
        self.device = torch.device("cuda:0")

        data_list, data, point_cloud = main.preprocess(
        data_path=data_path, 
        cam_model=CamModel.SFM, 
        initialize=Initialize.SFM, 
        depth_model=DepthModel.SFM, 
        no_alpha=True, 
        device=self.device)

        self.data_list = data_list
        self.data = data
        self.point_cloud = point_cloud

    def objective(self, trail):
        lr = trail.suggest_float("lr", 0.001, 0.1)
        l_ssim = 0.2
        l_depth = trail.suggest_float("l_depth", 0.0001, 0.25)
        l_smooth = trail.suggest_float("l_smooth", 0.0001, 0.25)

        trainer = main.train(
            data_list=self.data_list, 
            data=self.data, 
            point_cloud=self.point_cloud, 
            load_path=None,
            lr=lr,
            l_ssim=l_ssim,
            l_depth=l_depth,
            l_smooth=l_smooth,
            iterations=self.iterations,
            splats=0,
            device=self.device)
        
        loss = main.verify(
            trainer=trainer,
            save_path=None,
            data_list=self.data_list,
            data=self.data)  
        return loss    


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y   


def start(data_path: Path,
         storage_path: Path,
         study_name: str,
         n_trials: int = 100,
         iterations: int = 1000):
    
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name=study_name
    )
    ostudy = OptunaStudy(data_path, storage_path, iterations)
    study.optimize(ostudy.objective, n_trials=n_trials)
    # study.optimize(objective, n_trials=100)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")

if __name__ == '__main__':
    tyro.cli(start)
