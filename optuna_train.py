from pathlib import Path
from typing import Optional
import threading
import time
import os

import tyro
import torch
from torchrl.record import CSVLogger
import optuna

import main
from preprocess.preprocessor import Initialize, CamModel, DepthModel


class OptunaStudy:   
    def __init__(self, study_name: str, data_path: Path, log_path: Optional[Path] = None, iterations: int = 1000):
        self.iterations = iterations
        self.device = torch.device("cuda:0")

        self.logger = None
        if log_path is not None:
            start_time = time.time()
            path_name = "test"
            if data_path is not None:
                path_name = os.path.basename(data_path)
            log_name = f"{study_name} - {path_name} - {iterations} - {start_time}"
            self.logger = CSVLogger(log_dir=log_path, exp_name=log_name)

        data_list, data, point_cloud = main.preprocess(
            data_path=data_path,
            cam_model=CamModel.SFM,
            initialize=Initialize.SFM,
            depth_model=DepthModel.SFM,
            no_alpha=True,
            edge_low=0.5,
            edge_high=0.8,
            device=self.device,
            logger=self.logger)

        self.data_list = data_list
        self.data = data
        self.point_cloud = point_cloud

    def objective(self, trail):
        lr = trail.suggest_float("lr", 0.001, 0.1)
        l_ssim = 0.2
        l_depth = trail.suggest_float("l_depth", 0.0001, 0.25)
        l_smooth = trail.suggest_float("l_smooth", 0.0001, 0.25)

        if self.logger is not None:
            hyperparameters = {
                "lr": lr,
                "l_ssim": l_ssim,
                "l_depth": l_depth,
                "l_smooth": l_smooth,
                "iterations": self.iterations
            }
            self.logger.log_hparams(hyperparameters)

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
            device=self.device,
            logger=self.logger)
        
        loss = main.verify(
            trainer=trainer,
            save_path=None,
            data_list=self.data_list,
            data=self.data,
            iterations=self.iterations,
            logger=self.logger)
        return loss    


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y   


def block(
        study_name: str,
        data_path: Path,
        study: optuna.Study,
        log_path: Optional[Path] = None,
        n_trials: int = 10,
        iterations: int = 1000
):
    optuna_study = OptunaStudy(study_name, data_path, log_path, iterations)
    study.optimize(optuna_study.objective, n_trials=n_trials)


def start(
        data_path: Path,
        storage_path: Path,
        study_name: str,
        log_path: Optional[Path] = None,
        n_trials: int = 100,
        blocks: int = 1,
        iterations: int = 1000
):
    studies = optuna.get_all_study_names(storage="sqlite:///db.sqlite3")

    if study_name in studies:
        study = optuna.load_study(storage="sqlite:///db.sqlite3", study_name=study_name)
    else:
        study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name=study_name)
    
    threads = []
    for i in range(blocks):
        thread = threading.Thread(target=block, args=(study_name, data_path, study, log_path, n_trials, iterations))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        
    # optuna_study = OptunaStudy(data_path, storage_path, iterations)
    # study.optimize(optuna_study.objective, n_trials=n_trials)
    # study.optimize(objective, n_trials=100)
    # print(f"Best value: {study.best_value} (params: {study.best_params})")


if __name__ == '__main__':
    tyro.cli(start)
