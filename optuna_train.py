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
    def __init__(
            self, 
            study_name: str, 
            data_path: Path, 
            log_path: Optional[Path] = None, 
            
            iterations: int = 1000, 
            lr: Optional[float] = None,
            l_ssim: Optional[float] = None,
            l_depth: Optional[float] = None,
            l_smooth: Optional[float] = None,
            edge_low: Optional[float] = None,
            edge_high: Optional[float] = None
    ):
        self.study_name = study_name
        self.data_path = data_path
        self.log_path = log_path

        self.initialize = edge_low is not None or edge_high is not None
        self.device = torch.device("cuda:0")

        self.iterations = iterations
        self.lr = lr
        self.l_ssim = l_ssim
        self.l_depth = l_depth
        self.l_smooth = l_smooth
        self.edge_low = edge_low
        self.edge_high = edge_high

        if not self.initialize:
            self._initialize()

    def _initialize(self, trial) -> tuple[float, float]:
        self.logger = None
        if self.log_path is not None:
            start_time = time.time()
            path_name = "test"
            if self.data_path is not None:
                path_name = os.path.basename(self.data_path)
            log_name = f"{self.study_name} - {path_name} - {self.iterations} - {start_time}"
            self.logger = CSVLogger(log_dir=self.log_path, exp_name=log_name)

        edge_low = self.edge_low
        edge_high = self.edge_high
        if edge_low is None:
            if edge_high is not None:
                edge_low = trial.suggest_float("edge_low", 0.01, min(edge_high, 1.0))
            else:
                edge_low = trial.suggest_float("edge_low", 0.01, 1.0)
        if edge_high is None:
            edge_high = trial.suggest_float("edge_high", edge_low, 1.5)

        data_list, data, point_cloud = main.preprocess(
            data_path=self.data_path,
            cam_model=CamModel.SFM,
            initialize=Initialize.SFM,
            depth_model=DepthModel.SFM,
            no_alpha=True,
            edge_low=edge_low,
            edge_high=edge_high,
            device=self.device,
            logger=self.logger)

        self.data_list = data_list
        self.data = data
        self.point_cloud = point_cloud

        return edge_low, edge_high

    def objective(self, trial):
        edge_low = self.edge_low
        edge_high = self.edge_high        
        if self.initialize:
            edge_low, edge_high = self._initialize(trial)

        lr = self.lr
        if lr is None:
            lr = trial.suggest_float("lr", 0.001, 0.1)
        l_ssim = self.l_ssim
        if l_ssim is None:
            l_ssim = trial.suggest_float("l_ssim", 0.001, 0.1)
        l_depth = self.l_depth
        if l_depth is None:
            l_depth = trial.suggest_float("l_depth", 0.0001, 0.25)
        l_smooth = self.l_smooth
        if l_smooth is None:
            l_smooth = trial.suggest_float("l_smooth", 0.0001, 0.25)

        if self.logger is not None:
            hyperparameters = {
                "lr": lr,
                "l_ssim": l_ssim,
                "l_depth": l_depth,
                "l_smooth": l_smooth,
                "edge_low": edge_low,
                "edge_high": edge_high,
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

        iterations: int = 1000,
        lr: Optional[float] = None,
        l_ssim: Optional[float] = None,
        l_depth: Optional[float] = None,
        l_smooth: Optional[float] = None,
        edge_low: Optional[float] = None,
        edge_high: Optional[float] = None
):
    optuna_study = OptunaStudy(
        study_name=study_name, 
        data_path=data_path, 
        log_path=log_path, 
        iterations=iterations,
        lr=lr,
        l_ssim=l_ssim,
        l_depth=l_depth,
        l_smooth=l_smooth,
        edge_low=edge_low,
        edge_high=edge_high)
    study.optimize(optuna_study.objective, n_trials=n_trials)


def start(
        data_path: Optional[Path] = None,
        study_name: Optional[str] = None,
        storage_path: Path = "sqlite:///db.sqlite3",
        log_path: Optional[Path] = None,
        n_trials: int = 100,
        blocks: int = 1,

        iterations: int = 1000,
        lr: Optional[float] = None,
        l_ssim: Optional[float] = None,
        l_depth: Optional[float] = None,
        l_smooth: Optional[float] = None,
        edge_low: Optional[float] = None,
        edge_high: Optional[float] = None
):
    studies = optuna.get_all_study_names(storage=storage_path)

    if study_name is None:
        study_name = "Test Study"
    if data_path is None:
        optuna_study = OptunaStudy(data_path, storage_path, iterations)
        study.optimize(optuna_study.objective, n_trials=n_trials)
        study.optimize(objective, n_trials=100)
        print(f"Best value: {study.best_value} (params: {study.best_params})")
        return

    if study_name in studies:
        study = optuna.load_study(storage="sqlite:///db.sqlite3", study_name=study_name)
    else:
        study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name=study_name)
    
    threads = []
    for i in range(blocks):
        thread = threading.Thread(target=block, 
                                  args=(
                                      study_name, 
                                      data_path, 
                                      study, 
                                      log_path, 
                                      n_trials, 
                                      iterations,
                                      lr,
                                      l_ssim,
                                      l_depth,
                                      l_smooth,
                                      edge_low,
                                      edge_high))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    tyro.cli(start)
