from pathlib import Path
from typing import Optional
import threading
import time
import os
import json

import tyro
import torch
from torchrl.record import CSVLogger
import optuna

import main
from isosplat.optimization_params import OptimizationParams
from preprocess.preprocess_params import PreProcessParams


class OptunaStudy:     
    def __init__(
            self,
            train_param_path: Path, 
            study_name: str, 
            data_path: Path, 
            log_path: Optional[Path] = None, 
            
            opt_param_path: Optional[Path] = None,
            pre_param_path: Optional[Path] = None,
    ):
        self.device = torch.device("cuda:0")

        self.study_name = study_name
        self.data_path = data_path
        self.log_path = log_path

        self.preprocess_params = PreProcessParams()
        if pre_param_path is not None:
            self.preprocess_params.load_json(pre_param_path)
        self.optimizable_params = OptimizationParams()
        if opt_param_path is not None:
            self.optimizable_params.load_json(opt_param_path)

        with open(train_param_path) as f:
            params = f.read()
        self.train_params = json.loads(params)

    def objective(self, trial):
        logger = None
        if self.log_path is not None:
            start_time = time.time()
            path_name = trial.number
            if self.data_path is not None:
                path_name = os.path.basename(self.data_path)
            log_name = f"{self.study_name} - {path_name} - {self.optimizable_params.iterations} - {start_time}"
            logger = CSVLogger(log_dir=self.log_path, exp_name=log_name)

        opt_parameters = self.optimizable_params.get_param_dictionary()
        pre_parameters = self.preprocess_params.get_param_dictionary()
        new_opt_parameters = {}
        new_pre_parameters = {}
        for key in self.train_params:
            mi, ma = self.train_params[key]
            x = trial.suggest_float(key, mi, ma)
            if key in opt_parameters:
                new_opt_parameters[key] = x
            if key in pre_parameters:
                new_pre_parameters[key] = x
        self.optimizable_params.load_dictionary(new_opt_parameters)
        self.preprocess_params.load_dictionary(new_opt_parameters)

        if logger is not None:
            logger.log_hparams(self.optimizable_params.get_param_dictionary())
            logger.log_hparams(self.preprocess_params.get_param_dictionary())

        data_list, data, point_cloud = main.preprocess(
            data_path=self.data_path,
            preprocess_params=self.preprocess_params,
            device=self.device,
            logger=logger)

        self.trainer = main.train(
            data_list=data_list, 
            data=data, 
            point_cloud=point_cloud, 
            load_path=None,
            optimization_params=self.optimizable_params,
            splats=self.preprocess_params.splats,
            device=self.device,
            logger=logger)
        
        loss = main.verify(
            trainer=self.trainer,
            save_path=None,
            data_list=data_list,
            data=data,
            logger=logger)
        return loss    


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y   


def block(
        train_param_path: Path,
        study_name: str,
        data_path: Path,
        study: optuna.Study,
        log_path: Optional[Path] = None,
        n_trials: int = 10,
        opt_param_path: Optional[Path] = None,
        pre_param_path: Optional[Path] = None,
):
    optuna_study = OptunaStudy(
        study_name=study_name, 
        data_path=data_path, 
        log_path=log_path, 
        train_param_path=train_param_path,
        opt_param_path=opt_param_path,
        pre_param_path=pre_param_path)
    study.optimize(optuna_study.objective, n_trials=n_trials)


def start(
        train_param_path: Path,
        data_path: Optional[Path] = None,
        study_name: Optional[str] = None,
        storage_path: Path = "sqlite:///db.sqlite3",
        log_path: Optional[Path] = None,
        opt_param_path: Optional[Path] = None,
        pre_param_path: Optional[Path] = None,

        n_trials: int = 100,
        blocks: int = 1,
):
    studies = optuna.get_all_study_names(storage=storage_path)

    if study_name is None:
        study_name = "Test Study"
    if data_path is None:
        optuna_study = OptunaStudy(data_path, storage_path, opt_param_path, pre_param_path, train_param_path)
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
                                      train_param_path,
                                      study_name, 
                                      data_path, 
                                      study, 
                                      log_path, 
                                      n_trials, 
                                      opt_param_path, 
                                      pre_param_path))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    tyro.cli(start)
