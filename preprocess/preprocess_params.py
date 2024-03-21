from pathlib import Path
import json
from torchrl.record import CSVLogger
from enum import Enum


class InitModel(Enum):
    Random = 0
    SFM = 1


class CamModel(Enum):
    NoCam = 0
    CamFile = 1
    SFM = 2


class DepthModel(Enum):
    NoDepth = 0
    DepthFile = 1
    SFM = 2


class PreProcessParams:
    def __init__(self):
        self.splats = 10000

        self.init_model = InitModel.Random
        self.depth_model = DepthModel.NoDepth
        self.cam_model = CamModel.NoCam

        self.no_alpha = True

        self.edge_low = 0.3
        self.edge_high = 0.8

    def get_param_dictionary(self) -> dict[str, any]:
        parameters = {
            "splats": self.splats,

            "init_model": self.init_model,
            "depth_model": self.depth_model,
            "cam_model": self.cam_model,

            "no_alpha": self.no_alpha,

            "edge_low": self.edge_low,
            "edge_high": self.edge_high
        }
        return parameters

    def log_params(self, logger: CSVLogger):
        logger.log_hparams(self.get_param_dictionary())

    def load_dictionary(self, params: dict[str, any]):
        for key in params:
            if params[key] is not None:
                eval_string = f"self.{key} = {params[key]}"
                exec(eval_string)

    def load_json(self, param_path: Path):
        with open(param_path) as f:
            data = f.read()
        params = json.loads(data)
        self.load_dictionary(params)

    