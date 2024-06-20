#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from pathlib import Path
from typing import Optional
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


class GroupParams:
    def get_param_dictionary(self) -> dict[str, any]:
        params = {}
        for key, value in vars(self).items():
            params[key] = value
        return params
    
    def log_params(self, logger: CSVLogger):
        logger.log_hparams(self.get_param_dictionary())


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=None, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=None, type=t)

    def extract(self, args) -> GroupParams:
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                if arg[1] is not None:
                    setattr(group, arg[0], arg[1])
                else:
                    setattr(group, arg[0], vars(self)[arg[0]])
                # setattr(group, arg[0], vars(self)[arg[0]])
        return group
    
    def _load_dictionary(self, params: dict[str, any]):
        for key in params:
            if key in vars(self) and params[key] is not None:
                setattr(self, key, params[key])

    def load_json(self, json_path: Optional[Path]):
        if json_path is None:
            return
        with open(json_path) as f:
            data = f.read()
        params = json.loads(data)
        self._load_dictionary(params)

class ModelParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, sentinel: bool = False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        ##################### custom
        self.seed = 42
        # self.usefulldepth = False
        # self.usesingledepth = False
        # self.usefullcolmap = False
        # self.isBA = False
        self.kshot = 1000 # a large number
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class PreProcessParams(ParamGroup):
    def __init__(self, parser, json_path: Optional[Path] = None):
        self.splats = 10000

        self.init_model = "Random"
        self.depth_model = "NoDepth"
        self.cam_model = "NoCam"

        self.no_alpha = True
        self.no_segments = False

        self.edge_low = 0.3
        self.edge_high = 0.8
        super().__init__(parser, "PreProcess Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.iterations = 7_000
        self.min_iters = 1500
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.sh_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

        self.l_ssim = 0.2
        self.l_depth = 0.1
        self.l_smooth = 0.07
        self.l_bounds = 1.0

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500 #500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.random_background = False # At initial version we used, there was no random_background
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser) -> Namespace:
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        #if v != None:
        merged_dict[k] = v
    return Namespace(**merged_dict)
