from pathlib import Path
import json
from torchrl.record import CSVLogger

class OptimizationParams:
    def __init__(self):
        self.iterations = 7_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.sh_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001

        self.l_ssim = 0.2
        self.l_depth = 0.1
        self.l_smooth = 0.07

        self.densification_interval = 100
        self.opacity_reset_interval = 3000

        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.random_background = False

    def get_param_dictionary(self) -> dict[str, any]:
        parameters = {
        "iterations": self.iterations,
        "position_lr_init": self.position_lr_init,
        "position_lr_final": self.position_lr_final,
        "position_lr_delay_mult": self.position_lr_delay_mult,
        "position_lr_max_steps": self.position_lr_max_steps,
        "sh_lr": self.sh_lr,
        "opacity_lr": self.opacity_lr,
        "scaling_lr": self.scaling_lr,
        "rotation_lr": self.rotation_lr,

        "l_ssim": self.l_ssim,
        "l_depth": self.l_depth,
        "l_smooth": self.l_smooth,

        "densification_interval": self.densification_interval,
        "opacity_reset_interval": self.opacity_reset_interval,

        "densify_from_iter": self.densify_from_iter,
        "densify_until_iter": self.densify_until_iter,
        "densify_grad_threshold": self.densify_grad_threshold,

        "random_background": self.random_background, # todo, use
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