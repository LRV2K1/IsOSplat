from main import main
from pathlib import Path
from arguments import ModelParams, PipelineParams, PreProcessParams, OptimizationParams, get_combined_args, GroupParams
from argparse import ArgumentParser

import torch

trainings = ["trexD"] #["fountainD", "fortressD", "hornsD", "orchidsD", "roomD", "flowerD", "fernD", "leavesD", "trexD"]
params = ["mask_100"] #["random_bg", "full", "depth_mask", "mask_000", "mask_025", "mask_050", "mask_075", "mask_100"]

def start(parser):
    optimization_args = OptimizationParams(parser)
    preprocess_args = PreProcessParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    args = get_combined_args(parser)

    for data in trainings:
        for param in params:
            try:
                print(f"{data}-{param}")
                save_path = Path(f"train_output/{data}/{param}")
                data_path = Path(f"train_data/{data}")
                log_path = save_path / f"log"

                pre_param_path = Path(f"params/pre_{param}.json")
                opt_param_path = Path(f"params/opt_{param}.json")
                optimization_args.load_json(opt_param_path)
                preprocess_args.load_json(pre_param_path)

                optimization_params = optimization_args.extract(args)
                preprocess_params = preprocess_args.extract(args)
                # optimization_params.iterations = 100

                main(optimization_params, preprocess_params, data_path, save_path, None, log_path)
            except:
                print("fail")

        
if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")

    torch.set_num_threads(8)

    start(parser)
