"""
This script loads hyperparameters from JSON files and trains models on specified datasets using
the `create_dataset_model_and_train` function from `train.py` or its PyTorch equivalent. The results
are saved in the output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and then calls the appropriate training function based on the
framework (PyTorch or JAX).

Arguments for `run_experiments`:
- `model_names`: List of model architectures to use.
- `dataset_names`: List of datasets to train on.
- `experiment_folder`: Directory containing JSON configuration files.
- `pytorch_experiments`: Boolean indicating whether to use PyTorch (True) or JAX (False).

The script also provides a command-line interface (CLI) for specifying whether to run PyTorch experiments.

Usage:
- Use the `--pytorch_experiments` flag to run experiments with PyTorch; otherwise, JAX is used by default.
"""

import argparse
import json
import os

# For determinism
# Must be set *before* importing jax or triggering any GPU ops
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

def run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments, lr, n_blocks, ssm_dim, embed_dim, time):

    for model_name in model_names:
        for dataset_name in dataset_names:
            with open(
                experiment_folder + f"/{model_name}/{dataset_name}.json", "r"
            ) as file:
                data = json.load(file)

            seeds = data["seeds"]
            data_dir = data["data_dir"]
            output_parent_dir = data["output_parent_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            early_stopping_steps = data["early_stopping_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]
            use_presplit = data["use_presplit"]
            T = data["T"]
            dt0 = None
                
            scale = data["scale"]
            include_time = time
            hidden_dim = embed_dim

            block_size = None
            ssm_dim = ssm_dim
            stepsize = 1
            logsig_depth = 1
            lambd = None
            num_blocks = n_blocks
            vf_depth = None
            vf_width = None
 
            ssm_blocks = None
            output_step = 1

            from train import create_dataset_model_and_train

            model_args = {"num_blocks": num_blocks, "block_size": block_size, "hidden_dim": hidden_dim,
                "vf_depth": vf_depth, "vf_width": vf_width, "ssm_dim": ssm_dim, "ssm_blocks": ssm_blocks,
                "dt0": dt0, "solver": None, "stepsize_controller": None,
                "scale": scale, "lambd": lambd }
            
            run_args = {"data_dir": data_dir, "use_presplit": use_presplit, "dataset_name": dataset_name, "output_step": output_step, "metric": metric, 
                        "include_time": include_time, "T": T, "model_name": model_name, "stepsize": stepsize, "logsig_depth": logsig_depth,
                         "model_args": model_args, "num_steps": num_steps, "print_steps": print_steps, "early_stopping_steps": early_stopping_steps, "lr": lr,
                         "lr_scheduler": lr_scheduler,  "batch_size": batch_size, "output_parent_dir": output_parent_dir  }
            run_fn = create_dataset_model_and_train

            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--pytorch_experiments", action="store_true")
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--n_blocks", type=int, default=2)
    args.add_argument("--ssm_dim", type=int, default=16)
    args.add_argument("--embed_dim", type=int, default=64)
    args.add_argument("--task", type=str, default="EigenWorms")
    args.add_argument("-time", action="store_true")
    args = args.parse_args()
    pytorch_experiments = args.pytorch_experiments

    experiment_folder = "experiment_configs/repeats"
    model_names = ['pdssm']
    dataset_names = [args.task]

    run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments, lr=args.lr, n_blocks=args.n_blocks, ssm_dim=args.ssm_dim, embed_dim=args.embed_dim, time=args.time)
