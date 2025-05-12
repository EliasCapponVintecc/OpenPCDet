# evaluate_multiple_models.py (without reset_cfg, fix for LOCAL_RANK)
import argparse
import datetime
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict  # Import EasyDict
from eval_utils import eval_utils
from pcdet.config import cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


# Keep the original eval_single_ckpt function (modified to accept cfg explicitly)
def eval_single_ckpt_modified(
    model, test_loader, args, eval_output_dir, logger, epoch_id, current_cfg, ckpt_path, dist_test=False
):
    """Modified to explicitly take current_cfg and ckpt_path"""
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=dist_test)
    model.cuda()
    eval_utils.eval_one_epoch(
        current_cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=eval_output_dir
    )


def parse_eval_config():
    # ... (parse_eval_config function remains the same) ...
    parser = argparse.ArgumentParser(description="arg parser for multiple model evaluation")
    parser.add_argument("--cfg_file", type=str, required=True, help="YAML file listing models to evaluate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size PER GPU. Overrides YAML global.")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers. Overrides YAML global.")
    parser.add_argument("--extra_tag", type=str, default=None, help="Extra tag for output dir. Overrides YAML global.")
    parser.add_argument("--eval_tag", type=str, default=None, help="Evaluation tag. Overrides YAML global.")
    parser.add_argument(
        "--save_to_file",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save predictions. Overrides YAML global.",
    )
    parser.add_argument(
        "--infer_time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Calculate inference latency. Overrides YAML global.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888, help="tcp port for distrbuted training")
    parser.add_argument("--local_rank", type=int, default=None, help="local rank for distributed training")
    args = parser.parse_args()
    try:
        with open(args.eval_config_file) as f:
            eval_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation config file not found: {args.eval_config_file}")
    except Exception as e:
        raise RuntimeError(f"Error parsing YAML file {args.eval_config_file}: {e}")
    if "models_to_evaluate" not in eval_config or not isinstance(eval_config["models_to_evaluate"], list):
        raise ValueError(f"YAML file {args.eval_config_file} must contain a list under 'models_to_evaluate' key.")
    global_cfg = eval_config.get("global_settings", {})
    for key in ["batch_size", "workers", "extra_tag", "eval_tag", "save_to_file", "infer_time"]:
        cmd_val = getattr(args, key)
        yaml_val = eval_config.get(key, None)
        yaml_global_val = global_cfg.get(key, None)
        final_val = cmd_val if cmd_val is not None else (yaml_val if yaml_val is not None else yaml_global_val)
        if final_val is not None:
            setattr(args, key, final_val)
        elif getattr(args, key) is None:
            if key == "batch_size":
                args.batch_size = 4
            if key == "workers":
                args.workers = 4
            if key == "extra_tag":
                args.extra_tag = "multi_eval_default"
            if key == "eval_tag":
                args.eval_tag = "results"
            if key == "save_to_file":
                args.save_to_file = False
            if key == "infer_time":
                args.infer_time = False
    np.random.seed(1024)
    return args, eval_config["models_to_evaluate"]


def main():
    args, models_to_evaluate = parse_eval_config()

    if args.infer_time:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # --- Distributed setup (run once) ---
    if args.launcher == "none":
        dist_test = False
        total_gpus = 1
        local_rank = 0  # Explicitly set local_rank for non-distributed case
    else:
        # Ensure local_rank is determined *before* the loop
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # Get from env if not passed

        total_gpus, determined_local_rank = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        local_rank = determined_local_rank  # Use the rank returned by init_dist
        dist_test = True

    # Check and calculate batch size per GPU
    if args.batch_size % total_gpus != 0:
        print(
            f"Warning: Batch size {args.batch_size} not divisible by total GPUs {total_gpus}. Results might be inconsistent if batch size matters."
        )
        # Decide if you want to floor/ceil or raise error. Floor is safer.
        # batch_size_per_gpu = args.batch_size // total_gpus
        # Or just proceed, maybe the model handles it. Let's keep original calculation for now.
    batch_size_per_gpu = args.batch_size // total_gpus
    print(f"Rank {local_rank}: Using batch size per GPU: {batch_size_per_gpu} (Total batch: {args.batch_size})")

    # --- Loop through each model defined in the YAML ---
    for model_info in models_to_evaluate:
        model_name = model_info.get("name", Path(model_info["cfg_file"]).stem)
        cfg_file = model_info["cfg_file"]
        ckpt_path = model_info["ckpt"]

        # Print only on rank 0 to avoid clutter
        if local_rank == 0:
            print("\n" + "=" * 40)
            print(f" Evaluating Model: {model_name} ")
            print(f" Config: {cfg_file} ")
            print(f" Checkpoint: {ckpt_path} ")
            print("=" * 40 + "\n")

        # --- Create a NEW, EMPTY config object for THIS model ---
        current_cfg = edict()

        # --- Load specific config for THIS model INTO the NEW object---
        cfg_from_yaml_file(cfg_file, current_cfg)  # Pass current_cfg as the target

        # Set essential attributes derived from the config file path
        current_cfg.TAG = Path(cfg_file).stem
        if current_cfg.get("ROOT_DIR", None) is None:
            script_dir = Path(__file__).resolve().parent
            current_cfg.ROOT_DIR = script_dir.parent
            if local_rank == 0:  # Print warning only once
                print(f"Warning: ROOT_DIR not set in {cfg_file}. Assuming {current_cfg.ROOT_DIR}")
        else:
            current_cfg.ROOT_DIR = Path(current_cfg.ROOT_DIR)

        try:
            tools_dir = current_cfg.ROOT_DIR / "tools"
            if tools_dir.exists():
                relative_cfg_path = Path(cfg_file).resolve().relative_to(tools_dir)
                current_cfg.EXP_GROUP_PATH = str(relative_cfg_path.parent.parent)
            else:
                if local_rank == 0:
                    print(
                        f"Warning: Cannot find 'tools' directory under {current_cfg.ROOT_DIR}. Cannot determine EXP_GROUP_PATH accurately."
                    )
                current_cfg.EXP_GROUP_PATH = Path(cfg_file).parent.name
        except ValueError as e:
            if local_rank == 0:
                print(
                    f"Warning: Could not automatically determine EXP_GROUP_PATH from {cfg_file} relative to {tools_dir}. Using cfg file parent dir name. Error: {e}"
                )
            current_cfg.EXP_GROUP_PATH = Path(cfg_file).parent.name

        # Apply model-specific overrides from YAML ('set_cfgs') to the NEW object
        if "set_cfgs" in model_info:
            cfg_from_list(model_info["set_cfgs"], current_cfg)  # Pass current_cfg as the target

        # ******************************************************************** #
        # *                          ADD LOCAL_RANK                          * #
        # Manually add the determined local_rank to the current config object  #
        # as functions like eval_one_epoch might expect it directly on cfg     #
        current_cfg.LOCAL_RANK = local_rank
        # Optionally add total GPUs if needed by any internal function, though less common
        # current_cfg.TOTAL_GPUS = total_gpus
        # ******************************************************************** #

        # --- Setup Output Dirs and Logger for THIS model ---
        model_extra_tag = model_info.get("extra_tag", args.extra_tag)
        model_eval_tag = model_info.get("eval_tag", args.eval_tag)

        output_dir = current_cfg.ROOT_DIR / "output" / current_cfg.EXP_GROUP_PATH / current_cfg.TAG / model_extra_tag
        eval_output_dir = output_dir / "eval"

        num_list = re.findall(r"checkpoint_epoch_(\d+)\.pth", ckpt_path)
        epoch_id_str = num_list[-1] if num_list else "specified_ckpt"

        eval_output_dir = eval_output_dir / f"eval_{model_name}_epoch_{epoch_id_str}" / model_eval_tag

        # Create output dir only on rank 0
        if local_rank == 0:
            eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Wait briefly for rank 0 to create directory in distributed setting
        if dist_test:
            torch.distributed.barrier()

        log_file = eval_output_dir / (
            "log_eval_%s_%s.txt" % (model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        # Logger setup respects the local_rank passed to it
        logger = common_utils.create_logger(log_file, rank=local_rank)

        # Log initial info only on rank 0
        if local_rank == 0:
            logger.info(f"--- Evaluating Model: {model_name} ---")
            logger.info(f"Config File: {cfg_file}")
            logger.info(f"Checkpoint: {ckpt_path}")
            logger.info(f"Output Directory: {eval_output_dir}")
            logger.info("**********************Args**********************")
            merged_args = vars(args).copy()
            merged_args["current_model_name"] = model_name
            merged_args["current_cfg_file"] = cfg_file
            merged_args["current_ckpt"] = ckpt_path
            merged_args["batch_size_per_gpu"] = batch_size_per_gpu
            for key, val in merged_args.items():
                logger.info(f"{key:16} {val}")

            logger.info("**********************Config**********************")
            # Log the specific configuration we loaded for this model
            log_config_to_file(current_cfg, logger=logger)  # Pass current_cfg

        # --- Build Dataloader and Network for THIS model ---
        current_batch_size_per_gpu = model_info.get("batch_size", batch_size_per_gpu)
        current_workers = model_info.get("workers", args.workers)

        if local_rank == 0:  # Log effective batch/workers only once
            logger.info(f"Using effective batch size per GPU: {current_batch_size_per_gpu}, workers: {current_workers}")

        # Pass the specific current_cfg to build_dataloader
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=current_cfg.DATA_CONFIG,
            class_names=current_cfg.CLASS_NAMES,
            batch_size=current_batch_size_per_gpu,
            dist=dist_test,
            workers=current_workers,
            logger=logger,
            training=False,
            # root_path=current_cfg.ROOT_DIR # Pass root_path if needed by dataset
        )

        # Pass the specific current_cfg to build_network
        model = build_network(model_cfg=current_cfg.MODEL, num_class=len(current_cfg.CLASS_NAMES), dataset=test_set)

        current_save_to_file = model_info.get("save_to_file", args.save_to_file)
        current_infer_time = model_info.get("infer_time", args.infer_time)

        temp_args = argparse.Namespace(**vars(args))
        temp_args.save_to_file = current_save_to_file
        temp_args.infer_time = current_infer_time

        # --- Run Evaluation ---
        with torch.no_grad():
            # Pass the specific current_cfg to the evaluation function
            eval_single_ckpt_modified(
                model=model,
                test_loader=test_loader,
                args=temp_args,
                eval_output_dir=eval_output_dir,
                logger=logger,
                epoch_id=epoch_id_str,
                current_cfg=current_cfg,  # Pass the correct config (now with LOCAL_RANK)
                ckpt_path=ckpt_path,
                dist_test=dist_test,
            )

        if local_rank == 0:
            logger.info(f"--- Finished Evaluating Model: {model_name} ---")

        # Optional barrier to ensure all ranks finish before next model
        if dist_test:
            torch.distributed.barrier()

        # Optional: Clear CUDA cache if memory becomes an issue between models
        # torch.cuda.empty_cache()

    if local_rank == 0:
        print("\nAll specified models evaluated.")


if __name__ == "__main__":
    main()
