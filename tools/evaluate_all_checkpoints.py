import _init_path # Assuming this is for your project's path setup
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for evaluation')
    parser.add_argument('--checkpoints_dir_to_eval', type=str, required=True,
                        help='Directory containing checkpoint .pth files to evaluate')

    parser.add_argument('--batch_size', type=int, default=None, help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='eval_all_script', help='extra tag for this experiment')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment (subfolder under eval)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--start_epoch', type=int, default=0, help='Optional: only evaluate epochs >= start_epoch')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='if save results to file') # from original
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency') # from original


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    # --- Output Directory Setup ---
    # Base output directory related to the config and extra_tag
    output_dir_base = Path(cfg.ROOT_DIR) / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # Main 'eval' directory
    eval_root_dir = output_dir_base / 'eval'

    # Optional sub-tag directory within 'eval'
    if args.eval_tag != 'default' and args.eval_tag is not None:
        eval_parent_dir_for_epochs = eval_root_dir / args.eval_tag
    else:
        eval_parent_dir_for_epochs = eval_root_dir
    eval_parent_dir_for_epochs.mkdir(parents=True, exist_ok=True)

    # --- Logger Setup (logs to the eval_parent_dir_for_epochs) ---
    log_file = eval_parent_dir_for_epochs / ('log_eval_all_script_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start Evaluation Script**********************')
    gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    logger.info(f"Evaluating checkpoints from: {args.checkpoints_dir_to_eval}")

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # --- Dataloader and Model ---
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # --- TensorBoard Setup (logs to the eval_parent_dir_for_epochs) ---
    tb_log = None
    if cfg.LOCAL_RANK == 0:
        tb_log_dir = eval_parent_dir_for_epochs / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")


    # --- Iterate and Evaluate Checkpoints ---
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoints_dir_to_eval, 'checkpoint_epoch_*.pth')))
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {args.checkpoints_dir_to_eval} matching 'checkpoint_epoch_*.pth'")
        return

    logger.info(f"Found {len(checkpoint_files)} checkpoint(s) to evaluate.")

    with torch.no_grad():
        for ckpt_file_path in checkpoint_files:
            ckpt_filename = Path(ckpt_file_path).name
            # Extract epoch number
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt_filename)
            if not match:
                # Try a more general pattern if the first one fails, e.g. epoch_xx.pth
                match_general = re.search(r'epoch_(\d+)', ckpt_filename)
                if not match_general:
                    logger.warning(f"Could not extract epoch number from checkpoint: {ckpt_filename}. Skipping.")
                    continue
                epoch_id = match_general.group(1)
            else:
                epoch_id = match.group(1)

            if int(epoch_id) < args.start_epoch:
                logger.info(f"Skipping epoch {epoch_id} as it is less than start_epoch {args.start_epoch}.")
                continue

            logger.info(f"--- Evaluating checkpoint: {ckpt_filename} (Epoch: {epoch_id}) ---")

            # Load checkpoint
            model.load_params_from_file(filename=ckpt_file_path, logger=logger, to_cpu=dist_test)
            model.cuda()
            model.eval() # Ensure model is in evaluation mode

            # Define current result directory: eval_parent_dir_for_epochs / epoch_X / val
            current_epoch_result_dir = eval_parent_dir_for_epochs / f'epoch_{epoch_id}' / cfg.DATA_CONFIG.DATA_SPLIT['test']
            current_epoch_result_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Results for epoch {epoch_id} will be saved to: {current_epoch_result_dir}")


            # Start evaluation for this epoch
            tb_dict = eval_utils.eval_one_epoch(
                cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
                result_dir=current_epoch_result_dir
            )

            if cfg.LOCAL_RANK == 0 and tb_log is not None and tb_dict:
                for key, val in tb_dict.items():
                    tb_log.add_scalar(f'{key}', val, int(epoch_id))
                logger.info(f"Logged metrics for epoch {epoch_id} to TensorBoard.")

            logger.info(f"Finished evaluating epoch {epoch_id}")
            if dist_test: # synchronize after each epoch if distributed
                torch.distributed.barrier()


    if cfg.LOCAL_RANK == 0 and tb_log is not None:
        tb_log.close()
    logger.info('**********************Evaluation Script Finished**********************')

if __name__ == '__main__':
    main()
