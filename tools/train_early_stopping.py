import argparse
import datetime
import glob
import os
from pathlib import Path

import tqdm

import torch
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from tensorboardX import SummaryWriter
from torch import nn
import numpy as np
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import (
    checkpoint_state,
    disable_augmentation_hook,
    save_checkpoint,
    train_one_epoch,
)
from eval_utils import eval_utils

from test import repeat_eval_ckpt
# --- End Imports ---


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--cfg_file", type=str, default=None, help="specify the config for training")

    parser.add_argument("--batch_size", type=int, default=None, required=False, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=None, required=False, help="number of epochs to train for")
    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for this experiment")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to start from")
    parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained_model")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888, help="tcp port for distrbuted training")
    parser.add_argument("--sync_bn", action="store_true", default=False, help="whether to use sync bn")
    parser.add_argument("--fix_random_seed", action="store_true", default=False, help="")
    parser.add_argument(
        "--ckpt_save_interval", type=int, default=1, help="number of training epochs"
    )  # Default to 1 for early stopping best model saving
    parser.add_argument("--local_rank", type=int, default=None, help="local rank for distributed training")
    parser.add_argument("--max_ckpt_save_num", type=int, default=30, help="max number of saved checkpoint")
    parser.add_argument("--merge_all_iters_to_one_epoch", action="store_true", default=False, help="")
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )

    parser.add_argument("--max_waiting_mins", type=int, default=0, help="max waiting minutes")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    # num_epochs_to_eval is less relevant when using early stopping for the primary model selection
    parser.add_argument(
        "--num_epochs_to_eval",
        type=int,
        default=1,
        help="number of final checkpoints to evaluate after training finishes",
    )
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")

    parser.add_argument(
        "--use_tqdm_to_record",
        action="store_true",
        default=False,
        help="if True, the intermediate losses will not be logged to file, only tqdm will be used",
    )
    parser.add_argument("--logger_iter_interval", type=int, default=50, help="")
    parser.add_argument("--ckpt_save_time_interval", type=int, default=300, help="in terms of seconds")
    parser.add_argument("--wo_gpu_stat", action="store_true", help="")
    parser.add_argument("--use_amp", action="store_true", help="use mix precision training")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get("USE_AMP", False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # Set default early stopping config if not present
    if "EARLY_STOPPING" not in cfg:
        cfg.EARLY_STOPPING = argparse.Namespace()  # Use Namespace for dot access
        cfg.EARLY_STOPPING.ENABLE = False
    # Ensure default values if partially defined
    if not hasattr(cfg.EARLY_STOPPING, "ENABLE"):
        cfg.EARLY_STOPPING.ENABLE = False
    if not hasattr(cfg.EARLY_STOPPING, "PATIENCE"):
        cfg.EARLY_STOPPING.PATIENCE = 10
    if not hasattr(cfg.EARLY_STOPPING, "METRIC"):
        cfg.EARLY_STOPPING.METRIC = "mAP"  # Common default
    if not hasattr(cfg.EARLY_STOPPING, "MODE"):
        cfg.EARLY_STOPPING.MODE = "max"
    if not hasattr(cfg.EARLY_STOPPING, "MIN_DELTA"):
        cfg.EARLY_STOPPING.MIN_DELTA = 0.0

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == "none":
        dist_train = False
        total_gpus = 1
        cfg.LOCAL_RANK = 0  # Set local rank to 0 for single GPU case
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, "Batch size should match the number of gpus"
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / "ckpt"
    eval_output_dir = output_dir / "eval"  # Define eval dir earlier
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)  # Create eval dir

    log_file = output_dir / ("train_%s.log" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ.keys() else "ALL"
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_train:
        logger.info("Training in distributed mode : total_batch_size: %d" % (total_gpus * args.batch_size))
    else:
        logger.info("Training with a single process")

    for key, val in vars(args).items():
        logger.info(f"{key:16} {val}")
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system("cp %s %s" % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if cfg.LOCAL_RANK == 0 else None

    # --- Create dataloaders ---
    logger.info("----------- Create dataloaders -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None,
    )

    # --- Create Validation Loader ---
    # Use the split defined in cfg.DATA_CONFIG.DATA_SPLIT['test'] (usually 'val')
    val_set, val_loader, val_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False,  # <<< Set to False for validation/test set
    )

    # --- Create network & optimizer ---
    logger.info("----------- Create network & optimizer -----------")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # --- Load checkpoint ---
    start_epoch = it = 0
    last_epoch = -1  # Necessary for scheduler
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch
    else:  # Try to resume from latest automatically
        ckpt_list = glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pth"))
        ckpt_list.sort(key=os.path.getmtime)
        resumed = False
        # Try resuming from best_model first if early stopping might have been used before
        best_ckpt_path = ckpt_dir / "best_model.pth"
        if best_ckpt_path.exists():
            try:
                logger.info(f"Attempting to resume from best model: {best_ckpt_path}")
                it, start_epoch = model.load_params_with_optimizer(
                    best_ckpt_path, to_cpu=dist_train, optimizer=optimizer, logger=logger
                )
                last_epoch = start_epoch
                resumed = True
            except Exception as e:
                logger.warning(f"Could not load best model checkpoint: {e}. Trying latest epoch checkpoint.")

        if not resumed and len(ckpt_list) > 0:  # Fallback to latest epoch checkpoint
            try:
                logger.info(f"Attempting to resume from latest epoch checkpoint: {ckpt_list[-1]}")
                it, start_epoch = model.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                )
                last_epoch = start_epoch
                resumed = True
            except Exception as e:
                logger.warning(f"Could not load latest epoch checkpoint {ckpt_list[-1]}: {e}. Starting from scratch.")

        if not resumed:  # If neither best nor latest worked
            logger.info("No valid checkpoint found or specified, starting from scratch.")
            start_epoch = 0
            it = 0
            last_epoch = -1
    freeze_backbone = cfg.OPTIMIZATION.get("FREEZE_BACKBONE", False)
    backbone_name = cfg.OPTIMIZATION.get("BACKBONE_NAME", "backbone_3d")

    if freeze_backbone:
        logger.info(f"================== Freezing {backbone_name} ==================")
        num_params_frozen = 0
        num_params_trainable = 0
        num_params_total = 0
        found_backbone = False

        # Handle potential DDP wrapping later, operate on base model now
        base_model = model

        for name, param in base_model.named_parameters():
            num_params_total += param.numel()
            # Check if the parameter belongs to the specified backbone module
            if name.startswith(f"{backbone_name}."):
                param.requires_grad = False
                num_params_frozen += param.numel()
                found_backbone = True
            else:
                param.requires_grad = True  # Ensure other parts are trainable
                num_params_trainable += param.numel()

        if not found_backbone:
            logger.warning(
                f"Could not find parameters starting with '{backbone_name}.' to freeze. ALL parameters remain trainable."
            )
            # Ensure all grads are True if backbone wasn't found
            for param in base_model.parameters():
                param.requires_grad = True
            num_params_trainable = num_params_total
            num_params_frozen = 0
        else:
            logger.info(f"Froze {num_params_frozen} parameters in {backbone_name}.")

        logger.info(f"Total parameters: {num_params_total}")
        logger.info(f"Trainable parameters: {num_params_trainable}")
        logger.info("=======================================================")
    else:
        logger.info("Backbone not frozen. All parameters are trainable.")
        # Explicitly ensure all parameters are trainable if flag is off
        for param in model.parameters():
            param.requires_grad = True

    model.train()  # Ensure train mode before DDP wrap
    if dist_train:
        # find_unused_parameters can be necessary if some parameters aren't used in forward pass (e.g. during fine-tuning)
        find_unused = cfg.get("FIND_UNUSED_PARAMETERS", False)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=find_unused
        )
    logger.info(f"----------- Model {cfg.MODEL.NAME} created -----------")
    # logger.info(model) # Can be very verbose

    # --- Scheduler ---
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader),
        total_epochs=args.epochs,
        last_epoch=last_epoch,  # Pass last_epoch correctly
        optim_cfg=cfg.OPTIMIZATION,
    )

    # ----------------------- Early Stopping Initialization ---------------------
    early_stopping_enabled = cfg.EARLY_STOPPING.ENABLE
    patience = cfg.EARLY_STOPPING.PATIENCE
    es_metric = cfg.EARLY_STOPPING.METRIC
    es_mode = cfg.EARLY_STOPPING.MODE
    min_delta = cfg.EARLY_STOPPING.MIN_DELTA
    patience_counter = 0
    best_val_metric = -np.inf if es_mode == "max" else np.inf
    best_epoch = -1
    stop_training_flag = False  # Flag to signal stopping across ranks

    if early_stopping_enabled and cfg.LOCAL_RANK == 0:
        logger.info(
            f"Early stopping enabled: metric='{es_metric}', mode='{es_mode}', patience={patience}, min_delta={min_delta}"
        )

    # ----------------------- Start Training Loop ---------------------------
    logger.info(
        "**********************Start training %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    accumulated_iter = it
    total_it_each_epoch = len(train_loader)
    # Handle merging iters for datasets like Waymo
    if args.merge_all_iters_to_one_epoch:
        assert hasattr(train_loader.dataset, "merge_all_iters_to_one_epoch")
        train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=args.epochs)
        total_it_each_epoch = len(train_loader) // max(args.epochs, 1)

    dataloader_iter = iter(train_loader)

    # Hook for disabling data augmentation near end of training
    hook_config = cfg.get("HOOK", None)
    augment_disable_flag = False

    # --- Main Epoch Loop ---
    with tqdm.trange(
        start_epoch,
        args.epochs,
        desc="epochs",
        dynamic_ncols=True,
        leave=(cfg.LOCAL_RANK == 0),
        disable=(cfg.LOCAL_RANK != 0),  # Only show bar on rank 0
    ) as epoch_tbar:
        for cur_epoch in epoch_tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # Determine current scheduler
            if lr_warmup_scheduler is not None and cur_epoch < cfg.OPTIMIZATION.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # --- Train One Epoch ---
            augment_disable_flag = disable_augmentation_hook(
                hook_config, dataloader_iter, args.epochs, cur_epoch, cfg, augment_disable_flag, logger
            )

            model.train()  # Ensure model is in train mode for the epoch
            accumulated_iter = train_one_epoch(
                model,
                optimizer,
                train_loader,
                model_fn_decorator(),
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=cfg.OPTIMIZATION,
                rank=cfg.LOCAL_RANK,
                tbar=epoch_tbar,
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == args.epochs),
                use_logger_to_record=not args.use_tqdm_to_record,
                logger=logger,
                logger_iter_interval=args.logger_iter_interval,
                cur_epoch=cur_epoch,
                total_epochs=args.epochs,
                ckpt_save_dir=ckpt_dir,
                ckpt_save_time_interval=args.ckpt_save_time_interval,
                show_gpu_stat=not args.wo_gpu_stat,
                use_amp=args.use_amp,
            )
            trained_epoch = cur_epoch + 1

            # --- Save Checkpoint (Regular Interval) ---
            if trained_epoch % args.ckpt_save_interval == 0 and cfg.LOCAL_RANK == 0:
                ckpt_list = glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pth"))
                ckpt_list.sort(key=os.path.getmtime)

                if len(ckpt_list) >= args.max_ckpt_save_num > 0:
                    for cur_file_idx in range(len(ckpt_list) - args.max_ckpt_save_num + 1):
                        try:
                            os.remove(ckpt_list[cur_file_idx])
                            # logger.info(f"Removed old checkpoint: {ckpt_list[cur_file_idx]}") # Can be verbose
                        except OSError as e:
                            logger.error(f"Error removing checkpoint {ckpt_list[cur_file_idx]}: {e}")

                ckpt_name = ckpt_dir / ("checkpoint_epoch_%d" % trained_epoch)
                state = checkpoint_state(
                    model.module if dist_train else model, optimizer, trained_epoch, accumulated_iter
                )
                save_checkpoint(state, filename=ckpt_name)
                # logger.info(f"Saved checkpoint to {ckpt_name}.pth") # Can be verbose

            # --- Evaluate After Each Epoch (for Early Stopping) ---
            if early_stopping_enabled:
                # Sync processes before evaluation to ensure model weights are consistent
                if dist_train:
                    torch.distributed.barrier()

                current_val_metric = None  # Initialize for this epoch
                # Evaluate only on Rank 0
                if cfg.LOCAL_RANK == 0:
                    # logger.info(f"--- Starting evaluation for epoch {trained_epoch} ---")
                    model.eval()  # Set model to evaluation mode

                    # Define output dir for this epoch's eval results (optional)
                    epoch_eval_output_dir = (
                        eval_output_dir / f"epoch_{trained_epoch}" / cfg.DATA_CONFIG.DATA_SPLIT["test"]
                    )
                    epoch_eval_output_dir.mkdir(parents=True, exist_ok=True)

                    # Run evaluation using eval_one_epoch from eval_utils
                    ret_dict = eval_utils.eval_one_epoch(
                        cfg,
                        args,
                        model.module if dist_train else model,
                        val_loader,
                        trained_epoch,
                        logger,
                        dist_test=False,  # Important: eval on rank 0 uses the single val_loader
                        result_dir=epoch_eval_output_dir
                    )

                    # logger.info(f"--- Finished evaluation for epoch {trained_epoch} ---")

                    # Extract the monitored metric
                    try:
                        current_val_metric = float(ret_dict[es_metric])
                        epoch_tbar.set_postfix(val_metric=f"{current_val_metric:.4f}")  # Show metric in tqdm bar
                        logger.info(f"Validation metric ({es_metric}): {current_val_metric:.6f}") # Can be verbose
                    except KeyError:
                        logger.error(
                            f"Early stopping metric '{es_metric}' not found in eval results: {ret_dict.keys()}. Disabling early stopping."
                        )
                        early_stopping_enabled = False  # Disable if metric key is wrong
                    except ValueError:
                        logger.error(f"Val metric '{es_metric}' value '{ret_dict[es_metric]}' not float. Disabling ES.")
                        early_stopping_enabled = False

                    if early_stopping_enabled and current_val_metric is not None:
                        # Check for improvement
                        improved = False
                        if (es_mode == "max" and current_val_metric > best_val_metric + min_delta) or (
                            es_mode == "min" and current_val_metric < best_val_metric - min_delta
                        ):
                            improved = True

                        if improved:
                            best_val_metric = current_val_metric
                            best_epoch = trained_epoch
                            patience_counter = 0
                            logger.info(
                                f"*** Best Val ({es_metric}): {best_val_metric:.6f} at epoch {best_epoch}. Resetting patience. ***"
                            )

                            # Save the best model checkpoint
                            best_ckpt_name = ckpt_dir / "best_model"
                            state = checkpoint_state(
                                model.module if dist_train else model, optimizer, trained_epoch, accumulated_iter
                            )
                            save_checkpoint(state, filename=best_ckpt_name)
                            logger.info(f"Saved best model checkpoint to {best_ckpt_name}.pth")

                        else:
                            patience_counter += 1
                            logger.info(f"Val metric ({es_metric}) did not improve for {patience_counter} epochs.") # Can be verbose

                        # Check if patience is exceeded
                        if patience_counter >= patience:
                            logger.warning(
                                f"Early stopping triggered at epoch {trained_epoch} after {patience} epochs without improvement."
                            )
                            stop_training_flag = True  # Signal to stop

                    model.train()  # Set model back to training mode for next epoch

                # --- Distribute Stop Signal ---
                if dist_train:
                    # Broadcast the stop flag from rank 0 to all other ranks
                    stop_tensor = torch.tensor(int(stop_training_flag), dtype=torch.int, device=model.device)
                    torch.distributed.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        # logger.info(f"Rank {cfg.LOCAL_RANK} received stop signal.") # Can be verbose
                        break  # Exit loop on all ranks if flag is set

            # --- End of Epoch ---
            lr_scheduler.step(cur_epoch)  # Step scheduler at end of epoch

    # ----------------------- End Training Loop ---------------------------
    logger.info("Finished Training Loop.")
    if hasattr(train_set, "use_shared_memory") and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info(
        "**********************End training %s/%s(%s)**********************\n\n\n"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    # ----------------------- Final Evaluation ---------------------
    # Always evaluate the best model found if early stopping was enabled and a best model exists
    if cfg.LOCAL_RANK == 0:  # Only rank 0 performs final evaluation
        logger.info(
            "**********************Start Final Evaluation %s/%s(%s)**********************"
            % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
        )
        final_eval_output_dir = eval_output_dir / "final_eval"
        final_eval_output_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt_path = ckpt_dir / "best_model.pth"
        if early_stopping_enabled and best_ckpt_path.exists():
            logger.info(f"Evaluating best model found by early stopping at epoch {best_epoch}: {best_ckpt_path}")
            # Load the best checkpoint before evaluation
            # model.load_params_from_file(str(best_ckpt_path), to_cpu=dist_train, logger=logger) # Load state dict directly is simpler
            eval_single_ckpt(
                cfg,
                model.module if dist_train else model,
                val_loader,
                args,
                final_eval_output_dir,
                logger,
                epoch_id=f"best_epoch_{best_epoch}",
                dist_test=False,  # Final eval on rank 0
                specified_ckpt_path=str(best_ckpt_path),  # Pass path to load inside
            )
        elif not early_stopping_enabled:
            # If early stopping wasn't used, evaluate the last checkpoint saved
            logger.info("Early stopping not enabled. Evaluating last saved checkpoint.")
            ckpt_list = glob.glob(str(ckpt_dir / "checkpoint_epoch_*.pth"))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) > 0:
                last_ckpt_path = ckpt_list[-1]
                logger.info(f"Evaluating last checkpoint: {last_ckpt_path}")
                last_epoch_num = int(Path(last_ckpt_path).stem.split("_")[-1])
                eval_single_ckpt(
                    cfg,
                    model.module if dist_train else model,
                    val_loader,
                    args,
                    final_eval_output_dir,
                    logger,
                    epoch_id=f"final_epoch_{last_epoch_num}",
                    dist_test=False,
                    specified_ckpt_path=str(last_ckpt_path),
                )
            else:
                logger.warning("No checkpoints found to evaluate.")
        else:
            logger.info(
                "Early stopping was enabled, but no 'best_model.pth' found. Skipping final evaluation of best model."
            )

        logger.info(
            "**********************End Final Evaluation %s/%s(%s)**********************"
            % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
        )


if __name__ == "__main__":
    main()
