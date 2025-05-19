import json
import pickle
import time

from pathlib import Path

import numpy as np
from sympy import sec
import torch
import tqdm
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            # Handle potential NaN/Inf which are not valid JSON
            if np.isnan(o):
                return None
            if np.isinf(o):
                return 1e300 if o > 0 else -1e300  # Or None or 'Infinity' string
            return float(o)
        if isinstance(o, np.ndarray):
            # Convert array elements recursively if necessary, but tolist() is usually fine
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        # Let the base class default method raise the TypeError for other types
        return super(NumpyEncoder, self).default(o)


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] += ret_dict.get("roi_%s" % str(cur_thresh), 0)
        metric["recall_rcnn_%s" % str(cur_thresh)] += ret_dict.get("rcnn_%s" % str(cur_thresh), 0)
    metric["gt_num"] += ret_dict.get("gt", 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict["recall_%s" % str(min_thresh)] = "(%d, %d) / %d" % (
        metric["recall_roi_%s" % str(min_thresh)],
        metric["recall_rcnn_%s" % str(min_thresh)],
        metric["gt_num"],
    )


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / "final_result" / "data"
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        "gt_num": 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] = 0
        metric["recall_rcnn_%s" % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, "infer_time", False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info("*************** EPOCH %s EVALUATION *****************" % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True)
    loop_start_time = time.time()  # Renamed
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, "infer_time", False):
            batch_infer_start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, "infer_time", False):
            inference_time = time.time() - batch_infer_start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict["infer_time"] = f"{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})"

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / "tmpdir")
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / "tmpdir")

    logger.info("*************** Performance of EPOCH %s *****************" % epoch_id)
    sec_per_example = (time.time() - loop_start_time) / len(dataloader.dataset)
    logger.info("Generate label finished(sec_per_example: %.4f second)." % sec_per_example)
    logger.info("average inference time: %.4f ms" % (infer_time_meter.avg if getattr(args, "infer_time", False) else 0))

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric["gt_num"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric["recall_roi_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric["recall_rcnn_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info("recall_roi_%s: %f" % (cur_thresh, cur_roi_recall))
        logger.info("recall_rcnn_%s: %f" % (cur_thresh, cur_rcnn_recall))
        ret_dict["recall/roi_%s" % str(cur_thresh)] = cur_roi_recall
        ret_dict["recall/rcnn_%s" % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno["name"].__len__()
    logger.info(
        "Average predicted number of objects(%d samples): %.3f"
        % (len(det_annos), total_pred_objects / max(1, len(det_annos)))
    )

    # --- Save results to JSON ---
    if getattr(args, "infer_time", False):
        logger.info("Average inference time: %.4f ms" % (infer_time_meter.avg))
    json_output_path = result_dir / "result.json"
    logger.info(f"Saving results to JSON: {json_output_path}")
    try:
        with open(json_output_path, "w") as f:
            # Use the custom NumpyEncoder
            json.dump(det_annos, f, indent=4, cls=NumpyEncoder)
        logger.info("Successfully saved results as JSON.")
    except TypeError as e:
        logger.error(f"JSON Serialization Error: {e}. Check if all numpy types are handled in NumpyEncoder.")
    except Exception as e:
        logger.error(f"Error writing JSON file: {e}")

    with open(result_dir / "result.pkl", "wb") as f:
        pickle.dump(det_annos, f)

    # Calculate Custom Metrics!

    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names, eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC, output_path=final_output_dir
    # )
    if getattr(args, "infer_time", False):
        logger.info("Average inference time: %.4f ms" % (infer_time_meter.avg))
        logger.info("Average inference time per example: %.4f ms" % Inference_time_per_example)
        logger.info("Average inference time per batch: %.4f ms" % infer_time_meter.avg)
        json_output_path = result_dir / "infer_time.json"
        logger.info(f"Saving results to JSON: {json_output_path}")
        Inference_time_per_example = ((infer_time_meter.avg * len(dataloader)) / len(dataloader.dataset))/1000
        try:
            with open(json_output_path, "w") as f:
                # Use the custom NumpyEncoder
                infer_time_dump = {
                    "avg_infer_time/batch": infer_time_meter.avg,
                    "Number of batches": len(dataloader),
                    "Number of datapoints": len(dataloader.dataset),
                    "sec_per_example": sec_per_example,
                    "Inference Time per example": Inference_time_per_example,
                    "Other": sec_per_example - Inference_time_per_example,
                    "GPU Name": torch.cuda.get_device_name(0),
                }
                json.dump(infer_time_dump, f, indent=4, cls=NumpyEncoder)
            logger.info("Successfully saved results as JSON.")
        except TypeError as e:
            logger.error(f"JSON Serialization Error: {e}. Check if all numpy types are handled in NumpyEncoder.")
        except Exception as e:
            logger.error(f"Error writing JSON file: {e}")
    json_output_path = result_dir / "metrics.json"
    logger.info(f"Saving results to JSON: {json_output_path}")
    # try:
    #     with open(json_output_path, "w") as f:
    #         # Use the custom NumpyEncoder
    #         json.dump(result_dict, f, indent=5, cls=NumpyEncoder)

    #     logger.info("Successfully saved results as JSON.")
    # except TypeError as e:
    #     logger.error(f"JSON Serialization Error: {e}. Check if all numpy types are handled in NumpyEncoder.")
    # except Exception as e:
    #     logger.error(f"Error writing JSON file: {e}")
    # logger.info(result_str)
    # ret_dict.update(result_dict)
    # logger.info("Result is saved to %s" % result_dir)

    logger.info("****************Evaluation done.*****************")
    return ret_dict

def eval_single_ckpt(cfg, model, dataloader, args, result_dir: Path, logger, epoch_id: str,
                       dist_test: bool = False, specified_ckpt_path: str = None):
    """
    Evaluates a single checkpoint.

    Args:
        cfg: Config object.
        model: The base model instance (e.g., model.module if DDP was used for training).
        dataloader: DataLoader for the evaluation dataset.
        args: Arguments from the calling script (should include 'save_to_file', 'infer_time' if used).
        result_dir: Path object for storing evaluation results.
        logger: Logger instance.
        epoch_id: A string identifier for the checkpoint being evaluated (e.g., "best_model", "epoch_80").
        dist_test: Boolean, True if evaluation should be distributed.
        specified_ckpt_path: Path to the checkpoint file to load.
    """
    assert specified_ckpt_path is not None, "A checkpoint path must be specified for eval_single_ckpt."
    assert Path(specified_ckpt_path).exists(), f"Checkpoint path does not exist: {specified_ckpt_path}"

    # 1. Load checkpoint
    logger.info(f"Loading checkpoint for evaluation: {specified_ckpt_path}")
    try:
        checkpoint = torch.load(specified_ckpt_path, map_location='cpu',weights_only=False)
        # Prioritize 'model_state', then 'state_dict', then the checkpoint itself
        ckpt_state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))

        unwrapped_state_dict = {}
        is_ddp_state_dict = any(k.startswith('module.') for k in ckpt_state_dict.keys())
        for k, v in ckpt_state_dict.items():
            if is_ddp_state_dict and k.startswith('module.'):
                unwrapped_state_dict[k[len('module.'):]] = v
            else:
                unwrapped_state_dict[k] = v

        # If the model is already on CUDA, ensure loaded state dict is also on the same device
        # or load to CPU first then model.load_state_dict, then model.to(device)
        # For simplicity, we assume model is correctly placed on device before this call or handled by DDP wrapping.
        model.load_state_dict(unwrapped_state_dict, strict=True)
        logger.info(f"Successfully loaded checkpoint from {specified_ckpt_path} into the model.")
    except Exception as e:
        logger.error(f"Error loading checkpoint {specified_ckpt_path}: {e}")
        raise e

    model.cuda() # Ensure model is on GPU after loading state dict (if not already)

    # --- Setup directories and metrics ---
    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / "final_result_data" # Renamed to avoid conflict if also saving per-epoch results

    # Use getattr for save_to_file as it might not always be in args from train.py
    if getattr(args, "save_to_file", False):
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {"gt_num": 0}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric["recall_roi_%s" % str(cur_thresh)] = 0
        metric["recall_rcnn_%s" % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    infer_time_meter = None
    if getattr(args, "infer_time", False):
        infer_time_meter = common_utils.AverageMeter()

    logger.info(f"*************** EVALUATION of CKPT '{epoch_id}' ({Path(specified_ckpt_path).name}) *****************")

    # --- Distributed Test Setup ---
    # If dist_test is True, the model (which is the base model) needs to be wrapped.
    # This differs slightly from eval_one_epoch where it's called during training loop and model might already be wrapped.
    eval_model = model
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        # find_unused_parameters might be needed depending on the model and evaluation task
        find_unused = cfg.get("FIND_UNUSED_PARAMETERS_EVAL", cfg.get("FIND_UNUSED_PARAMETERS", False))
        eval_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=find_unused
        )
    eval_model.eval()

    progress_bar = None
    if not dist_test or cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc=f"Eval Ckpt {epoch_id}", dynamic_ncols=True)

    loop_start_time = time.time()

    # --- Evaluation Loop ---
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        batch_infer_start_time = None
        if infer_time_meter:
            batch_infer_start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict_batch = eval_model(batch_dict)

        disp_dict = {}
        if infer_time_meter and batch_infer_start_time is not None:
            inference_time = time.time() - batch_infer_start_time
            infer_time_meter.update(inference_time * 1000)
            disp_dict['infer_time'] = f"{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})"

        statistics_info(cfg, ret_dict_batch, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if getattr(args, "save_to_file", False) else None
        )
        det_annos.extend(annos) # Use extend

        if progress_bar:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if progress_bar:
        progress_bar.close()

    # --- Merge Results if Distributed ---
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / f"tmpdir_annos_{epoch_id}")

        # Metric is a dict. Wrap in list for merging, then aggregate.
        merged_metrics_list = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / f"tmpdir_metric_{epoch_id}")

        if cfg.LOCAL_RANK == 0:
            # Aggregate metrics from all ranks
            aggregated_metric = {"gt_num": 0} # Initialize with all expected keys
            for cur_thresh_val in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
                aggregated_metric["recall_roi_%s" % str(cur_thresh_val)] = 0
                aggregated_metric["recall_rcnn_%s" % str(cur_thresh_val)] = 0

            for m_dict in merged_metrics_list:
                for key, val in m_dict.items():
                    if key in aggregated_metric:
                         aggregated_metric[key] += val
                    else: # Should not happen if initialized correctly
                        aggregated_metric[key] = val
            metric = aggregated_metric
        else: # Non-master ranks in dist_test mode exit here
            return {}

    # --- Final Calculations and Logging (on rank 0 or if not dist_test) ---
    logger.info(f"*************** Performance of CKPT '{epoch_id}' *****************")
    sec_per_example = (time.time() - loop_start_time) / max(1, len(dataloader.dataset))
    logger.info(f"Evaluation finished (sec_per_example: {sec_per_example:.4f}).")

    if infer_time_meter:
        logger.info(f"Average inference time per batch: {infer_time_meter.avg:.4f} ms")

    final_ret_dict = {} # Stores recall and official evaluation metrics
    gt_num_cnt = metric["gt_num"]
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric["recall_roi_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric["recall_rcnn_%s" % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info(f"recall_roi_{cur_thresh}: {cur_roi_recall:.6f}")
        logger.info(f"recall_rcnn_{cur_thresh}: {cur_rcnn_recall:.6f}")
        final_ret_dict[f"recall/roi_{cur_thresh}"] = cur_roi_recall
        final_ret_dict[f"recall/rcnn_{cur_thresh}"] = cur_rcnn_recall

    total_pred_objects = sum(len(anno["name"]) for anno in det_annos)
    logger.info(
        f"Average predicted number of objects ({len(det_annos)} samples): {total_pred_objects / max(1, len(det_annos)):.3f}"
    )

    # --- Save Detection Results (JSON and PKL) ---
    if getattr(args, "save_to_file", True): # Default to True for eval_single_ckpt
        det_json_path = result_dir / f"detections_{epoch_id}.json"
        det_pkl_path = result_dir / f"detections_{epoch_id}.pkl"
        logger.info(f"Saving detection results to {det_json_path} and {det_pkl_path}")
        try:
            with open(det_json_path, "w") as f:
                json.dump(det_annos, f, indent=4, cls=NumpyEncoder)
            with open(det_pkl_path, "wb") as f:
                pickle.dump(det_annos, f)
            logger.info("Successfully saved detection results.")
        except Exception as e:
            logger.error(f"Error saving detection results: {e}")

    # --- Official Dataset Evaluation (e.g., mAP) ---
    eval_metric_dict = {}
    if hasattr(dataset, 'evaluation') and callable(dataset.evaluation):
        logger.info(f"Calculating official evaluation metrics for '{epoch_id}' using dataset.evaluation...")
        dataset_eval_output_dir = result_dir / f"official_eval_output_{epoch_id}"
        dataset_eval_output_dir.mkdir(parents=True, exist_ok=True)

        eval_output = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=dataset_eval_output_dir
        )

        result_str = "Official evaluation performed."
        if isinstance(eval_output, tuple) and len(eval_output) == 2:
            result_str, eval_metric_dict = eval_output
        elif isinstance(eval_output, dict):
            eval_metric_dict = eval_output
            result_str += "\n" + "\n".join([f"  {k}: {v}" for k, v in eval_metric_dict.items()])
        else:
            logger.warning("dataset.evaluation returned an unexpected format.")
            result_str = "Dataset evaluation performed, but metrics format not recognized."

        logger.info(result_str)
        final_ret_dict.update(eval_metric_dict)

        metrics_json_path = result_dir / f"official_metrics_{epoch_id}.json"
        try:
            with open(metrics_json_path, "w") as f:
                json.dump(eval_metric_dict, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Successfully saved official metrics to {metrics_json_path}")
        except Exception as e:
            logger.error(f"Error saving official metrics JSON: {e}")
    else:
        logger.warning("Dataset object does not have 'evaluation' method. Skipping official evaluation.")

    # --- Save Inference Time Summary ---
    if infer_time_meter:
        avg_infer_ms_batch = infer_time_meter.avg
        num_batches = len(dataloader)
        num_datapoints = len(dataloader.dataset)
        avg_infer_ms_example = (avg_infer_ms_batch * num_batches) / max(1, num_datapoints)

        infer_time_summary = {
            "avg_inference_time_ms_per_batch": avg_infer_ms_batch,
            "avg_inference_time_ms_per_example": avg_infer_ms_example,
            "total_batches_processed": num_batches,
            "total_datapoints_processed": num_datapoints,
            "total_loop_sec_per_example": sec_per_example, # Includes data loading, model inference, post-processing
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        infer_time_json_path = result_dir / f"inference_time_summary_{epoch_id}.json"
        try:
            with open(infer_time_json_path, "w") as f:
                json.dump(infer_time_summary, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Successfully saved inference time summary to {infer_time_json_path}")
        except Exception as e:
            logger.error(f"Error writing inference time summary: {e}")

    logger.info(f"**************** Evaluation of CKPT '{epoch_id}' done. Results in: {result_dir} *****************")
    return final_ret_dict

if __name__ == "__main__":
    pass
