import copy
import numpy as np
import torch
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# --- Dependency: pcdet.ops.iou3d_nms.iou3d_nms_utils.boxes_iou3d_gpu ---
try:
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
    _BOXES_IOU3D_GPU_AVAILABLE = True
except ImportError:
    _BOXES_IOU3D_GPU_AVAILABLE = False
    # Define a placeholder function that will raise an error if called.
    # The calculate_pr_data_gpu function has its own checks and error messages
    # if GPU IoU is attempted without this function being available.
    def boxes_iou3d_gpu(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor: # type: ignore
        print("Critical Error: `boxes_iou3d_gpu` is not available. Install pcdet ops correctly.")
        raise NotImplementedError(
            "GPU 3D IoU function (boxes_iou3d_gpu) is not available. "
            "Please ensure pcdet.ops are compiled successfully."
        )

# This global variable will be used by calculate_pr_data_gpu
USE_GPU_IOU = _BOXES_IOU3D_GPU_AVAILABLE

# --- Helper: Angle Normalization (as used in calculate_pr_data_gpu) ---
def normalize_angle_delta(delta: float) -> float:
    """Normalize angle difference to [-pi, pi)."""
    # Add 2*pi to ensure positive, then modulo 2*pi, then shift by -pi
    return (delta + np.pi) % (2 * np.pi) - np.pi

# --- Core Logic: calculate_pr_data_gpu (Provided Code) ---
# This function is provided in the problem description.
# For brevity, I will assume it's defined in this scope.
# Pasting the full function here:
def calculate_pr_data_gpu(
    model_predictions_lists: List[List[Dict[str, Any]]],
    gt_annos: List[Dict[str, Any]],  # These should already be filtered if needed
    model_names: List[str],
    classes_to_evaluate: List[str],
    iou_threshold: float,
    device=None,
    min_score_threshold: float = 0.01,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Calculates data (scores, TP/FP status, orientation similarity, total GT)
    for plotting PR curves and calculating AOS for multiple models at a
    specific IoU threshold.

    Assumes gt_annos and model_predictions_lists ONLY contain data for the
    frames to be evaluated (e.g., the validation set).

    Args:
        model_predictions_lists: List of prediction lists (filtered) for each model.
        gt_annos: List of ground truth annotations (filtered).
        model_names: List of model names.
        classes_to_evaluate: List of class names to evaluate.
        iou_threshold: The 3D IoU threshold for considering a match (TP).
        device: Target computation device (e.g., torch.device("cuda:0")).
        min_score_threshold: Minimal score to consider a detection.

    Returns:
        Dict: {model_name: {class_name: {'scores': np.array,
                                        'match_status': np.array (1=TP, 0=FP),
                                        'orientation_similarity': np.array (value for TP, 0 for FP),
                                        'total_gt': int}}}
              Returns empty dict on critical error.
    """
    # --- Input validation and Device Setup ---
    if not USE_GPU_IOU and device is not None and "cuda" in str(device):
        print("Error: GPU device specified but GPU IoU function is not available (USE_GPU_IOU is False). Cannot proceed.")
        return {}
    if not isinstance(model_predictions_lists, list) or not isinstance(gt_annos, list):
        print("Error: model_predictions_lists and gt_annos must be lists.")
        return {}

    if len(model_predictions_lists) != len(model_names):
        print(
            f"Error: Mismatch between prediction lists ({len(model_predictions_lists)}) and model names ({len(model_names)})."
        )
        return {}

    if device is None:
        if torch.cuda.is_available() and USE_GPU_IOU:
            device = torch.device("cuda:0")
            print(f"Using automatically selected CUDA device: {device}")
        else:
            if not USE_GPU_IOU and torch.cuda.is_available():
                print("Warning: CUDA is available, but USE_GPU_IOU is False (pcdet.ops.iou3d_nms.iou3d_nms_utils.boxes_iou3d_gpu import failed).")
                print("Proceeding with placeholder CPU IoU. Results will NOT be accurate if a real CPU IoU is not implemented in boxes_iou3d_gpu.")
            elif not USE_GPU_IOU: # CUDA not available, USE_GPU_IOU is False
                 print("Warning: CUDA not available and USE_GPU_IOU is False. Ensure a CPU IoU is available or results will be inaccurate.")
            else: # CUDA not available, but USE_GPU_IOU is somehow True (should not happen with current setup)
                 print("Warning: CUDA not available, using CPU for tensor placement. GPU IoU function might fail if it strictly requires CUDA.")
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f"Using device for PR/AOS data calculation: {device}")

    num_models = len(model_names)
    num_gt_samples = len(gt_annos)
    print(f"Calculating PR/AOS data for {num_models} models, {num_gt_samples} GT frames...")
    print(f"Classes: {classes_to_evaluate}, IoU Threshold: {iou_threshold}, Min Score: {min_score_threshold}")
    start_time = time.time()

    pr_data = {
        m_name: {
            c_name: {"scores": [], "match_status": [], "orientation_similarity": [], "total_gt": 0}
            for c_name in classes_to_evaluate
        }
        for m_name in model_names
    }
    temp_scores = {m: {c: [] for c in classes_to_evaluate} for m in model_names}
    temp_matches = {m: {c: [] for c in classes_to_evaluate} for m in model_names}
    temp_similarity = {m: {c: [] for c in classes_to_evaluate} for m in model_names}

    model_preds_by_frame = defaultdict(dict)
    for model_idx, model_name in enumerate(model_names):
        pred_list = model_predictions_lists[model_idx]
        if not isinstance(pred_list, list):
            print(f"Internal Error: Predictions for model '{model_name}' is not a list.")
            continue

        frames_processed_for_model = set()
        for pred_anno in pred_list:
            frame_id = pred_anno.get("frame_id")
            if frame_id is not None:
                if frame_id in frames_processed_for_model:
                    # print(
                    #     f"Warning: Duplicate frame_id '{frame_id}' found within predictions for model '{model_name}'. Keeping first."
                    # ) # This can be verbose if predictions are per-object not per-frame
                    continue
                boxes_lidar = pred_anno.get("boxes_lidar", np.array([]))
                if isinstance(boxes_lidar, torch.Tensor): # Ensure numpy for shape check
                    boxes_lidar = boxes_lidar.cpu().numpy()

                if "boxes_lidar" in pred_anno and boxes_lidar.shape[-1] == 7:
                    model_preds_by_frame[model_name][frame_id] = pred_anno
                    frames_processed_for_model.add(frame_id) # This logic assumes one pred_anno per frame_id
                # else: # Allow frames with no boxes for a given model
                #     box_shape = boxes_lidar.shape
                #     print(
                #         f"Warning: Skipping prediction for frame {frame_id} in model {model_name} due to missing/invalid boxes_lidar (shape {box_shape}, expected N x 7)."
                #     )
            # else:
            #     print("Warning: Prediction annotation found without frame_id during PR calculation. Skipping.")

    print("Calculating total ground truth counts per class...")
    global_total_gt = {c_name: 0 for c_name in classes_to_evaluate}
    gt_frame_ids_set = set()
    for gt_anno in gt_annos:
        frame_id = gt_anno.get("frame_id")
        if frame_id is None:
            continue
        gt_frame_ids_set.add(frame_id)
        gt_names = gt_anno.get("name", np.array([]))
        gt_boxes = gt_anno.get("gt_boxes_lidar", np.zeros((0, 7)))
        if isinstance(gt_boxes, torch.Tensor): # Ensure numpy
            gt_boxes = gt_boxes.cpu().numpy()

        if gt_boxes.shape[-1] != 7 and gt_boxes.size > 0:
            print(
                f"ERROR: GT boxes for frame {frame_id} do not have 7 elements (shape {gt_boxes.shape}). Skipping frame for GT count."
            )
            continue
        if isinstance(gt_names, np.ndarray):
            for c_name in classes_to_evaluate:
                global_total_gt[c_name] += np.sum(gt_names == c_name)

    for model_name in model_names:
        for c_name in classes_to_evaluate:
            pr_data[model_name][c_name]["total_gt"] = global_total_gt[c_name]
            if global_total_gt[c_name] == 0:
                print(
                    f"Warning: No ground truth found for class '{c_name}'. PR/AOS cannot be calculated."
                )

    print(f"Total GT counts: {global_total_gt}")
    print(f"Unique GT frame IDs found: {len(gt_frame_ids_set)}")

    print("Matching predictions to ground truth...")
    processed_frame_count = 0
    gt_frames_without_preds_in_val = {m: 0 for m in model_names}

    for sample_idx, gt_anno in enumerate(gt_annos):
        gt_frame_id = gt_anno.get("frame_id")
        if gt_frame_id is None:
            continue
        processed_frame_count += 1

        gt_names_np = gt_anno.get("name", np.array([], dtype=str))
        gt_boxes_np = gt_anno.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32))
        if isinstance(gt_boxes_np, torch.Tensor): gt_boxes_np = gt_boxes_np.cpu().numpy()
        if isinstance(gt_names_np, list): gt_names_np = np.array(gt_names_np)


        if gt_boxes_np.ndim != 2 or (gt_boxes_np.shape[1] != 7 and gt_boxes_np.size > 0):
            print(
                f"ERROR: Invalid GT boxes shape {gt_boxes_np.shape} for frame {gt_frame_id}. Must be (N, 7). Skipping frame."
            )
            continue
        if gt_boxes_np.size == 0: # Ensure it's (0,7) if empty
            gt_boxes_np = np.zeros((0, 7), dtype=np.float32)


        for model_name in model_names:
            dt_anno = model_preds_by_frame[model_name].get(gt_frame_id)

            if dt_anno is None:
                gt_frames_without_preds_in_val[model_name] += 1
                # If no DT for this frame, all GTs are FN for this frame.
                # Detections for this frame from this model are effectively zero.
                # AP calculation naturally handles this (no TPs or FPs from this frame for this model).
                # We still need to account for all detections from other frames for this model.
                continue # Skips to next model for this GT frame

            dt_names_np = dt_anno.get("name", np.array([], dtype=str))
            dt_boxes_np = dt_anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))
            dt_scores_np = dt_anno.get("score", np.array([], dtype=np.float32))

            if isinstance(dt_boxes_np, torch.Tensor): dt_boxes_np = dt_boxes_np.cpu().numpy()
            if isinstance(dt_names_np, list): dt_names_np = np.array(dt_names_np)
            if isinstance(dt_scores_np, list): dt_scores_np = np.array(dt_scores_np)


            if dt_boxes_np.ndim != 2 or (dt_boxes_np.shape[1] != 7 and dt_boxes_np.size > 0):
                print(
                    f"ERROR: Invalid DT boxes shape {dt_boxes_np.shape} for model {model_name}, frame {gt_frame_id}. Must be (N, 7). Skipping."
                )
                continue # Skips this frame for this model
            if dt_boxes_np.size == 0: # Ensure (0,7) if empty
                 dt_boxes_np = np.zeros((0, 7), dtype=np.float32)
            if dt_scores_np.ndim != 1:
                dt_scores_np = dt_scores_np.flatten()


            if min_score_threshold > 0 and dt_scores_np.size > 0:
                score_mask = dt_scores_np >= min_score_threshold
                dt_names_np = dt_names_np[score_mask]
                dt_boxes_np = dt_boxes_np[score_mask]
                dt_scores_np = dt_scores_np[score_mask]

            # If after filtering, no detections remain for this frame, effectively dt_boxes_np is empty
            if dt_scores_np.size == 0:
                dt_names_np = np.array([])
                dt_boxes_np = np.zeros((0, 7), dtype=np.float32)
                # dt_scores_np is already empty

            for class_name in classes_to_evaluate:
                if global_total_gt[class_name] == 0:
                    continue

                gt_mask = gt_names_np == class_name
                dt_mask = dt_names_np == class_name

                cur_gt_boxes_np = gt_boxes_np[gt_mask]
                cur_dt_boxes_np = dt_boxes_np[dt_mask]
                cur_dt_scores_np = dt_scores_np[dt_mask]
                num_gt = cur_gt_boxes_np.shape[0]
                num_dt = cur_dt_boxes_np.shape[0]

                if num_dt > 0:
                    temp_scores[model_name][class_name].extend(cur_dt_scores_np.tolist())

                match_status_for_frame = np.zeros(num_dt, dtype=np.int8)
                similarity_for_frame = np.zeros(num_dt, dtype=np.float32)

                if num_gt > 0 and num_dt > 0:
                    try:
                        cur_gt_boxes_torch = torch.from_numpy(cur_gt_boxes_np).float().to(device)
                        cur_dt_boxes_torch = torch.from_numpy(cur_dt_boxes_np).float().to(device)

                        # This is where boxes_iou3d_gpu is called
                        iou_matrix_gpu = boxes_iou3d_gpu(cur_dt_boxes_torch, cur_gt_boxes_torch)
                        iou_matrix = iou_matrix_gpu.cpu().numpy()

                        gt_matched = np.zeros(num_gt, dtype=bool)
                        sorted_dt_indices = np.argsort(-cur_dt_scores_np) # Sort by score desc

                        for original_dt_idx in sorted_dt_indices:
                            ious_for_this_dt = iou_matrix[original_dt_idx, :]

                            valid_gt_indices = np.where((ious_for_this_dt >= iou_threshold) & (~gt_matched))[0]

                            if len(valid_gt_indices) > 0:
                                best_relative_idx = np.argmax(ious_for_this_dt[valid_gt_indices])
                                best_gt_match_idx = valid_gt_indices[best_relative_idx]

                                match_status_for_frame[original_dt_idx] = 1 # TP
                                gt_matched[best_gt_match_idx] = True

                                gt_heading = cur_gt_boxes_np[best_gt_match_idx, 6]
                                dt_heading = cur_dt_boxes_np[original_dt_idx, 6]
                                delta_theta = normalize_angle_delta(dt_heading - gt_heading)
                                similarity = (1.0 + np.cos(delta_theta)) / 2.0
                                similarity_for_frame[original_dt_idx] = similarity
                            # Else: No suitable unmatched GT found, detection remains FP (status=0, similarity=0)
                    except Exception as e:
                        print(
                            f"\nError during IoU/matching for frame {gt_frame_id}, model {model_name}, class {class_name}: {e}"
                        )
                        if "boxes_iou3d_gpu" in str(e) and "not available" in str(e): # Propagate critical error
                            return {}
                        match_status_for_frame = np.zeros(num_dt, dtype=np.int8) # Treat as all FP on error
                        similarity_for_frame = np.zeros(num_dt, dtype=np.float32)

                temp_matches[model_name][class_name].extend(match_status_for_frame.tolist())
                temp_similarity[model_name][class_name].extend(similarity_for_frame.tolist())

    print("Finalizing PR/AOS data arrays...")
    total_detections_processed = 0
    for model_name in model_names:
        for class_name in classes_to_evaluate:
            scores_list = temp_scores[model_name][class_name]
            matches_list = temp_matches[model_name][class_name]
            similarity_list = temp_similarity[model_name][class_name]

            if not (len(scores_list) == len(matches_list) == len(similarity_list)):
                print(f"FATAL INTERNAL ERROR: Mismatch in collected array lengths for {model_name}/{class_name}.")
                print(
                    f"  Scores: {len(scores_list)}, Matches: {len(matches_list)}, Similarities: {len(similarity_list)}"
                )
                return {}

            pr_data[model_name][class_name]["scores"] = np.array(scores_list, dtype=np.float32)
            pr_data[model_name][class_name]["match_status"] = np.array(matches_list, dtype=np.int8)
            pr_data[model_name][class_name]["orientation_similarity"] = np.array(similarity_list, dtype=np.float32)
            total_detections_processed += len(scores_list)

    elapsed_time = time.time() - start_time
    print(f"PR/AOS data calculation finished in {elapsed_time:.2f} seconds.")
    print(f"Processed {processed_frame_count} GT frames.")
    print(f"Total detections considered (score >= {min_score_threshold}): {total_detections_processed}")
    for model_name in model_names:
        if model_name in gt_frames_without_preds_in_val:
            print(
                f"  Model '{model_name}': No predictions found for {gt_frames_without_preds_in_val[model_name]} GT frames."
            )

    if total_detections_processed == 0 and all(gt == 0 for gt in global_total_gt.values()):
        print("Warning: No detections passed threshold and/or no GT objects found.")

    return pr_data


# --- Helper: Metric Calculation (AP/AOS) ---
def compute_average_interpolated_metric(
    recall_levels_sorted_by_score: np.ndarray,
    metric_values_sorted_by_score: np.ndarray,
    num_sample_pts: int = 11
) -> float:
    """
    Calculates the average of a metric, interpolated at specific recall levels.
    This is used for AP and AOS calculation in KITTI style.

    Args:
        recall_levels_sorted_by_score (np.ndarray): Recall values for detections, sorted by score (descending).
                                       Shape (N_det,).
        metric_values_sorted_by_score (np.ndarray): Metric values (e.g., precision or
                                                  orientation similarity) for detections,
                                                  sorted by score (descending). Shape (N_det,).
                                                  For AP, this is precision. For AOS, this is (1+cos)/2 for TPs, 0 for FPs.
        num_sample_pts (int): Number of recall points to sample (e.g., 11 for VOC, 41 for R40).

    Returns:
        float: The calculated average metric.
    """
    if metric_values_sorted_by_score.size == 0 or recall_levels_sorted_by_score.size == 0:
        return 0.0

    recall_sample_points = np.linspace(0.0, 1.0, num_sample_pts, endpoint=True)
    average_metric = 0.0

    for r_thresh in recall_sample_points:
        # Find metric values where recall is >= r_thresh
        # Note: recall_levels_sorted_by_score is already sorted by score,
        # so taking elements where recall >= r_thresh corresponds to KITTI's method.
        relevant_indices = np.where(recall_levels_sorted_by_score >= r_thresh)[0]
        if relevant_indices.size > 0:
            # Maximize the metric at this recall level (KITTI smoothing for precision/metric)
            # This means for a given recall threshold r_th, we consider all detections d'
            # with recall(d') >= r_th, and take the maximum of metric(d').
            average_metric += np.max(metric_values_sorted_by_score[relevant_indices])
        # If no detections satisfy recall >= r_thresh, their contribution to sum is 0.

    return average_metric / num_sample_pts


# --- Main Function to Implement ---
def get_official_eval_result(
    gt_annos: List[Dict[str, Any]],
    dt_annos: List[Dict[str, Any]],
    current_classes: List[str],
    iou_threshold: float = 0.25,
    min_score_threshold: float = 0.01,
    num_pr_points: int = 41, # Standard for OpenPCDet (R40)
    device_str: str = None # e.g., "cuda:0" or "cpu", None for auto
) -> Tuple[str, Dict[str, Any]]:
    """
    Calculates custom evaluation metrics (AP and AOS) using the provided PR data calculation function.

    Args:
        gt_annos: List of ground truth annotations. Each element is a dict for a frame,
                  expected to contain 'frame_id', 'name' (np.array), 'gt_boxes_lidar' (np.array Nx7).
        dt_annos: List of detection annotations. Each element is a dict for a frame,
                  expected to contain 'frame_id', 'name' (np.array), 'boxes_lidar' (np.array Mx7),
                  'score' (np.array M).
        current_classes: List of class names to evaluate.
        iou_threshold: The 3D IoU threshold for considering a match (TP). Applied to all classes.
        min_score_threshold: Minimal score to consider a detection.
        num_pr_points: Number of points for PR curve interpolation (e.g., 11 for VOC, 41 for R40).
        device_str: Target computation device string (e.g., "cuda:0", "cpu"). Auto-selects if None.

    Returns:
        Tuple[str, Dict[str, Any]]:
            - ap_result_str: A string summarizing the AP and AOS results.
            - ap_dict: A dictionary containing detailed AP and AOS per class, and mAP/mAOS.
                       Format: {'ClassName': {'AP': float, 'AOS': float, 'total_gt': int, ...},
                                'mAP': float, 'mAOS': float}
    """
    # Ensure frame_id is present in annos or add them.
    # Assuming frame_id is already part of the annotation dicts as calculate_pr_data_gpu relies on it.
    # If not, they would need to be added, e.g.:
    # for i, anno in enumerate(gt_annos): anno.setdefault('frame_id', str(i))
    # for i, anno in enumerate(dt_annos): anno.setdefault('frame_id', str(i))
    # This step is typically handled by the dataset loader.

    model_name = "custom_eval_model" # Placeholder for the single set of predictions
    model_predictions_lists = [dt_annos] # calculate_pr_data_gpu expects a list of model prediction lists
    model_names = [model_name]

    # Determine computation device
    if device_str is None:
        if torch.cuda.is_available() and USE_GPU_IOU:
            # Let calculate_pr_data_gpu pick the default CUDA device if device is None
            device = None
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    # --- Call the core PR data calculation function ---
    pr_data_all_models = calculate_pr_data_gpu(
        model_predictions_lists=model_predictions_lists,
        gt_annos=gt_annos,
        model_names=model_names,
        classes_to_evaluate=current_classes,
        iou_threshold=iou_threshold,
        device=device, # Pass the torch.device object or None
        min_score_threshold=min_score_threshold,
    )

    if not pr_data_all_models: # Empty dict means a critical error occurred in calculate_pr_data_gpu
        error_msg = "Error: PR data calculation failed. Check logs from calculate_pr_data_gpu."
        if not USE_GPU_IOU and (device_str is None or "cuda" in device_str):
             error_msg += " This might be due to attempting GPU IoU when pcdet ops are not available."
        return error_msg, {}

    # Extract data for our single "model"
    pr_data_single_model = pr_data_all_models[model_name]

    # --- Calculate AP and AOS for each class ---
    ap_dict: Dict[str, Any] = {}
    ap_result_lines = [
        f"Custom Evaluation Results (IoU thresh: {iou_threshold}, PR points: {num_pr_points}, Min score: {min_score_threshold}):"
    ]

    overall_ap_values = []
    overall_aos_values = []

    for class_name in current_classes:
        class_results = pr_data_single_model.get(class_name)

        if not class_results:
            print(f"Warning: No PR data retrieved for class '{class_name}'. Skipping calculation for this class.")
            ap_dict[class_name] = {"AP": 0.0, "AOS": 0.0, "total_gt": 0, "total_dt": 0, "note": "No PR data found"}
            ap_result_lines.append(f"  {class_name} AP:  0.0000 (Error: No PR data)")
            ap_result_lines.append(f"  {class_name} AOS: 0.0000 (Error: No PR data)")
            continue

        scores = class_results["scores"]
        match_status = class_results["match_status"]  # 1=TP, 0=FP
        orientation_similarity = class_results["orientation_similarity"] # (1+cos)/2 for TP, 0 for FP
        total_gt = class_results["total_gt"]
        num_dt = len(scores)

        ap_dict_class: Dict[str, Any] = {"total_gt": total_gt, "total_dt": num_dt}
        current_ap_value = 0.0
        current_aos_value = 0.0

        if total_gt == 0:
            ap_dict_class.update({"AP": 0.0, "AOS": 0.0, "note": "No GT objects"})
            ap_result_lines.append(f"  {class_name} AP:  0.0000 (0 GTs)")
            ap_result_lines.append(f"  {class_name} AOS: 0.0000 (0 GTs)")
        elif num_dt == 0:
            # No detections means TP=0, FP=0. AP and AOS are 0.
            ap_dict_class.update({"AP": 0.0, "AOS": 0.0, "note": "No detections passing score threshold"})
            ap_result_lines.append(f"  {class_name} AP:  0.0000 (0 Dets)")
            ap_result_lines.append(f"  {class_name} AOS: 0.0000 (0 Dets)")
        else:
            # Sort detections by score (descending)
            sorted_indices = np.argsort(-scores)
            # sorted_scores = scores[sorted_indices] # Not directly used after sorting for metric calculation

            # These are now sorted by score:
            sorted_match_status = match_status[sorted_indices] # 1 for TP, 0 for FP
            sorted_orientation_similarity = orientation_similarity[sorted_indices] # (1+cos)/2 for TPs, 0 for FPs

            # Calculate cumulative TP and FP
            tp_cumsum = np.cumsum(sorted_match_status).astype(np.float32)
            fp_cumsum = np.cumsum(1 - sorted_match_status).astype(np.float32) # (1 - 0) = 1 for FP, (1-1)=0 for TP

            # Calculate precision and recall arrays (sorted by score)
            precision_sorted_by_score = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall_sorted_by_score = tp_cumsum / total_gt

            # AP Calculation
            # The metric being averaged for AP is precision.
            current_ap_value = compute_average_interpolated_metric(
                recall_levels_sorted_by_score=recall_sorted_by_score,
                metric_values_sorted_by_score=precision_sorted_by_score,
                num_sample_pts=num_pr_points
            )

            # AOS Calculation
            # The metric being averaged for AOS is orientation similarity (already 0 for FPs).
            current_aos_value = compute_average_interpolated_metric(
                recall_levels_sorted_by_score=recall_sorted_by_score,
                metric_values_sorted_by_score=sorted_orientation_similarity,
                num_sample_pts=num_pr_points
            )

            ap_dict_class.update({"AP": current_ap_value, "AOS": current_aos_value})
            ap_result_lines.append(f"  {class_name} AP:  {current_ap_value:.4f}")
            ap_result_lines.append(f"  {class_name} AOS: {current_aos_value:.4f}")

        ap_dict[class_name] = ap_dict_class
        if total_gt > 0: # Only include in mAP/mAOS if class has GT objects
            overall_ap_values.append(current_ap_value)
            overall_aos_values.append(current_aos_value)

    # Calculate mAP and mAOS
    if overall_ap_values: # Check if list is not empty
        mean_ap = np.mean(overall_ap_values)
        ap_result_lines.append(f"  Overall mAP: {mean_ap:.4f}")
        ap_dict["mAP"] = mean_ap
    else:
        ap_dict["mAP"] = 0.0
        ap_result_lines.append(f"  Overall mAP: 0.0000 (No classes with GTs or all GTs=0)")


    if overall_aos_values: # Check if list is not empty
        mean_aos = np.mean(overall_aos_values)
        ap_result_lines.append(f"  Overall mAOS: {mean_aos:.4f}")
        ap_dict["mAOS"] = mean_aos
    else:
        ap_dict["mAOS"] = 0.0
        ap_result_lines.append(f"  Overall mAOS: 0.0000 (No classes with GTs or all GTs=0)")


    ap_result_str = "\n".join(ap_result_lines)
    return ap_result_str, ap_dict
