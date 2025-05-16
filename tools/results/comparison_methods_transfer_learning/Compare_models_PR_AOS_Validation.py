# --- Imports (no changes needed here) ---
import glob
import json
import numbers
import os
import sys  # Added for error exit
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set  # Added Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# --- Dependency: IoU/NMS Functions ---
try:
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

    print("Successfully imported GPU IoU function (boxes_iou3d_gpu).")
    USE_GPU_IOU = True
except ImportError:
    print("--------------------------------------------------------------------")
    print("ERROR: Could not import 'boxes_iou3d_gpu' from 'pcdet'.")
    print("Falling back to a placeholder CPU IoU for demonstration purposes.")
    print("WARNING: Results calculated without GPU IoU will NOT be accurate.")
    print("--------------------------------------------------------------------")
    USE_GPU_IOU = False

    def boxes_iou3d_gpu(boxes1_tensor: torch.Tensor, boxes2_tensor: torch.Tensor) -> torch.Tensor:
        """Placeholder 3D IoU calculation (NOT ACCURATE)."""
        # This is a dummy implementation and should not be used for real evaluation
        print("Warning: Using placeholder CPU IoU function. Results are illustrative only.")
        num1 = boxes1_tensor.shape[0]
        num2 = boxes2_tensor.shape[0]
        # Return small random values to avoid division by zero issues downstream
        # but emphasize this isn't a real calculation.
        return torch.rand((num1, num2), device=boxes1_tensor.device, dtype=torch.float32) * 0.01


# --- Helper function (get_split_parts - potentially unused but kept) ---
def get_split_parts(num, num_parts):
    """Splits a number into roughly equal parts."""
    part_len = num // num_parts + (1 if num % num_parts > 0 else 0)
    parts = []
    remaining = num
    for _ in range(num_parts):
        current_part = min(part_len, remaining)
        if current_part <= 0:
            break
        parts.append(current_part)
        remaining -= current_part
    if sum(parts) != num and num > 0:
        diff = num - sum(parts)
        if parts:
            parts[-1] += diff
        else:
            return [num] if num > 0 else []
    parts = [p for p in parts if p > 0]
    return parts


# --- NEW: Function to load validation set IDs ---
def load_validation_ids(split_file_path: str) -> Optional[Set[str]]:
    """Loads frame IDs from a split file (e.g., val.txt).

    Args:
        split_file_path (str): Path to the text file containing frame IDs,
                               one per line.

    Returns:
        Optional[Set[str]]: A set of frame IDs from the file, or None if
                            the file cannot be read.
    """
    if not os.path.exists(split_file_path):
        print(f"Error: Validation split file not found: {split_file_path}")
        return None
    try:
        with open(split_file_path, "r") as f:
            frame_ids = {line.strip() for line in f if line.strip()}
        if not frame_ids:
            print(f"Warning: Validation split file '{split_file_path}' is empty.")
        print(f"Loaded {len(frame_ids)} frame IDs from validation split file: {split_file_path}")
        return frame_ids
    except Exception as e:
        print(f"Error reading validation split file {split_file_path}: {e}")
        return None


# --- Data Loading Functions (Modified to accept optional validation_ids) ---
def load_model_results_from_json(
    json_file_path: str,
    validation_ids: Optional[Set[str]] = None,  # Added optional argument
) -> List[Dict[str, Any]]:
    """Loads model prediction results from a JSON file, optionally filtering by validation IDs.

    Args:
        json_file_path (str): Path to the input JSON file.
        validation_ids (Optional[Set[str]]): If provided, only load frames whose
                                              frame_id is in this set.

    Returns:
        List[Dict[str, Any]]: List of prediction dictionaries, one per frame.
                               Contains keys: 'name', 'score', 'boxes_lidar', 'frame_id'.
                               Returns an empty list on error or if no valid frames found.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found: {json_file_path}")
        return []

    try:
        with open(json_file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {json_file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        return []

    if not isinstance(data, list):
        print(f"Error: Expected JSON file to contain a list of frame results, but got {type(data)}")
        return []

    all_predictions: List[Dict[str, Any]] = []
    processed_frame_ids = set()
    original_count = len(data)
    filtered_count = 0

    for frame_idx, frame_data in enumerate(data):
        if not isinstance(frame_data, dict):
            # print(f"Warning: Skipping item at index {frame_idx} in JSON list as it's not a dictionary: {frame_data}")
            continue

        frame_id = frame_data.get("frame_id")

        # --- Filtering Step ---
        if frame_id is None:
            # print(f"Warning: Skipping frame data entry at index {frame_idx} due to missing 'frame_id'.")
            continue
        if validation_ids is not None and frame_id not in validation_ids:
            filtered_count += 1
            continue  # Skip this frame if not in the validation set
        # --- End Filtering Step ---

        # Extract necessary fields using .get() for safety
        names_list = frame_data.get("name")
        scores_list = frame_data.get("score")
        boxes_list = frame_data.get("boxes_lidar")  # Should contain [x, y, z, dx, dy, dz, heading]
        pred_labels = frame_data.get("pred_labels")

        # Basic validation
        if not isinstance(names_list, list) or not isinstance(scores_list, list) or not isinstance(boxes_list, list):
            # print(f"Warning: Skipping frame_id '{frame_id}' due to non-list type for name/score/boxes_lidar.")
            continue

        if frame_id in processed_frame_ids:
            # print(f"Warning: Duplicate frame_id '{frame_id}' found in JSON (within validation set). Keeping first.")
            continue  # Use continue instead of just printing to avoid adding duplicate frame
        processed_frame_ids.add(frame_id)  # Add here after validation checks

        num_detections = len(names_list)
        if not (len(scores_list) == num_detections and len(boxes_list) == num_detections):
            # print(
            #     f"Warning: Skipping frame_id '{frame_id}' due to inconsistent lengths: "
            #     f"names ({len(names_list)}), scores ({len(scores_list)}), boxes ({len(boxes_list)})."
            # )
            continue

        # Convert to NumPy arrays
        try:
            if num_detections > 0:
                np_names = np.array(names_list, dtype=object)
                np_scores = np.array(scores_list, dtype=np.float32)
                np_boxes = np.array(boxes_list, dtype=np.float32)
                if np_boxes.ndim == 1 and num_detections == 1:
                    np_boxes = np_boxes.reshape(1, -1)
                # **** Check for 7 elements (including heading) ****
                if np_boxes.shape != (num_detections, 7):
                    # Try to handle if only 6 elements were provided (missing heading?)
                    if np_boxes.shape == (num_detections, 6):
                        # print(
                        #     f"Warning: boxes_lidar for frame {frame_id} has shape {np_boxes.shape}. Assuming 0 heading."
                        # )
                        np_boxes = np.hstack(
                            [np_boxes, np.zeros((num_detections, 1), dtype=np.float32)]
                        )  # Add heading=0
                    else:
                        raise ValueError(f"Expected boxes_lidar shape ({num_detections}, 7), got {np_boxes.shape}")

                np_labels = np.array(pred_labels) if pred_labels is not None else np.array([])

            else:
                np_names = np.array([], dtype=object)
                np_scores = np.array([], dtype=np.float32)
                np_boxes = np.zeros((0, 7), dtype=np.float32)  # Ensure shape is correct even when empty
                np_labels = np.array([])

        except ValueError as e:
            # print(f"Error converting data for frame_id '{frame_id}': {e}. Skipping frame.")
            continue

        output_frame_anno = {
            "name": np_names,
            "score": np_scores,
            "boxes_lidar": np_boxes,  # Now guaranteed to be (N, 7)
            "frame_id": frame_id,
            "pred_labels": np_labels,
        }
        all_predictions.append(output_frame_anno)

    # Optional: Sort predictions by frame_id
    try:
        all_predictions.sort(
            key=lambda x: (
                int(x["frame_id"]) if isinstance(x["frame_id"], str) and x["frame_id"].isdigit() else str(x["frame_id"])
            )
        )
    except Exception as e:
        print(f"Warning: Could not sort predictions by frame_id: {e}")

    num_loaded = len(all_predictions)
    filter_msg = ""
    if validation_ids is not None:
        filter_msg = f" (filtered from {original_count} total frames using validation set, skipped {filtered_count})"

    print(f"Successfully loaded predictions for {num_loaded} frames from {json_file_path}{filter_msg}")
    return all_predictions


def load_custom_gt_dataset(
    dataset_path: str,
    label_subdir: str = "labels",
    file_extension: str = ".txt",
    validation_ids: Optional[Set[str]] = None,  # Added optional argument
) -> List[Dict[str, Any]]:
    """Loads ground truth annotations, optionally filtering by validation IDs.

    Args:
        dataset_path (str): Path to the root directory of the custom dataset.
        label_subdir (str): Name of the subdirectory containing label files.
        file_extension (str): The file extension of the label files.
        validation_ids (Optional[Set[str]]): If provided, only load frames whose
                                              frame_id (basename without ext) is in this set.

    Returns:
        List[Dict[str, Any]]: A list of GT annotation dictionaries, one per frame.
                               Contains: 'name', 'gt_boxes_lidar', 'frame_id'.
                               Returns empty list on error or if no valid frames found.
    """
    label_dir = os.path.join(dataset_path, label_subdir)
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path not found or not a directory: {dataset_path}")
        return []
    if not os.path.isdir(label_dir):
        print(f"Error: Label subdirectory not found: {label_dir}")
        return []

    search_pattern = os.path.join(label_dir, f"*{file_extension}")
    all_label_files = sorted(glob.glob(search_pattern))

    if not all_label_files:
        print(f"Error: No label files found in {label_dir} with extension {file_extension}")
        return []

    print(f"Found {len(all_label_files)} potential label files in {label_dir}")

    all_gt_annos: List[Dict[str, Any]] = []
    original_count = len(all_label_files)
    filtered_count = 0

    for label_file_path in all_label_files:
        frame_id = os.path.splitext(os.path.basename(label_file_path))[0]

        # --- Filtering Step ---
        if validation_ids is not None and frame_id not in validation_ids:
            filtered_count += 1
            continue  # Skip this frame if not in the validation set
        # --- End Filtering Step ---

        object_names = []
        object_boxes = []  # Store as list of lists initially

        try:
            with open(label_file_path) as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ")
                # Expect 8 parts: 7 box parameters + 1 class name
                if len(parts) == 8:
                    try:
                        class_name = parts[-1]
                        box_params = [float(p) for p in parts[:-1]]  # [x,y,z,dx,dy,dz,heading]

                        # Basic validation
                        if any(dim <= 0 for dim in box_params[3:6]):
                            # print(
                            #     f"Warning: Skipping line {line_num + 1} in {label_file_path} for frame {frame_id}: Non-positive dimensions {box_params[3:6]}."
                            # )
                            continue
                        # Check heading range (optional, [-pi, pi])
                        if not (-np.pi <= box_params[6] <= np.pi):
                            norm_heading = np.arctan2(np.sin(box_params[6]), np.cos(box_params[6]))
                            if abs(norm_heading - box_params[6]) > 1e-3:  # Allow small tolerance
                                # print(
                                #     f"Warning: GT heading {box_params[6]:.4f} in {label_file_path} (line {line_num + 1}) is outside [-pi, pi]. Using normalized value {norm_heading:.4f}."
                                # )
                                pass  # Suppress warning for brevity
                            box_params[6] = norm_heading

                        object_names.append(class_name)
                        object_boxes.append(box_params)  # Append list [x,y,z,dx,dy,dz,heading]

                    except ValueError:
                        # print(
                        #     f"Warning: Skipping malformed line {line_num + 1} in {label_file_path} for frame {frame_id}: Invalid number format."
                        # )
                        continue
                else:
                    # print(
                    #     f"Warning: Skipping malformed line {line_num + 1} in {label_file_path} for frame {frame_id}: Expected 8 parts, got {len(parts)}."
                    # )
                    continue

        except Exception as e:
            print(f"Error reading or parsing file {label_file_path}: {e}")
            continue

        # Convert collected data to NumPy arrays
        num_objects = len(object_names)
        if num_objects > 0:
            np_names = np.array(object_names, dtype=object)
            # Ensure conversion handles list of lists correctly -> (N, 7)
            np_boxes = np.array(object_boxes, dtype=np.float32).reshape(num_objects, 7)
        else:
            np_names = np.array([], dtype=object)
            np_boxes = np.zeros((0, 7), dtype=np.float32)  # Correct shape for empty

        frame_anno = {
            "name": np_names,
            "gt_boxes_lidar": np_boxes,  # Now (N, 7)
            "frame_id": frame_id,
        }
        all_gt_annos.append(frame_anno)

    # Optional: Sort GT annotations by frame_id
    try:
        all_gt_annos.sort(
            key=lambda x: (
                int(x["frame_id"]) if isinstance(x["frame_id"], str) and x["frame_id"].isdigit() else str(x["frame_id"])
            )
        )
    except Exception as e:
        print(f"Warning: Could not sort GT annotations by frame_id: {e}")

    num_loaded = len(all_gt_annos)
    filter_msg = ""
    if validation_ids is not None:
        filter_msg = f" (filtered from {original_count} potential files using validation set, skipped {filtered_count})"

    print(f"Successfully loaded annotations for {num_loaded} GT frames{filter_msg}.")
    return all_gt_annos


# --- Precision-Recall Calculation Functions ---


# Utility function to normalize angle difference to [-pi, pi]
def normalize_angle_delta(delta_rad: float) -> float:
    """Normalizes an angle difference to the range [-pi, pi]."""
    return np.arctan2(np.sin(delta_rad), np.cos(delta_rad))


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
        print("Error: GPU device specified but GPU IoU function is not available (using placeholder). Cannot proceed.")
        return {}
    if not isinstance(model_predictions_lists, list) or not isinstance(gt_annos, list):
        print("Error: model_predictions_lists and gt_annos must be lists.")
        return {}
    # Allow empty lists if the validation set was empty or filtering removed everything
    # if not model_predictions_lists or not gt_annos:
    #     print("Warning: model_predictions_lists or gt_annos is empty (possibly due to filtering).")
    # Continue processing, results might be zero if one list is empty.

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
            if not USE_GPU_IOU:
                print("Warning: Proceeding with placeholder CPU IoU. Results will NOT be accurate.")
            else:
                print("Warning: CUDA not available, using CPU for tensor placement.")
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    print(f"Using device for PR/AOS data calculation: {device}")

    num_models = len(model_names)
    num_gt_samples = len(gt_annos)  # This is now the number of VALIDATION GT frames
    print(f"Calculating PR/AOS data for {num_models} models, {num_gt_samples} GT frames (validation set)...")
    print(f"Classes: {classes_to_evaluate}, IoU Threshold: {iou_threshold}, Min Score: {min_score_threshold}")
    start_time = time.time()

    # --- Prepare data structures ---
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

    # --- Convert model predictions to dictionaries keyed by frame_id ---
    # This uses the already filtered prediction lists
    model_preds_by_frame = defaultdict(dict)
    for model_idx, model_name in enumerate(model_names):
        pred_list = model_predictions_lists[model_idx]  # Should already be filtered
        if not isinstance(pred_list, list):
            print(f"Internal Error: Predictions for model '{model_name}' is not a list after loading/filtering.")
            continue  # Skip this model

        frames_processed_for_model = set()
        for pred_anno in pred_list:
            frame_id = pred_anno.get("frame_id")
            if frame_id is not None:
                # Since data is pre-filtered, duplicates shouldn't occur unless present in the source JSON *within the val set*
                if frame_id in frames_processed_for_model:
                    print(
                        f"Warning: Duplicate frame_id '{frame_id}' found within validation predictions for model '{model_name}'. Keeping first."
                    )
                    continue
                # Ensure boxes_lidar exists and has 7 elements before storing
                elif "boxes_lidar" in pred_anno and pred_anno["boxes_lidar"].shape[-1] == 7:
                    model_preds_by_frame[model_name][frame_id] = pred_anno
                    frames_processed_for_model.add(frame_id)
                else:
                    box_shape = pred_anno.get("boxes_lidar", np.array([])).shape
                    print(
                        f"Warning: Skipping prediction for validation frame {frame_id} in model {model_name} due to missing/invalid boxes_lidar (shape {box_shape}, expected N x 7). Needed for AOS."
                    )
            # else: # Should not happen if loader is correct
            #     print("Warning: Prediction annotation found without frame_id during PR calculation. Skipping.")

    # --- Calculate total GT per class (from the filtered GT list) ---
    print("Calculating total ground truth counts per class (validation set)...")
    global_total_gt = {c_name: 0 for c_name in classes_to_evaluate}
    gt_frame_ids_set = set()  # Set of frame IDs actually present in the filtered GT list
    for gt_anno in gt_annos:  # Iterate only over filtered GTs
        frame_id = gt_anno.get("frame_id")
        if frame_id is None:
            continue
        gt_frame_ids_set.add(frame_id)
        gt_names = gt_anno.get("name", np.array([]))
        gt_boxes = gt_anno.get("gt_boxes_lidar", np.zeros((0, 7)))
        if gt_boxes.shape[-1] != 7 and gt_boxes.size > 0:
            print(
                f"ERROR: GT boxes for validation frame {frame_id} do not have 7 elements (shape {gt_boxes.shape}). Cannot calculate AOS. Skipping frame for GT count."
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
                    f"Warning: No ground truth found for class '{c_name}' in the validation set. PR/AOS cannot be calculated."
                )

    print(f"Total validation GT counts: {global_total_gt}")
    print(f"Unique validation GT frame IDs found: {len(gt_frame_ids_set)}")

    # --- Iterate through GROUND TRUTH frames (validation set) for matching ---
    print("Matching predictions to ground truth across validation frames...")
    processed_frame_count = 0
    gt_frames_without_preds_in_val = {m: 0 for m in model_names}

    # This loop now only iterates over the filtered gt_annos
    for sample_idx, gt_anno in enumerate(gt_annos):
        gt_frame_id = gt_anno.get("frame_id")
        if gt_frame_id is None:
            continue  # Should not happen if loader works

        # Logging frequency can be adjusted
        # if (sample_idx + 1) % 50 == 0:
        #     print(f"  Processing validation GT frame {sample_idx + 1}/{num_gt_samples} (ID: {gt_frame_id})...")
        processed_frame_count += 1

        gt_names = gt_anno.get("name", np.array([]))
        gt_boxes_np = gt_anno.get("gt_boxes_lidar", np.zeros((0, 7), dtype=np.float32))

        if gt_boxes_np.ndim != 2 or (gt_boxes_np.shape[1] != 7 and gt_boxes_np.size > 0):
            print(
                f"ERROR: Invalid GT boxes shape {gt_boxes_np.shape} for validation frame {gt_frame_id}. Must be (N, 7). Skipping frame."
            )
            continue
        if gt_boxes_np.size == 0:
            gt_boxes_np = np.zeros((0, 7), dtype=np.float32)

        for model_name in model_names:
            # Look up prediction for this specific validation frame ID
            dt_anno = model_preds_by_frame[model_name].get(gt_frame_id)

            if dt_anno is None:
                # This means the model did not have predictions for this *specific validation frame*
                gt_frames_without_preds_in_val[model_name] += 1
                continue

            dt_names = dt_anno.get("name", np.array([]))
            dt_boxes_np = dt_anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))
            dt_scores = dt_anno.get("score", np.array([]))

            if dt_boxes_np.ndim != 2 or (dt_boxes_np.shape[1] != 7 and dt_boxes_np.size > 0):
                print(
                    f"ERROR: Invalid DT boxes shape {dt_boxes_np.shape} for model {model_name}, validation frame {gt_frame_id}. Must be (N, 7). Skipping frame for this model."
                )
                continue
            if dt_boxes_np.size == 0:
                dt_boxes_np = np.zeros((0, 7), dtype=np.float32)
            if dt_scores.ndim != 1:
                dt_scores = dt_scores.flatten()

            # Apply MINIMAL score threshold
            if min_score_threshold > 0 and dt_scores.size > 0:
                score_mask = dt_scores >= min_score_threshold
                if not np.any(score_mask):
                    # No detections pass score threshold for this frame/model
                    # Add FPs if needed? No, AP calculation handles this. Just continue.
                    dt_names = np.array([])
                    dt_boxes_np = np.zeros((0, 7), dtype=np.float32)
                    dt_scores = np.array([])
                else:
                    dt_names = dt_names[score_mask]
                    dt_boxes_np = dt_boxes_np[score_mask]
                    dt_scores = dt_scores[score_mask]
            # elif dt_scores.size == 0: # Handled by checks below

            for class_name in classes_to_evaluate:
                if global_total_gt[class_name] == 0:
                    continue  # Skip class if no GT in validation set

                # Filter GT and DT for the current class
                gt_mask = gt_names == class_name
                dt_mask = dt_names == class_name

                cur_gt_boxes_np = gt_boxes_np[gt_mask]
                cur_dt_boxes_np = dt_boxes_np[dt_mask]
                cur_dt_scores_np = dt_scores[dt_mask]
                num_gt = cur_gt_boxes_np.shape[0]
                num_dt = cur_dt_boxes_np.shape[0]

                # If there are detections for this class, store their scores
                if num_dt > 0:
                    temp_scores[model_name][class_name].extend(cur_dt_scores_np.tolist())
                else:
                    # If num_dt is 0, we don't need to calculate IoU or matches.
                    # We still need empty arrays for match status and similarity
                    # if scores were added (e.g., from other frames).
                    # The logic below handles appending zeros correctly.
                    pass

                # Initialize match status and similarity arrays for detections *in this frame*
                match_status_for_frame = np.zeros(num_dt, dtype=np.int8)  # 0=FP (default), 1=TP
                similarity_for_frame = np.zeros(num_dt, dtype=np.float32)  # 0=FP or no match, (1+cos)/2 for TP

                # Only calculate IoU and match if there are both GT and DT for this class in this frame
                if num_gt > 0 and num_dt > 0:
                    try:
                        # Calculate IoU matrix
                        cur_gt_boxes_torch = torch.from_numpy(cur_gt_boxes_np).float().to(device)
                        cur_dt_boxes_torch = torch.from_numpy(cur_dt_boxes_np).float().to(device)
                        # print(f"DEBUG: dt_shape={cur_dt_boxes_torch.shape}, gt_shape={cur_gt_boxes_torch.shape}") # Debug shape
                        iou_matrix_gpu = boxes_iou3d_gpu(cur_dt_boxes_torch, cur_gt_boxes_torch)
                        # print(f"DEBUG: iou_shape={iou_matrix_gpu.shape}") # Debug shape
                        iou_matrix = iou_matrix_gpu.cpu().numpy()  # Shape: (num_dt, num_gt)

                        gt_matched = np.zeros(num_gt, dtype=bool)
                        # Match greedily by score (higher score gets priority)
                        sorted_dt_indices = np.argsort(-cur_dt_scores_np)

                        for original_dt_idx in sorted_dt_indices:  # Iterate using original indices sorted by score
                            ious_for_this_dt = iou_matrix[original_dt_idx, :]
                            best_gt_match_idx = -1

                            # Find the best *unmatched* GT box above the IoU threshold
                            valid_gt_indices = np.where((ious_for_this_dt >= iou_threshold) & (~gt_matched))[0]

                            if len(valid_gt_indices) > 0:
                                # Pick the one with the highest IoU among valid *unmatched* GTs
                                best_relative_idx = np.argmax(ious_for_this_dt[valid_gt_indices])
                                best_gt_match_idx = valid_gt_indices[best_relative_idx]  # Index within cur_gt_boxes_np

                                # Match found: This detection is TP
                                match_status_for_frame[original_dt_idx] = 1
                                gt_matched[best_gt_match_idx] = True

                                # Calculate Orientation Similarity
                                try:
                                    gt_heading = cur_gt_boxes_np[best_gt_match_idx, 6]
                                    dt_heading = cur_dt_boxes_np[original_dt_idx, 6]
                                    delta_theta = normalize_angle_delta(dt_heading - gt_heading)
                                    similarity = (1.0 + np.cos(delta_theta)) / 2.0
                                    similarity_for_frame[original_dt_idx] = similarity
                                except IndexError:
                                    print(
                                        f"\nERROR: Indexing error getting heading for validation frame {gt_frame_id}, model {model_name}, class {class_name}. Check box shapes."
                                    )
                                    similarity_for_frame[original_dt_idx] = 0  # Assign 0 similarity on error

                            # Else: No suitable unmatched GT found, detection remains FP (status=0, similarity=0)

                    except Exception as e:
                        print(
                            f"\nError during IoU/matching for validation frame {gt_frame_id}, model {model_name}, class {class_name}: {e}"
                        )
                        # If error occurs, treat all detections in this frame/class as FP for safety
                        match_status_for_frame = np.zeros(num_dt, dtype=np.int8)
                        similarity_for_frame = np.zeros(num_dt, dtype=np.float32)

                # Append results for this frame/class/model to temporary lists
                # Note: We added scores first. Now append matches and similarities corresponding to those scores.
                # If num_dt was 0, these lists are empty, so nothing is appended, which is correct.
                temp_matches[model_name][class_name].extend(match_status_for_frame.tolist())
                temp_similarity[model_name][class_name].extend(similarity_for_frame.tolist())

    # --- Finalize data structure ---
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
                # This indicates a bug in the frame processing loop logic
                return {}  # Return empty on critical error

            pr_data[model_name][class_name]["scores"] = np.array(scores_list, dtype=np.float32)
            pr_data[model_name][class_name]["match_status"] = np.array(matches_list, dtype=np.int8)
            pr_data[model_name][class_name]["orientation_similarity"] = np.array(similarity_list, dtype=np.float32)
            total_detections_processed += len(scores_list)

    elapsed_time = time.time() - start_time
    print(f"PR/AOS data calculation finished in {elapsed_time:.2f} seconds.")
    print(f"Processed {processed_frame_count} validation GT frames.")
    print(f"Total detections considered (score >= {min_score_threshold}): {total_detections_processed}")
    for model_name in model_names:
        if model_name in gt_frames_without_preds_in_val:  # Check if key exists
            print(
                f"  Model '{model_name}': No predictions found for {gt_frames_without_preds_in_val[model_name]} validation GT frames."
            )

    if total_detections_processed == 0 and all(gt == 0 for gt in global_total_gt.values()):
        print("Warning: No detections passed threshold and/or no GT objects found in the validation set.")

    return pr_data


# --- Function: Calculate Precision-Recall Curve (no changes needed) ---
def calculate_precision_recall_curve(
    scores: np.ndarray, match_status: np.ndarray, total_gt: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates precision and recall points for a PR curve."""
    if not isinstance(scores, np.ndarray) or not isinstance(match_status, np.ndarray):
        print("Error: scores and match_status must be numpy arrays.")
        return np.array([]), np.array([]), np.array([])
    if scores.shape != match_status.shape:
        print(f"Error: scores shape {scores.shape} != match_status shape {match_status.shape}")
        return np.array([]), np.array([]), np.array([])
    if total_gt < 0:
        print("Error: total_gt cannot be negative.")
        return np.array([]), np.array([]), np.array([])

    # Handle case of no GT objects: Precision is undefined (or 0), Recall is 0. Return a single point.
    if total_gt == 0:
        # Return recall=[0], precision=[0] or perhaps recall=[0], precision=[1]?
        # Let's return R=0, P=0 if no GT. AP will be 0.
        return np.array([0.0]), np.array([0.0]), np.array([])  # No thresholds applicable

    num_detections = len(scores)
    # Handle case of no detections: Precision is 1 (no FP), Recall is 0. Return standard start point.
    if num_detections == 0:
        # Standard way to start PR curve at (R=0, P=1)
        return np.array([0.0]), np.array([1.0]), np.array([])  # No thresholds applicable

    sorted_indices = np.argsort(scores)[::-1]
    scores_sorted = scores[sorted_indices]
    match_status_sorted = match_status[sorted_indices]

    tp_cumulative = np.cumsum(match_status_sorted, dtype=np.float64)
    fp_cumulative = np.cumsum(1 - match_status_sorted, dtype=np.float64)

    recall_points = tp_cumulative / total_gt  # Avoid adding epsilon here if total_gt > 0
    # Add epsilon for precision denominator to avoid 0/0
    precision_points = tp_cumulative / (tp_cumulative + fp_cumulative + 1e-10)

    # Add the starting point (R=0, P=1)
    recall_final = np.concatenate(([0.0], recall_points))
    precision_final = np.concatenate(([1.0], precision_points))

    # Thresholds corresponding to each point (optional to return)
    # Add a dummy threshold slightly higher than the max score for the P=1 point
    first_threshold = scores_sorted[0] + 1e-3 if num_detections > 0 else 1.0
    score_thresholds_final = np.concatenate(([first_threshold], scores_sorted))

    return recall_final, precision_final, score_thresholds_final


# --- Function: Calculate AP from PR (no changes needed) ---
def calculate_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculates Average Precision (AP) using trapezoidal integration from PR points.
    Assumes recall is sorted or sorts it.
    """
    if len(recall) == 0 or len(precision) == 0 or len(recall) != len(precision):
        return 0.0

    # Ensure recall is sorted non-decreasingly for correct integration
    # The calculate_precision_recall_curve should already produce sorted recall
    # but sorting again ensures correctness if points are passed differently.
    sort_indices = np.argsort(recall)
    recall_sorted = recall[sort_indices]
    precision_sorted = precision[sort_indices]

    # Ensure precision is monotonically decreasing (common practice for AP)
    # For each point, precision should be max of precision values to its right
    for i in range(len(precision_sorted) - 2, -1, -1):
        precision_sorted[i] = max(precision_sorted[i], precision_sorted[i + 1])

    # Use numpy.trapz for integration of the (now monotonic) PR curve
    # Note: Some methods use summation over recall steps (e.g., 11-point interpolation)
    # Trapezoidal rule on the actual points is generally preferred now.
    ap = np.trapz(precision_sorted, recall_sorted)
    return ap


# --- Function: Calculate AOS from PR data (no changes needed) ---
def calculate_aos_from_pr_data(match_status: np.ndarray, orientation_similarity: np.ndarray) -> float:
    """Calculates Average Orientation Similarity (AOS).

    AOS = Average of orientation_similarity for all True Positive detections.

    Args:
        match_status: NumPy array (N,) indicating TP (1) or FP (0).
        orientation_similarity: NumPy array (N,) with similarity (1+cos)/2 for TPs, 0 for FPs.

    Returns:
        float: The calculated AOS value. Returns 0.0 if no True Positives are found.
    """
    if not isinstance(match_status, np.ndarray) or not isinstance(orientation_similarity, np.ndarray):
        print("Error: match_status and orientation_similarity must be numpy arrays.")
        return 0.0
    if match_status.shape != orientation_similarity.shape:
        print(
            f"Error: Shape mismatch: match_status {match_status.shape}, orientation_similarity {orientation_similarity.shape}"
        )
        return 0.0

    # Filter similarities for True Positives only
    tp_mask = match_status == 1
    tp_similarities = orientation_similarity[tp_mask]

    num_tp = len(tp_similarities)

    if num_tp == 0:
        # print("Warning: No True Positives found for AOS calculation.")
        return 0.0  # AOS is 0 if no TPs

    # Calculate the mean similarity over TPs
    aos = np.mean(tp_similarities)

    return aos


# --- Function: Plot PR Curves (no changes needed) ---
def plot_pr_curves(
    pr_data: Dict[str, Dict[str, Dict[str, Any]]],
    model_names: List[str],
    classes_to_plot: List[str],
    iou_threshold: float,
    # NEW: Pass calculated AOS values
    aos_results: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: str = None,
    figsize_per_plot: Tuple[int, int] = (6, 5),
):
    """Plots Precision-Recall curves and optionally includes AOS in the legend.

    Args:
        pr_data: Data structure returned by calculate_pr_data_gpu.
        model_names: List of model names expected in pr_data.
        classes_to_plot: List of class names to plot curves for.
        iou_threshold: The IoU threshold used (for title).
        aos_results: Optional dict {model: {class: aos_value}} to display AOS.
        save_path: Optional path to save the figure. If None, shows the plot.
        figsize_per_plot: Tuple controlling (width, height) of each subplot.
    """
    num_classes = len(classes_to_plot)
    if num_classes == 0:
        print("No classes specified for plotting.")
        return
    if not pr_data:
        print("Error: pr_data dictionary is empty. Cannot plot.")
        return

    # Determine subplot grid layout
    cols = max(1, int(np.ceil(np.sqrt(num_classes))))
    rows = max(1, int(np.ceil(num_classes / cols)))
    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    plot_idx = 0
    valid_plots_generated = False  # Flag to check if any actual data was plotted

    print("\n--- Generating PR Curve Plots ---")
    for class_name in classes_to_plot:
        if plot_idx >= len(axes_flat):
            print(
                f"Warning: More classes ({num_classes}) than available subplots ({len(axes_flat)}). Skipping remaining classes."
            )
            break  # Stop if we run out of subplots

        ax = axes_flat[plot_idx]
        ax.set_title(f"Class: '{class_name}' (IoU={iou_threshold})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])  # Allow slightly > 1 for P=1 point visibility
        ax.grid(True, linestyle="--", alpha=0.6)

        class_has_data = False  # Track if any model had data for *this* class
        for model_name in model_names:
            # Check if model and class exist in the data structure
            if model_name not in pr_data or class_name not in pr_data[model_name]:
                # This is expected if a model had no detections for this class
                # print(f"Info: No PR data found for model '{model_name}', class '{class_name}'. Skipping plot line.")
                continue

            data = pr_data[model_name][class_name]
            scores = data.get("scores")
            match_status = data.get("match_status")
            total_gt = data.get("total_gt")

            # Validate data presence and type
            if (
                scores is None
                or match_status is None
                or total_gt is None
                or not isinstance(scores, np.ndarray)
                or not isinstance(match_status, np.ndarray)
                or not isinstance(total_gt, numbers.Integral)
            ):
                print(
                    f"Warning: Incomplete/invalid data for model '{model_name}', class '{class_name}'. Skipping plot line."
                )
                continue

            # Handle case where there are no ground truth objects for this class
            if total_gt == 0:
                # Don't plot a line, maybe add a note later?
                # print(f"Info: Skipping plot line for '{model_name}', class '{class_name}' because Total GT = 0.")
                continue  # Skip plotting for this model/class if no GT

            # Handle case where there are GT but no detections passing score threshold
            if len(scores) == 0:
                # Plot a point at (R=0, P=0) or indicate no detections
                # Let's plot a marker at R=0, P=0 to show the model exists but had no detections
                ap = 0.0
                aos = 0.0  # No TPs -> AOS=0
                if aos_results and model_name in aos_results and class_name in aos_results[model_name]:
                    aos = aos_results[model_name][class_name]  # Could still be 0 if calculated

                label = f"{model_name} (AP={ap:.3f}, AOS={aos:.3f}, No Dets)"
                ax.plot([0], [0], marker="x", markersize=8, linestyle="", label=label)
                class_has_data = True
                valid_plots_generated = True  # Count this as a plotted item
                continue

            # Calculate PR curve points
            recall, precision, _ = calculate_precision_recall_curve(scores, match_status, total_gt)

            if len(recall) > 0 and len(precision) > 0:
                # Calculate AP using the generated points
                ap = calculate_ap_from_pr(recall, precision)

                # Construct label, including AOS if available
                label = f"{model_name} (AP={ap:.3f}"
                if aos_results and model_name in aos_results and class_name in aos_results[model_name]:
                    aos = aos_results[model_name][class_name]
                    label += f", AOS={aos:.3f}"
                label += ")"

                # Plot the curve
                ax.plot(recall, precision, marker=".", markersize=3, linestyle="-", linewidth=1.5, label=label)
                class_has_data = True
                valid_plots_generated = True
            else:
                # This might happen if calculate_precision_recall_curve has issues (e.g., unexpected inputs)
                print(
                    f"Warning: Could not calculate valid PR points for model '{model_name}', class '{class_name}'. Skipping plot line."
                )

        # Add legend to the subplot if any lines were plotted
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize="small")
        elif not class_has_data:
            # If no model had data or GT for this class, add a placeholder text
            ax.text(
                0.5,
                0.5,
                "No data or GT=0",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="grey",
                style="italic",
                fontsize=10,
            )

        plot_idx += 1  # Move to the next subplot index

    # Hide any unused subplots
    for i in range(plot_idx, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap
    fig.suptitle(f"Precision-Recall Curves (IoU Threshold = {iou_threshold}, Validation Set)", fontsize=14, y=0.99)

    # Save or show the plot
    if valid_plots_generated:
        try:
            if save_path:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=150)
                print(f"PR curve plot saved to: {save_path}")
            else:
                print("Displaying PR curve plot...")
                plt.show()
        except Exception as e:
            print(f"Error saving or showing plot: {e}")
    elif not pr_data:
        print("Skipping plot generation because pr_data was empty.")
    else:
        print(
            "Skipping plot generation as no valid data points or curves were generated (check GT counts and detections)."
        )

    plt.close(fig)  # Close the figure to free memory


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Path to the root of your dataset (containing labels, point clouds etc.)
    DATASET_ROOT = "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/data/custom"
    # Subdirectory containing the ground truth label files
    GT_LABEL_SUBDIR = "labels"
    GT_FILE_EXTENSION = ".txt"

    # --- NEW: Path to the validation split file ---
    # This file should contain one frame ID (filename without extension) per line
    # E.g., /path/to/your/dataset/ImageSets/val.txt
    VALIDATION_SPLIT_FILE = "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/data/custom/ImageSets/val.txt"  # ADJUST THIS PATH

    # List of JSON result files from your models
    MODEL_RESULT_JSONS = ["/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn_early_stopping/test_intensity_freeze/eval/epoch_4/val/result.json",
        "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn_early_stopping/test_intensity/eval/epoch_13/val/result.json", "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn_early_stopping/no_pretrained_no_freeze/eval/epoch_17/val/result.json"
    ,"/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn/default/eval/epoch_7870/val/default/result.json"]
    # Corresponding names for the models (for legends)
    MODEL_NAMES = ["Pointrcnn+finetune+freeze", "Pointrcnn+finetune+no_freeze", "Pointrcnn+no_pretrained", "Pointrcnn_default"]

    # List of classes you want to evaluate (must match names in GT/DT files)
    CLASSES_TO_EVALUATE = ["Pedestrian"]
    # List of IoU thresholds to compute metrics and plot curves for
    IOU_THRESHOLDS_TO_PLOT = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7]
    # Minimum detection score threshold (detections below this are ignored)
    MIN_DETECTION_SCORE = 0.01  # Set to 0 to include all detections
    # Directory where the output plot images will be saved
    OUTPUT_PLOT_DIR = "./pr_aos_curve_plots_validation_fine_tuned"  # Changed dir name
    # Computation device ('cuda:0', 'cpu', etc.)
    DEVICE_ID = "cuda:0" if torch.cuda.is_available() and USE_GPU_IOU else "cpu"

    # --- End Configuration ---

    # --- Basic Input Validation ---
    if len(MODEL_RESULT_JSONS) != len(MODEL_NAMES):
        print("Error: Number of model result JSONs does not match the number of model names.")
        sys.exit(1)
    if not CLASSES_TO_EVALUATE:
        print("Error: CLASSES_TO_EVALUATE list cannot be empty.")
        sys.exit(1)
    if not IOU_THRESHOLDS_TO_PLOT:
        print("Error: IOU_THRESHOLDS_TO_PLOT list cannot be empty.")
        sys.exit(1)
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root directory not found: {DATASET_ROOT}")
        sys.exit(1)

    print("=" * 60)
    print("Starting PR & AOS Calculation Process (Validation Set Only)")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Validation Split File: {VALIDATION_SPLIT_FILE}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Classes: {CLASSES_TO_EVALUATE}")
    print(f"IoU Thresholds: {IOU_THRESHOLDS_TO_PLOT}")
    print(f"Min Score Threshold: {MIN_DETECTION_SCORE}")
    print(f"Device: {DEVICE_ID}")
    print(f"Using GPU IoU: {USE_GPU_IOU}")
    print("=" * 60)

    # 1. Load Validation Set Frame IDs
    print("\nLoading Validation Set IDs...")
    validation_frame_ids = load_validation_ids(VALIDATION_SPLIT_FILE)
    if validation_frame_ids is None:
        print("\nCritical Error: Failed to load validation set IDs. Exiting.")
        sys.exit(1)
    if not validation_frame_ids:
        print("\nWarning: Validation set ID list is empty. No data will be processed.")
        # Allow continuing, but expect no results.

    # 2. Load Ground Truth (filtered by validation IDs)
    print("\nLoading Ground Truth Annotations (Validation Set Only)...")
    # Pass the loaded validation IDs to the loading function
    gt_annotations_val = load_custom_gt_dataset(
        DATASET_ROOT,
        GT_LABEL_SUBDIR,
        GT_FILE_EXTENSION,
        validation_ids=validation_frame_ids,  # Pass the set here
    )
    if not gt_annotations_val and validation_frame_ids:
        print("\nWarning: No ground truth annotations found for the specified validation frame IDs.")
        # Allow continuing, but AP/AOS will likely be 0.

    # 3. Load Model Predictions (filtered by validation IDs)
    print("\nLoading Model Predictions (Validation Set Only)...")
    all_model_predictions_val = []
    valid_model_names_val = []  # Keep track of models loaded successfully for validation set

    for i, json_path in enumerate(MODEL_RESULT_JSONS):
        model_name = MODEL_NAMES[i]
        print(f"  Loading predictions for '{model_name}' from: {json_path}")
        # Pass the loaded validation IDs to the loading function
        model_preds = load_model_results_from_json(
            json_path,
            validation_ids=validation_frame_ids,  # Pass the set here
        )

        if not model_preds:
            print(
                f"  Info: No valid predictions found for '{model_name}' in the validation set (or failed to load JSON). Skipping this model."
            )
            continue  # Skip if loading failed or filtering resulted in empty list

        # Optional: Further check if any loaded predictions have valid boxes (already done in loader)
        has_valid_boxes = any(
            "boxes_lidar" in frame
            and isinstance(frame.get("boxes_lidar"), np.ndarray)
            and frame["boxes_lidar"].shape[-1] == 7
            for frame in model_preds
        )
        if not has_valid_boxes:
            print(
                f"  Warning: Although predictions were loaded for '{model_name}', none contained valid 7-element 'boxes_lidar' required for AOS calculation within the validation set. AOS will be 0."
            )
            # Still include the model for AP calculation if desired

        all_model_predictions_val.append(model_preds)
        valid_model_names_val.append(model_name)  # Add names of models successfully loaded with validation data

    if not all_model_predictions_val:
        print(
            "\nWarning: Failed to load valid predictions for ANY model for the validation set. No results can be computed."
        )
        # Allow continuing, but expect no results.
    elif len(valid_model_names_val) < len(MODEL_NAMES):
        print("\nWarning: Some models were skipped or had no predictions in the validation set.")
        print(f"Evaluating models with validation data: {valid_model_names_val}")

    # Check if we have both GT and Predictions for the validation set
    if not gt_annotations_val or not all_model_predictions_val:
        print("\nExiting: Cannot proceed without both ground truth and model predictions for the validation set.")
        sys.exit(0)  # Exit gracefully, not an error, just nothing to compute.

    # 4. Setup Device and Output Directory
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    try:
        evaluation_device = torch.device(DEVICE_ID)
        # Test device availability
        _ = torch.tensor([1.0], device=evaluation_device)
        print(f"Successfully set up computation device: {evaluation_device}")
    except Exception as e:
        print(f"\nError initializing torch device '{DEVICE_ID}': {e}. Falling back to CPU.")
        evaluation_device = torch.device("cpu")
        if USE_GPU_IOU:
            print(
                "Critical Warning: GPU specified and IoU function exists, but device failed. IoU calculations will likely fail or be inaccurate."
            )
            # Decide whether to exit or continue with placeholder
            # Forcing CPU placeholder if GPU IoU was intended but device failed
            print("Forcing use of placeholder CPU IoU due to device error.")
            USE_GPU_IOU = False

    # 5. Calculate and Plot for each IoU threshold using VALIDATION data
    all_results_summary = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )  # iou -> model -> class -> metric_value

    for iou_val in IOU_THRESHOLDS_TO_PLOT:
        print(f"\n--- Processing for IoU Threshold: {iou_val:.2f} (Validation Set) ---")

        # Calculate PR data (including orientation similarity) using filtered data
        pr_data_for_iou = calculate_pr_data_gpu(
            model_predictions_lists=all_model_predictions_val,  # Use validation predictions
            gt_annos=gt_annotations_val,  # Use validation GT
            model_names=valid_model_names_val,  # Use names of models with validation data
            classes_to_evaluate=CLASSES_TO_EVALUATE,
            iou_threshold=iou_val,
            device=evaluation_device,
            min_score_threshold=MIN_DETECTION_SCORE,
        )

        if not pr_data_for_iou:
            print(
                f"Warning: Failed to generate PR/AOS data for IoU {iou_val:.2f}. Skipping calculations and plot for this IoU."
            )
            continue  # Skip to next IoU threshold

        # Calculate AP & AOS for each model/class from the generated PR data
        current_aos_results = defaultdict(dict)
        current_ap_results = defaultdict(dict)

        print(f"\n--- Results for IoU = {iou_val:.2f} (Validation Set) ---")
        print(f"{'Model':<25} {'Class':<15} {'AP':<10} {'AOS':<10} {'Total GT':<10} {'Num Dets':<10} {'Num TPs':<10}")
        print("-" * 90)

        no_results_for_iou = True
        for model_name in valid_model_names_val:
            if model_name not in pr_data_for_iou:
                continue  # Should not happen if logic is correct

            for class_name in CLASSES_TO_EVALUATE:
                if class_name not in pr_data_for_iou[model_name]:
                    # print(f"Info: No data for {model_name}/{class_name} at IoU {iou_val:.2f}")
                    continue  # Skip if class data doesn't exist (e.g., no GT/DT for this class)

                data = pr_data_for_iou[model_name][class_name]
                scores = data.get("scores")
                match_status = data.get("match_status")
                orientation_similarity = data.get("orientation_similarity")
                total_gt = data.get("total_gt", 0)  # Get total GT count for this class in val set

                ap = 0.0
                aos = 0.0
                num_dets = 0
                num_tps = 0

                # Check if essential data exists before calculation
                if (
                    scores is not None
                    and match_status is not None
                    and orientation_similarity is not None
                    and total_gt is not None
                ):
                    num_dets = len(scores)
                    num_tps = np.sum(match_status == 1)

                    # Only calculate metrics if there's ground truth for this class
                    if total_gt > 0:
                        # Calculate AP even if num_dets is 0 (AP will be 0)
                        recall, precision, _ = calculate_precision_recall_curve(scores, match_status, total_gt)
                        ap = calculate_ap_from_pr(recall, precision)

                        # Calculate AOS only if there are detections
                        if num_dets > 0:
                            aos = calculate_aos_from_pr_data(match_status, orientation_similarity)
                        # Else AOS remains 0 (no TPs possible without detections)
                    # Else AP/AOS remain 0 if total_gt is 0

                    # Store results
                    current_ap_results[model_name][class_name] = ap
                    current_aos_results[model_name][class_name] = aos
                    all_results_summary[iou_val][model_name][f"{class_name}_AP"] = ap
                    all_results_summary[iou_val][model_name][f"{class_name}_AOS"] = aos
                    no_results_for_iou = False  # We got some results

                    print(
                        f"{model_name:<25} {class_name:<15} {ap:<10.4f} {aos:<10.4f} {total_gt:<10} {num_dets:<10} {num_tps:<10}"
                    )
                else:
                    # This case indicates missing arrays in pr_data structure, shouldn't happen ideally
                    print(
                        f"Warning: Missing essential data components for {model_name}/{class_name} at IoU {iou_val:.2f}. Skipping metrics."
                    )
                    print(
                        f"{model_name:<25} {class_name:<15} {'N/A':<10} {'N/A':<10} {total_gt if total_gt is not None else 'N/A':<10} {num_dets:<10} {num_tps:<10}"
                    )

        if no_results_for_iou:
            print("No valid AP/AOS results were calculated for this IoU threshold.")
        print("-" * 90)

        # Plotting (pass the calculated AOS values)
        plot_filename = f"pr_aos_curves_val_iou_{iou_val:.2f}.png"
        plot_save_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)

        plot_pr_curves(
            pr_data=pr_data_for_iou,
            model_names=valid_model_names_val,  # Use valid names
            classes_to_plot=CLASSES_TO_EVALUATE,
            iou_threshold=iou_val,
            aos_results=current_aos_results,  # Pass the calculated AOS
            save_path=plot_save_path,
        )

    print("\n--- PR & AOS Calculation Complete (Validation Set) ---")
    print(f"Plots saved in: {os.path.abspath(OUTPUT_PLOT_DIR)}")

    # Optional: Print summary table across IoUs
    print("\n--- Overall Summary (Validation Set) ---")
    try:
        summary_df = pd.DataFrame.from_dict(
            {
                (iou, model): all_results_summary[iou][model]
                for iou in all_results_summary
                for model in all_results_summary[iou]
            },
            orient="index",
        )
        if not summary_df.empty:
            summary_df.index = pd.MultiIndex.from_tuples(summary_df.index, names=["IoU", "Model"])
            # Format floats for better readability
            pd.options.display.float_format = "{:.4f}".format
            print(summary_df)
        else:
            print("No summary data generated (check if any results were calculated).")
    except Exception as e:
        print(f"Error generating summary DataFrame: {e}")
        print("Raw summary data:", json.dumps(all_results_summary, indent=2))

    if not USE_GPU_IOU:
        print("\n####################################################################")
        print("WARNING: Results generated using placeholder CPU IoU.")
        print("         Plots and Metrics (AP/AOS) are NOT ACCURATE.")
        print("####################################################################")

    print("\nDone.")
