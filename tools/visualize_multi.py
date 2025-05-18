import argparse
import glob
import json
from pathlib import Path

import numpy as np
from cycler import cycler  # For cycling through colors

try:
    from visual_utils import open3d_vis_utils as V

    # Define default colors suitable for Open3D (RGB 0-1 float)
    DEFAULT_MODEL_COLORS = [
        [0, 0, 1],  # Blue
        [0, 1, 0],  # Green
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [1, 0.5, 0],  # Orange
        [0.5, 0, 1],  # Purple
    ]
    OPEN3D_FLAG = True
    print("Using Open3D visualizer.")
except ImportError:  # Use ImportError for clarity
    try:
        from mayavi import mlab
        from visual_utils import visualize_utils as V

        # Define default colors suitable for Mayavi (RGB 0-1 float often works)
        DEFAULT_MODEL_COLORS = [
            (0, 0, 1),  # Blue
            (0, 1, 0),  # Green
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1),  # Cyan
            (1, 0.5, 0),  # Orange
            (0.5, 0, 1),  # Purple
        ]
        OPEN3D_FLAG = False
        print("Using Mayavi visualizer.")
    except ImportError:
        print("Neither Open3D nor Mayavi found for visualization. Install one (e.g., pip install open3d).")

        # Define a dummy V.draw_scenes if neither is installed to avoid NameError later
        class DummyViz:
            def draw_scenes(self, *args, **kwargs):
                print("Visualization skipped: No visualization backend (Open3D or Mayavi) found.")

        V = DummyViz()
        DEFAULT_MODEL_COLORS = []
        OPEN3D_FLAG = None  # Indicate no backend available


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


# Corrected DemoDataset class
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext=".bin"):
        """Args:
        root_path (str or Path): Path to the directory containing data files or path to a single data file.
        dataset_cfg (dict): Configuration for the dataset.
        class_names (list): List of class names.
        training (bool): Boolean indicating if it's for training.
        logger (logging.Logger): Logger object.
        ext (str): File extension to look for (e.g., '.bin' or '.npy').
        """
        # Ensure root_path is provided and convert to Path object
        if root_path is None:
            raise ValueError("root_path must be provided.")
        self.root_path = Path(root_path)
        self.ext = ext

        # Initialize the base class *before* using self.logger potentially
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=self.root_path, logger=logger
        )

        # Find data files
        if self.root_path.is_dir():
            # Search for files with the specified extension within the directory
            search_pattern = str(self.root_path / f"*{self.ext}")
            data_file_list = glob.glob(search_pattern)
            if not data_file_list and self.logger:
                self.logger.warning(f"No files found matching '{search_pattern}'")
            elif not data_file_list:
                print(f"Warning: No files found matching '{search_pattern}'")
        elif self.root_path.is_file() and self.root_path.suffix == self.ext:
            # Handle the case where root_path is a single valid file
            data_file_list = [str(self.root_path)]
        else:
            # Handle invalid root_path (not a directory, not a valid file)
            message = f"Root path '{self.root_path}' is not a directory or a valid '{self.ext}' file."
            if self.logger:
                self.logger.error(message)
            else:
                print(f"Error: {message}")
            data_file_list = []  # Ensure it's an empty list

        # Sort the file list for consistent order
        data_file_list.sort()
        self.sample_file_list = data_file_list

        if self.logger:
            self.logger.info(
                f"Found {len(self.sample_file_list)} samples in '{self.root_path}' with extension '{self.ext}'"
            )
        else:
            print(f"Found {len(self.sample_file_list)} samples in '{self.root_path}' with extension '{self.ext}'")

    def __len__(self):
        """Returns the number of samples found."""
        return len(self.sample_file_list)

    def __getitem__(self, index):
        """Loads and returns a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the data read from the file.
                  Expected keys: 'points', 'frame_id'.
        """
        # Get the full path of the specific file for this index
        file_path_str = self.sample_file_list[index]
        file_path = Path(file_path_str)

        frame_id = file_path.stem

        # Load points based on the file extension
        try:
            if self.ext == ".bin":
                # Assuming 4 columns (e.g., x, y, z, intensity/reflectance)
                points = np.fromfile(file_path_str, dtype=np.float32).reshape(-1, 4)
            elif self.ext == ".npy":
                points = np.load(file_path_str)


            else:

                raise NotImplementedError(f"Loading for extension '{self.ext}' is not implemented.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading file {file_path_str}: {e}")
            else:
                print(f"Error loading file {file_path_str}: {e}")
            raise e

        # Create the dictionary for the data sample
        input_dict = {
            "points": points,
            "frame_id": frame_id,  # Use the frame_id from the specific file
        }

        # Pass the raw data dictionary to the preparation step
        data_dict = self.prepare_data(data_dict=input_dict)

        # Ensure frame_id is preserved after prepare_data (it usually is, but good practice)
        if "frame_id" not in data_dict:
            data_dict["frame_id"] = frame_id

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/custom_models/pointrcnn_iou_early_stopping.yaml",  # Using a common default
        help="specify the config for dataset loading parameters",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/data/custom/points",  # CHANGE THIS
        help="specify the point cloud data directory (e.g., the 'points' folder)",
    )
    parser.add_argument(
        "--pred_json_paths",
        type=str,
        nargs="+",  # Accept one or more paths
        required=True,
        help="specify one or more paths to prediction JSON files",
    )
    parser.add_argument(
        "--label_dir_name",
        type=str,
        default="labels",
        help="Name of the directory containing GT label files (relative to parent of data_path)",
    )
    parser.add_argument("--ext", type=str, default=".npy", help="specify the extension of your point cloud data file")
    # Optional: Add argument for custom colors if needed later
    # parser.add_argument('--pred_colors', type=str, nargs='+', help='List of colors (e.g., "0,0,1" "0,1,0") matching pred_json_paths')

    args = parser.parse_args()

    # Load base config for dataset parameters primarily
    cfg_from_yaml_file(args.cfg_file, cfg)

    # Basic validation
    if len(args.pred_json_paths) > len(DEFAULT_MODEL_COLORS):
        print(
            f"Warning: More prediction files ({len(args.pred_json_paths)}) than default colors ({len(DEFAULT_MODEL_COLORS)}). Colors will repeat."
        )

    return args, cfg


def load_predictions(json_path, logger):
    """Loads predictions from a single JSON file."""
    pred_path_obj = Path(json_path)
    logger.info(f"Loading predictions from: {pred_path_obj}")
    predictions_by_frame = {}
    if pred_path_obj.is_file():
        try:
            with open(pred_path_obj) as f:
                all_predictions_list = json.load(f)
            predictions_by_frame = {str(pred["frame_id"]): pred for pred in all_predictions_list}
            logger.info(f"Loaded predictions for {len(predictions_by_frame)} frames from {pred_path_obj.name}.")
        except FileNotFoundError:
            logger.error(f"Error: Prediction JSON file not found at {pred_path_obj}")
        except json.JSONDecodeError:
            logger.error(f"Error: Could not decode JSON from {pred_path_obj}")
        except Exception as e:
            logger.error(f"Error loading predictions from {pred_path_obj}: {e}")
    else:
        logger.warning(f"Prediction JSON file not found at {pred_path_obj}. Skipping this file.")
    return predictions_by_frame


def extract_model_identifier(full_path_str: str) -> str:
    """Extracts the path components between 'output' and 'eval'.

    Args:
        full_path_str: The full path string.

    Returns:
        The extracted identifier string (e.g., "custom_models/pointrcnn/default/")
        or None if the structure isn't found.
    """
    try:
        p = Path(full_path_str)
        parts = p.parts

        # Find the indices of 'output' and 'eval'
        output_index = parts.index("output")
        eval_index = parts.index("eval")

        # Ensure 'eval' comes after 'output'
        if eval_index <= output_index:
            logger.warning(f"'eval' does not appear after 'output' in path: {full_path_str}")
            return None

        # Extract the parts between 'output' (exclusive) and 'eval' (exclusive)
        relevant_parts = parts[output_index + 1 : eval_index]

        if not relevant_parts:
            logger.warning(f"No path components found between 'output' and 'eval' in: {full_path_str}")
            return None

        # Join the relevant parts back into a string path segment
        # Using Path ensures correct separator, as_posix() standardizes to '/'
        # Add the trailing slash as per your examples
        identifier = Path(*relevant_parts).as_posix() + "/"
        return identifier

    except ValueError:
        # Handle cases where 'output' or 'eval' is not found in the path parts
        logger.warning(f"Could not find 'output' or 'eval' component in path: {full_path_str}")
        return None
    except Exception as e:
        logger.error(f"Error extracting identifier from path '{full_path_str}': {e}")
        return None


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------OpenPCDet Multi-Model Visualization-------------------------")
    if OPEN3D_FLAG is None:
        logger.error("No visualization backend found. Exiting.")
        return  # Exit if no viz library

    # --- Assign Colors to Models ---
    num_models = len(args.pred_json_paths)
    # Cycle through default colors if more models than colors
    color_cycler = cycler(color=DEFAULT_MODEL_COLORS)
    model_colors = [item["color"] for _, item in zip(range(num_models), color_cycler)]
    logger.info("Ground Truth: RED (default color)")
    logger.info(f"Assigning colors to {num_models} models [R,G,B]:")

    for i, path in enumerate(args.pred_json_paths):
        # Convert the original path string to a Path object
        model_identifier = extract_model_identifier(path)
        logger.info(f"  Model {i + 1} ({model_identifier}): {model_colors[i]}")

    # --- Load Predictions for Each Model ---
    all_predictions_by_frame_list = []  # List to hold prediction dicts, one per model
    for json_path in args.pred_json_paths:
        preds = load_predictions(json_path, logger)
        all_predictions_by_frame_list.append(preds)

    # --- Setup Dataset ---
    data_path_obj = Path(args.data_path)
    # Use dataset config from the loaded YAML (primarily for point feature encoding etc.)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,  # Class names might be needed by prepare_data
        training=False,
        root_path=data_path_obj,
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples found in data path: \t{len(demo_dataset)}")

    # --- Determine GT Label Path ---
    gt_label_base_path = data_path_obj.parent / args.label_dir_name
    logger.info(f"Expecting GT labels in: {gt_label_base_path}")

    # --- Loop through samples ---
    visualized_count = 0
    processed_count = 0
    for idx, data_dict_single in enumerate(demo_dataset):
        processed_count += 1
        current_frame_id = str(data_dict_single.get("frame_id", Path(demo_dataset.sample_file_list[idx]).stem))

        logger.debug(f"Processing sample index: {idx}, frame_id: {current_frame_id}")

        # --- Gather Predictions from ALL models for the current frame ---
        frame_all_pred_boxes = []
        frame_all_pred_scores = []
        frame_all_pred_labels = []
        frame_all_pred_colors = []
        has_any_predictions = False

        for model_idx, predictions_by_frame in enumerate(all_predictions_by_frame_list):
            pred_dict_for_frame = predictions_by_frame.get(current_frame_id)

            if pred_dict_for_frame is not None:
                pred_boxes_list = pred_dict_for_frame.get("boxes_lidar", [])
                pred_scores_list = pred_dict_for_frame.get("score", [])
                pred_labels_list = pred_dict_for_frame.get("pred_labels", [])

                if pred_boxes_list and len(pred_boxes_list) > 0:
                    num_boxes = len(pred_boxes_list)
                    model_color = model_colors[model_idx]

                    pred_boxes = np.array(pred_boxes_list, dtype=np.float32)
                    pred_scores = np.array(pred_scores_list, dtype=np.float32)
                    pred_labels = np.array(pred_labels_list, dtype=np.int64)
                    # Create color array for this model's boxes
                    pred_colors = np.tile(np.array(model_color), (num_boxes, 1))

                    frame_all_pred_boxes.append(pred_boxes)
                    frame_all_pred_scores.append(pred_scores)
                    frame_all_pred_labels.append(pred_labels)
                    frame_all_pred_colors.append(pred_colors)
                    has_any_predictions = True  # Mark that at least one model had preds
                    logger.debug(f"  Found {num_boxes} preds for frame {current_frame_id} from model {model_idx + 1}")

        # --- Consolidate Predictions if any were found ---
        final_pred_boxes = None
        final_pred_scores = None
        final_pred_labels = None
        final_pred_colors = None

        if has_any_predictions:
            final_pred_boxes = np.concatenate(frame_all_pred_boxes, axis=0)
            final_pred_scores = np.concatenate(frame_all_pred_scores, axis=0)
            final_pred_labels = np.concatenate(frame_all_pred_labels, axis=0)
            final_pred_colors = np.concatenate(frame_all_pred_colors, axis=0)
            logger.debug(f"  Total combined predictions for frame {current_frame_id}: {len(final_pred_boxes)}")


        # --- Load GT boxes ---
        gt_boxes = None  # Initialize as None
        gt_label_path = gt_label_base_path / f"{current_frame_id}.txt"
        has_gt = False

        if gt_label_path.is_file():
            loaded_gt = []
            try:
                with open(gt_label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            try:
                                x, y, z, l, w, h, ry = map(float, parts[:7])
                                if l > 0 and w > 0 and h > 0:
                                    loaded_gt.append([x, y, z, l, w, h, ry])
                                else:
                                    logger.warning(
                                        f"Skipping invalid GT box (<=0 dimension) in {gt_label_path}: {line.strip()}"
                                    )
                            except ValueError:
                                logger.warning(f"Skipping non-numeric GT line in {gt_label_path}: {line.strip()}")
                if loaded_gt:
                    gt_boxes = np.array(loaded_gt, dtype=np.float32)
                    has_gt = True
                    logger.debug(f"Loaded {len(gt_boxes)} GT boxes from {gt_label_path}")
                else:
                    logger.debug(f"GT file found ({gt_label_path}), but contained no valid boxes.")
            except Exception as e:
                logger.error(f"Error reading GT file {gt_label_path}: {e}")
                # gt_boxes remains None
        else:
            logger.debug(f"GT boxes file not found for frame_id: {current_frame_id} at {gt_label_path}")
            # gt_boxes remains None

        # --- Conditional Visualization: Only if GT and *some* predictions exist ---
        if has_any_predictions and has_gt:
            logger.info(
                f"Visualizing frame {current_frame_id} (Index: {idx}) - Found GT and Predictions from {sum(1 for x in frame_all_pred_boxes if x is not None)} models."
            )

            # Prepare data batch for visualization
            data_dict_batch = demo_dataset.collate_batch([data_dict_single])
            load_data_to_gpu(data_dict_batch)

            points_to_viz = data_dict_batch["points"][:, 1:]
            #print dimensions
            print(points_to_viz.shape)  # Should be (N, 4

            # Call visualizer with combined predictions and colors
            # **ASSUMES V.draw_scenes accepts ref_colors**
            try:
                V.draw_scenes(
                    points=points_to_viz,
                    ref_boxes=final_pred_boxes,  # Combined predicted boxes
                    ref_scores=final_pred_scores,  # Combined prediction scores
                    ref_labels=final_pred_labels,  # Combined prediction labels
                    ref_colors=final_pred_colors,  # Assigned colors per prediction box
                    gt_boxes=gt_boxes,
                    point_colors= points_to_viz[:4] # Ground truth boxes (likely default color)
                    # Add gt_labels, gt_colors if your visualizer supports them
                )
                visualized_count += 1

                if not OPEN3D_FLAG and OPEN3D_FLAG is not None:  # Mayavi requires explicit show
                    mlab.show(stop=True)

            except TypeError as e:
                if "ref_colors" in str(e):
                    logger.error(
                        f"Visualization failed for frame {current_frame_id}. Your 'V.draw_scenes' function might not support the 'ref_colors' argument."
                    )
                    logger.error(
                        "Check the visual_utils implementation or remove the 'ref_colors' parameter from the call."
                    )
                    # Optionally, try visualizing without colors as a fallback:
                    # logger.warning("Trying visualization without custom prediction colors...")
                    # try:
                    #     V.draw_scenes(points=points_to_viz, ref_boxes=final_pred_boxes, ref_scores=final_pred_scores, ref_labels=final_pred_labels, gt_boxes=gt_boxes)
                    #     visualized_count += 1
                    #     if not OPEN3D_FLAG and OPEN3D_FLAG is not None: mlab.show(stop=True)
                    # except Exception as e_fallback:
                    #      logger.error(f"Fallback visualization also failed: {e_fallback}")

                else:
                    logger.error(f"Visualization failed for frame {current_frame_id} with unexpected TypeError: {e}")
            except Exception as e:
                logger.error(f"Visualization failed for frame {current_frame_id} with unexpected error: {e}")

        else:
            # Log why visualization is skipped
            skip_reasons = []
            if not has_any_predictions:
                skip_reasons.append("no predictions found from any model")
            if not has_gt:
                skip_reasons.append("no ground truth found")
            logger.info(
                f"Skipping visualization for frame {current_frame_id} (Index: {idx}): {', '.join(skip_reasons)}."
            )
        # --- End Conditional Visualization ---

    logger.info("-----------------Finished Visualization-------------------------")
    logger.info(f"Processed {processed_count} samples.")
    logger.info(f"Visualized {visualized_count} samples (where both GT and predictions were present).")


if __name__ == "__main__":
    main()
