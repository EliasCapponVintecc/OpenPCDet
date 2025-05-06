import argparse
import glob
from pathlib import Path

try:
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    from mayavi import mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False


import json

import numpy as np
import torch
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

        # --- Corrected frame_id ---
        # Extract the frame_id (filename without extension) from this specific file path
        frame_id = file_path.stem
        # --- End Correction ---

        # Load points based on the file extension
        try:
            if self.ext == ".bin":
                # Assuming 4 columns (e.g., x, y, z, intensity/reflectance)
                points = np.fromfile(file_path_str, dtype=np.float32).reshape(-1, 4)
            elif self.ext == ".npy":
                points = np.load(file_path_str)

            else:
                # This case should ideally not be reached due to __init__ checks,
                # but included for robustness.
                raise NotImplementedError(f"Loading for extension '{self.ext}' is not implemented.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading file {file_path_str}: {e}")
            else:
                print(f"Error loading file {file_path_str}: {e}")
            # Decide how to handle errors: raise, return None, return empty data?
            # Returning an empty dict or raising might be appropriate.
            # For now, let's re-raise the exception.
            raise e

        # Create the dictionary for the data sample
        input_dict = {
            "points": points,
            "frame_id": frame_id,  # Use the frame_id from the specific file
        }

        # Pass the raw data dictionary to the preparation step
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/custom_models/pointrcnn_fine_tune.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/data/custom/points/",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument("--ckpt", type=str, default=None, help="specify the pretrained model")
    parser.add_argument("--ext", type=str, default=".npy", help="specify the extension of your point cloud data file")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")
    # --- Load predictions from JSON ---
    # pred_json_path = "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn_fine_tune/default/eval/epoch_10/val/default/result.json"
    pred_json_path = "/home/vintecc/Vision.Mono/projects/Vision.PointCloudAI/submodules/OpenPcdet/output/custom_models/pointrcnn/Office_Vintecc/eval/val/default/result.json"

    logger.info(f"Loading predictions from: {pred_json_path}")
    try:
        with open(pred_json_path) as f:
            all_predictions_list = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Prediction JSON file not found at {pred_json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {pred_json_path}")
        return

    # Convert list of predictions to a dictionary keyed by frame_id for fast lookup
    predictions_by_frame = {pred["frame_id"]: pred for pred in all_predictions_list}
    # print the frame_id and the number of predictions
    # for frame_id, pred in predictions_by_frame.items():
    #     logger.info(f"Frame ID: {frame_id}, Number of Predictions: {len(pred['boxes_lidar'])}")
    #     print(f"Frame ID: {frame_id}, Number of Predictions: {len(pred['boxes_lidar'])}")

    logger.info(f"Loaded predictions for {len(predictions_by_frame)} frames.")

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )

    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    with torch.no_grad():
        for idx, data_dict_single in enumerate(demo_dataset):
            # --- Get frame_id BEFORE collate_batch ---
            # The frame_id should be present in the dict returned by __getitem__
            current_frame_id = data_dict_single.get("frame_id")
            if current_frame_id is None:
                logger.warning(f"Frame ID missing for sample index: {idx}. Trying filename stem.")
                # Fallback: Recalculate from sample_file_list if needed
                current_frame_id = Path(demo_dataset.sample_file_list[idx]).stem

            logger.info(f"Visualizing sample index: {idx}, frame_id: {current_frame_id}")

            # --- Look up predictions for the current frame ---
            pred_dict_for_frame = predictions_by_frame.get(current_frame_id)

            if pred_dict_for_frame is None:
                logger.warning(
                    f"No predictions found for frame_id: {current_frame_id}. Skipping visualization for this frame."
                )
                pred_boxes = None
                pred_boxes = None
                pred_scores = None
                pred_labels = None
            else:
                # --- Extract prediction data ---
                # Convert lists from JSON to numpy arrays as expected by visualizer
                # Ensure keys match your JSON structure ('boxes_lidar', 'score', 'pred_labels')
                pred_boxes = np.array(pred_dict_for_frame.get("boxes_lidar", []), dtype=np.float32)
                pred_scores = np.array(pred_dict_for_frame.get("score", []), dtype=np.float32)
                pred_labels = np.array(pred_dict_for_frame.get("pred_labels", []), dtype=np.int64)  # Use int for labels

                # print boxes center x,y + prediction score
                for i in range(len(pred_boxes)):
                    # Extract the x and y coordinates of the box center
                    box_center_x = pred_boxes[i][0]
                    box_center_y = pred_boxes[i][1]
                    # Extract the prediction score
                    pred_score = pred_scores[i]
                    # Print the values
                    print(f"Box {i}: Center X: {box_center_x}, Center Y: {box_center_y}, Score: {pred_score}")

                # Handle empty predictions for a frame if keys exist but lists are empty
                if pred_boxes.size == 0:
                    logger.info(f"Frame {current_frame_id} has prediction entry but no boxes.")
                    pred_boxes = None  # Set to None for draw_scenes
                    pred_scores = None
                    pred_labels = None
            # -------------------------------------------

            # --- Prepare data for visualization ---
            # Collate batch (even for a single item) as expected by load_data_to_gpu and potentially V.draw_scenes
            data_dict_batch = demo_dataset.collate_batch([data_dict_single])
            load_data_to_gpu(data_dict_batch)
            # ------------------------------------
            # gt_box are frame_id.txt (2.35 0.07 -0.33 0.14 0.52 1.62 1.59 Pedestrian) files no in the same directory but 1 higher and then in the label as the npy files load them
            # --- Load GT boxes from text files ---
            gt_boxes = []
            gt_boxes_path = Path(args.data_path).parent / "labels" / f"{current_frame_id}.txt"

            try:
                with open(gt_boxes_path) as f:
                    for line in f:
                        # Assuming the format is: x y z l w h ry label
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            # Extract the relevant parts and convert to float
                            x, y, z, l, w, h, ry = map(float, parts[:7])
                            gt_boxes.append([x, y, z, l, w, h, ry])
            except FileNotFoundError:
                logger.warning(f"GT boxes file not found for frame_id: {current_frame_id} at {gt_boxes_path}")

            # # Convert to numpy array if not empty
            if gt_boxes:
                gt_boxes = np.array(gt_boxes, dtype=np.float32)
            else:
                gt_boxes = None

            # --- Call visualizer ---
            V.draw_scenes(
                points=data_dict_batch["points"][:, 1:],  # Pass points from the batch dict
                ref_boxes=pred_boxes,  # Pass the loaded numpy array (or None)
                ref_scores=pred_scores,  # Pass the loaded numpy array (or None)
                ref_labels=pred_labels,  # Pass the loaded numpy array (or None)
                gt_boxes=gt_boxes,  # Pass the loaded numpy array (or None)
            )
            # -----------------------

            if not OPEN3D_FLAG:
                mlab.show(stop=True)  # Keep interaction for Mayavi


if __name__ == "__main__":
    main()
