"""Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""

import matplotlib
import numpy as np
import open3d.visualization
import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[: max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(
    points,
    gt_boxes=None,
    ref_boxes=None,
    ref_labels=None,
    ref_scores=None,
    point_colors=None,
    draw_origin=True,
    ref_colors=None,
):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    # ### <<< ADDED: Handle pred_box_colors if it's a tensor >>> ###
    if isinstance(ref_colors, torch.Tensor):
        ref_colors = ref_colors.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 7
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        # Default to white if no point colors provided
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        # Ensure point_colors is numpy and correct shape
        if isinstance(point_colors, torch.Tensor):
            point_colors = point_colors.cpu().numpy()
        # Assuming point_colors is Nx3
        pts.colors = open3d.utility.Vector3dVector(point_colors[:, :3])

    # --- Draw GT Boxes (Unchanged) ---
    if gt_boxes is not None:
        # Assuming draw_box takes a color tuple (0,0,1) -> Blue for GT
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    # --- Draw Reference Boxes (MODIFIED BLOCK) ---
    if ref_boxes is not None:
        # ### <<< REMOVED OLD LINE: vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores) >>> ###

        # ### <<< ADDED LOOP AND COLOR LOGIC >>> ###
        default_ref_color = (0, 1, 0)  # Original green color as default

        for i in range(ref_boxes.shape[0]):
            box_color = default_ref_color  # Start with default

            # Check if custom colors are provided and valid for this index
            if ref_colors is not None and i < len(ref_colors):
                # Use the provided color, ensure it's a tuple/list
                box_color = tuple(ref_colors[i])

            # Prepare single box, label, score for the draw_box call
            # Pass the box as a (1, 7) array, assuming draw_box handles this shape
            single_box = ref_boxes[i : i + 1, :]

            # Handle optional labels and scores for the single box
            single_label = [ref_labels[i]] if ref_labels is not None and i < len(ref_labels) else None
            single_score = [ref_scores[i]] if ref_scores is not None and i < len(ref_scores) else None

            # Call draw_box for the individual box with its specific color
            # Pass labels/scores as lists (or None) assuming draw_box expects iterables
            vis = draw_box(vis, single_box, box_color, single_label, single_score)
        # ### <<< END MODIFIED BLOCK >>> ###

    vis.run()
    vis.destroy_window()


# Keep translate_boxes_to_open3d_instance as it is
def translate_boxes_to_open3d_instance(gt_boxes):
    """4-------- 6
     /|         /|
    5 -------- 3 .
    | |        | |
    . 7 -------- 1
    |/         |/
    2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # Corrected indices for standard bbox lines (optional, original might be fine too)
    # Default create_from_oriented_bounding_box lines:
    # 0-1, 1-3, 3-2, 2-0 (bottom face)
    # 4-5, 5-7, 7-6, 6-4 (top face)
    # 0-4, 1-5, 2-6, 3-7 (vertical edges)
    # No change needed unless the original concatenation was fixing a specific issue.
    # Keeping the original concatenation for consistency with provided code:
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)  # Keep original logic
    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


# --- MODIFIED draw_box ---
def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    # This loop will run once per call from the modified draw_scenes,
    # as gt_boxes will have shape (1, 7)
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        # === MODIFIED COLORING LOGIC ===
        # Always use the 'color' argument passed into this function.
        # The logic using box_colormap based on ref_labels is removed.
        line_set.paint_uniform_color(color)
        # ===============================

        vis.add_geometry(line_set)

        # add arrow for orientation (logic unchanged)
        arrow_length = box3d.extent[0] * 0.5
        cylinder_radius = arrow_length * 0.05
        cone_radius = arrow_length * 0.1
        cylinder_height = arrow_length * 0.75
        cone_height = arrow_length * 0.25

        arrow = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
            resolution=10,
        )
        arrow.paint_uniform_color((1, 0, 0))  # Red color for arrow

        # --- Rotation Logic (unchanged from your provided code) ---
        rotate_axis = np.array([0, 1, 0])
        rotate_angle = np.pi / 2
        axis_angle_vector = rotate_axis * rotate_angle
        rotate_Z_to_X = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle_vector)
        arrow.rotate(rotate_Z_to_X, center=(0, 0, 0))
        arrow.rotate(box3d.R, center=(0, 0, 0))
        # --- End Rotation Logic ---

        # --- Translation Logic (unchanged from your provided code) ---
        forward_vec = box3d.R[:, 0]
        front_center = box3d.center + forward_vec * (box3d.extent[0] / 2.0)
        arrow.translate(front_center, relative=False)
        # --- End Translation Logic ---
        vis.add_geometry(arrow)

        # Score text logic remains commented out as in your original code.
        # If you want to enable it, ensure 'score' is handled correctly (it's passed as a list).
        # Example check:
        # if score is not None and i < len(score) and score[i] is not None:
        #     current_score = score[i] # Get the actual score value
        #     corners = box3d.get_box_points()
        #     score_text = f"{current_score:.2f}" # Use f-string
        #     center = np.mean(np.asarray(corners), axis=0)
        #
        #     # Consider using Add3DLabel if available and simpler, or refine the TriangleMesh text approach
        #     # vis.add_3d_label(center + [0,0,0.5], score_text) # Example placement adjustment
        #
        #     # --- Text Mesh Logic (from original, needs verification/update) ---
        #     # try:
        #     #     # Requires Open3D >= 0.10.0 for t.geometry
        #     #     text_mesh = open3d.t.geometry.TriangleMesh.create_text(score_text, depth=0.1).to_legacy() # Adjust depth?
        #     #     vertices = np.asarray(text_mesh.vertices)
        #     #     if vertices.size > 0: # Check if text mesh creation succeeded
        #     #         mesh_center = np.mean(vertices, axis=0)
        #     #         text_mesh.scale(0.1, center=mesh_center) # Scale text
        #     #         text_mesh.paint_uniform_color((1.0, 1.0, 1.0)) # White text
        #     #
        #     #         # Simpler translation to box center (adjust Z offset as needed)
        #     #         text_translation = center - mesh_center
        #     #         text_mesh.translate(text_translation, relative=True)
        #     #         text_mesh.translate([0, 0, box3d.extent[2] / 2 + 0.1]) # Move slightly above the box center Z
        #     #
        #     #         # Add text to the visualization
        #     #         vis.add_geometry(text_mesh)
        #     # except AttributeError:
        #     #      print("Warning: open3d.t.geometry.TriangleMesh.create_text not available (requires Open3D >= 0.10). Skipping score text.")
        #     # except Exception as e:
        #     #      print(f"Warning: Error creating score text: {e}")
        #     # --- End Text Mesh Logic ---
        #     pass # End of score text block

    return vis
