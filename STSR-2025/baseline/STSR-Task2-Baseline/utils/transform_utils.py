# utils/transform_utils.py

import numpy as np
import torch


def recover_original_coordinates(points, center, scale):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()

    scaled_points = points * scale
    original_points = scaled_points + center
    return original_points


def recover_original_transform(transform, stl_center, cbct_center, scale):

    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    if isinstance(stl_center, torch.Tensor):
        stl_center = stl_center.cpu().numpy()
    if isinstance(cbct_center, torch.Tensor):
        cbct_center = cbct_center.cpu().numpy()

    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale

    center_matrix = np.eye(4)
    center_matrix[:3, 3] = cbct_center - stl_center

    scale_inv_matrix = np.linalg.inv(scale_matrix)
    original_transform = center_matrix @ scale_matrix @ transform @ scale_inv_matrix

    return original_transform