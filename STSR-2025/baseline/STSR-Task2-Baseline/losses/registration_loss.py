import torch
import torch.nn.functional as F


def transform_points(points, transform):
    points_h = F.pad(points, (0, 1), mode='constant', value=1.0)
    transformed_points_h = torch.bmm(points_h, transform.transpose(1, 2))
    return transformed_points_h[:, :, :3]


def registration_loss(p_src, transform_pred, transform_gt):
    p_src_pred = transform_points(p_src, transform_pred)
    p_src_gt = transform_points(p_src, transform_gt)

    loss = F.mse_loss(p_src_pred, p_src_gt)
    return loss
