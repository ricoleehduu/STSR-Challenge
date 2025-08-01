# dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import trimesh
import nibabel as nib
from .transforms import RandomRigidTransform


def compute_aabb(points):
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    center = (min_corner + max_corner) / 2.0

    return min_corner, max_corner, center


def compute_mesh_bounding_box(mesh, obb=False):
    if obb:
        bbox = mesh.bounding_box_oriented
    else:
        bbox = mesh.bounding_box

    vertices = bbox.vertices
    center = bbox.centroid
    return bbox, vertices, center

def chamfer_distance(p1, p2):
    dist = torch.cdist(p1, p2)  # (B, N, M)
    min_dist_p1_to_p2, _ = dist.min(dim=2)  # (B, N)
    min_dist_p2_to_p1, _ = dist.min(dim=1)  # (B, M)
    chamfer_dist = min_dist_p1_to_p2.mean(dim=1) + min_dist_p2_to_p1.mean(dim=1)
    return chamfer_dist

def center_align_points(points, center):
    return points - center


class DentalDataset(Dataset):
    def __init__(self, data_root, jaw_type, num_points_stl=4096, num_points_cbct=8192,
                 use_augmentation=False, has_labels=True):
        assert jaw_type in ['lower', 'upper'], "jaw_type must be 'lower' or 'upper'"
        self.data_root = Path(data_root)
        self.jaw_type = jaw_type
        self.num_points_stl = num_points_stl
        self.num_points_cbct = num_points_cbct
        self.use_augmentation = use_augmentation
        self.has_labels = has_labels
        self.transform_aug = RandomRigidTransform(mag_trans=0.1, mag_rot=30)
        self.image_dir = self.data_root / "Images"

        if self.has_labels:
            self.label_dir = self.data_root / "Labels"

        self.case_folders = sorted([
            p for p in self.image_dir.iterdir()
            if p.is_dir() and (p / f"{self.jaw_type}.stl").exists()
        ])

    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        patient_id = case_folder.name

        # --- 1. Load and process STL (Source) --- (no change)
        stl_path = case_folder / f"{self.jaw_type}.stl"
        mesh = trimesh.load(str(stl_path))

        # 计算STL的边界框和中心点
        stl_min, stl_max, stl_center = compute_aabb(mesh.vertices)

        # 采样点云并中心对齐
        p_src, _ = trimesh.sample.sample_surface(mesh, self.num_points_stl)
        p_src = center_align_points(p_src, stl_center)

        # --- 2. Load and process CBCT (Target) --- (no change)
        cbct_path = case_folder / "CBCT.nii.gz"
        cbct_img = nib.load(str(cbct_path))
        cbct_data = cbct_img.get_fdata()


        coords = np.argwhere(cbct_data > 800)
        coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
        p_tgt = (cbct_img.affine @ coords_h.T).T[:, :3]

        cbct_min, cbct_max, cbct_center = compute_aabb(p_tgt)

        p_tgt = center_align_points(p_tgt, cbct_center)

        if p_tgt.shape[0] > self.num_points_cbct:
            indices = np.random.choice(p_tgt.shape[0], self.num_points_cbct, replace=False)
            p_tgt = p_tgt[indices]
        elif p_tgt.shape[0] < self.num_points_cbct:
            indices = np.random.choice(p_tgt.shape[0], self.num_points_cbct, replace=True)
            p_tgt = p_tgt[indices]

        # --- 3. Load GT transformation matrix --- (no change)
        transform_gt = np.eye(4).astype(np.float32)
        has_gt = False

        if self.has_labels:
            gt_path = self.label_dir / patient_id / f"{self.jaw_type}_gt.npy"
            im_path = self.image_dir / patient_id
            if gt_path.exists():
                transform_gt = np.load(str(gt_path)).astype(np.float32)
                has_gt = True
            elif gt_path.exists() and im_path.exists():
                print(f"Warning: Missing GT for {patient_id}, using identity matrix")

        # --- 4. Data normalization and augmentation --- (no change)
        scale = np.max(np.linalg.norm(p_tgt, axis=1))
        p_src /= scale
        p_tgt /= scale

        if has_gt:
            center_offset = cbct_center - stl_center
            transform_gt[:3, 3] += center_offset

            transform_gt[:3, 3] /= scale

        if self.use_augmentation:
            p_src_aug, transform_aug = self.transform_aug(p_src)
            if has_gt:
                transform_gt = transform_gt @ np.linalg.inv(transform_aug)
            p_src = p_src_aug

        # --- 5. Return Tensors ---
        return {
            "p_src": torch.from_numpy(p_src).float(),
            "p_tgt": torch.from_numpy(p_tgt).float(),
            "transform_gt": torch.from_numpy(transform_gt).float(),
            "has_gt": has_gt,
            "patient_id": patient_id,
            "stl_center": torch.from_numpy(stl_center).float(),
            "cbct_center": torch.from_numpy(cbct_center).float(),
            "stl_bbox_min": torch.from_numpy(stl_min).float(),
            "stl_bbox_max": torch.from_numpy(stl_max).float(),
            "cbct_bbox_min": torch.from_numpy(cbct_min).float(),
            "cbct_bbox_max": torch.from_numpy(cbct_max).float(),
            "scale": torch.tensor(scale).float()
        }