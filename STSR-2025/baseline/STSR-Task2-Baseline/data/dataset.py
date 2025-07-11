# data/dataset.py (modified version)

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import trimesh
import nibabel as nib
from .transforms import RandomRigidTransform

class DentalDataset(Dataset):
    def __init__(self, data_root, jaw_type, num_points_stl=4096, num_points_cbct=8192, use_augmentation=False):
        assert jaw_type in ['lower', 'upper'], "jaw_type must be 'lower' or 'upper'"
        self.data_root = Path(data_root)
        self.jaw_type = jaw_type
        self.num_points_stl = num_points_stl
        self.num_points_cbct = num_points_cbct
        self.use_augmentation = use_augmentation
        self.transform_aug = RandomRigidTransform(mag_trans=0.1, mag_rot=30)

        self.image_dir = self.data_root / "Images"
        self.label_dir = self.data_root / "Labels"
        self.case_folders = sorted([p for p in self.image_dir.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.case_folders)

    def __getitem__(self, idx):
        case_folder = self.case_folders[idx]
        patient_id = case_folder.name

        # --- 1. Load and process STL (Source) --- (no change)
        stl_path = case_folder / f"{self.jaw_type}.stl"
        mesh = trimesh.load(str(stl_path))
        p_src, _ = trimesh.sample.sample_surface(mesh, self.num_points_stl)
        
        # --- 2. Load and process CBCT (Target) --- (no change)
        cbct_path = case_folder / "CBCT.nii.gz"
        cbct_img = nib.load(str(cbct_path))
        cbct_data = cbct_img.get_fdata()
        
        coords = np.argwhere(cbct_data > 800)
        coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
        p_tgt = (cbct_img.affine @ coords_h.T).T[:, :3]

        if p_tgt.shape[0] > self.num_points_cbct:
            indices = np.random.choice(p_tgt.shape[0], self.num_points_cbct, replace=False)
            p_tgt = p_tgt[indices]
        elif p_tgt.shape[0] < self.num_points_cbct:
             # If not enough points, upsample with replacement
            indices = np.random.choice(p_tgt.shape[0], self.num_points_cbct, replace=True)
            p_tgt = p_tgt[indices]

        # --- 3. Load GT transformation matrix --- (no change)
        gt_path = self.label_dir / patient_id / f"{self.jaw_type}_gt.npy"
        transform_gt = np.load(str(gt_path)).astype(np.float32)

        # --- 4. Data normalization and augmentation --- (no change)
        center = p_tgt.mean(axis=0)
        p_src -= center
        p_tgt -= center
        
        scale = np.max(np.linalg.norm(p_tgt, axis=1))  # Use norm for more stability
        p_src /= scale
        p_tgt /= scale
        
        transform_gt[:3, 3] -= center
        transform_gt[:3, 3] /= scale

        if self.use_augmentation:
            p_src_aug, transform_aug = self.transform_aug(p_src)
            transform_gt = transform_gt @ np.linalg.inv(transform_aug)
            p_src = p_src_aug

        # --- 5. Return Tensors --- (now simpler)
        return {
            "p_src": torch.from_numpy(p_src).float(),
            "p_tgt": torch.from_numpy(p_tgt).float(),
            "transform_gt": torch.from_numpy(transform_gt).float(),
        }