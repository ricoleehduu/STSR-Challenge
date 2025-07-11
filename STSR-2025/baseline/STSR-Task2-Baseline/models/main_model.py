# models/main_model.py (modified version)

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from .pointnet2 import PointNet2Encoder
from .registration_head import SVDHead

class RegistrationModel(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim

        self.src_encoder = PointNet2Encoder(out_channels=feat_dim)
        self.tgt_encoder = PointNet2Encoder(out_channels=feat_dim)

        self.head = SVDHead(in_dim_src=feat_dim, in_dim_tgt=feat_dim)

    def forward(self, p_src, p_tgt):
        # --- Input and format conversion (no change) ---
        B, N_s, _ = p_src.shape
        _, N_t, _ = p_tgt.shape

        p_src_flat = p_src.view(-1, 3)
        batch_src_orig = torch.arange(B, device=p_src.device).repeat_interleave(N_s)

        p_tgt_flat = p_tgt.view(-1, 3)
        batch_tgt_orig = torch.arange(B, device=p_tgt.device).repeat_interleave(N_t)

        # --- Encoder (now receives correct return values) ---
        f_src_sampled, p_src_sampled, batch_src_sampled = self.src_encoder(p_src_flat, batch_src_orig)
        f_tgt_sampled, p_tgt_sampled, batch_tgt_sampled = self.tgt_encoder(p_tgt_flat, batch_tgt_orig)

        # --- Reshape using correct batch indices ---
        # f_src_sampled: [B * N_s_sampled, C]
        # batch_src_sampled: [B * N_s_sampled]
        # Now their dimensions match!
        f_src, _ = to_dense_batch(f_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)
        p_src_ds, _ = to_dense_batch(p_src_sampled, batch_src_sampled, fill_value=0.0, max_num_nodes=512)

        # Do the same for target
        # Assume target has 512 points after downsampling; adjust max_num_nodes if needed
        f_tgt, _ = to_dense_batch(f_tgt_sampled, batch_tgt_sampled, fill_value=0.0, max_num_nodes=512)
        p_tgt_ds, _ = to_dense_batch(p_tgt_sampled, batch_tgt_sampled, fill_value=0.0, max_num_nodes=512)

        # --- Dimension check ---
        # f_src: [B, 512, C]
        # p_src_ds: [B, 512, 3]
        # f_tgt: [B, 512, C]
        # p_tgt_ds: [B, 512, 3]
        # Dimensions are now correct and dense tensors are ready for use

        # --- SVD Head (input remains unchanged) ---
        transform_pred = self.head(p_src_ds, f_src, p_tgt_ds, f_tgt)

        return transform_pred