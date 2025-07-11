# models/pointnet2.py

import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius, PointNetConv


# Set Abstraction (SA) Module
class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        """
        Set Abstraction Module.

        Args:
            ratio (float): Sampling ratio for furthest point sampling.
            r (float): Radius for neighborhood search.
            nn (torch.nn.Module): Mini-PointNet (usually an MLP) for local feature extraction.
        """
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        """
        Args:
            x (Tensor): Input features of points [N, C_in]. Can be None for the first layer.
            pos (Tensor): Positions of input points [N, 3].
            batch (Tensor): Batch indices for each point [N].

        Returns:
            Tensor: Features of new sampled points [N_new, C_out].
            Tensor: Positions of new sampled points [N_new, 3].
            Tensor: Batch indices of new sampled points [N_new].
        """
        # 1. Sampling
        # fps returns the indices of the sampled points
        idx = fps(pos, batch, ratio=self.ratio)

        # 2. Grouping
        # radius_graph creates edges between center points (from idx) and points within the radius
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # 3. Local Feature Extraction
        # PointNetConv will perform message passing on the created graph
        # If x is None, it uses position `pos` as features.
        x_dest = self.conv(x, (pos, pos[idx]), edge_index)

        # The new positions and batch indices are those of the sampled points
        pos_new, batch_new = pos[idx], batch[idx]

        return x_dest, pos_new, batch_new


# models/pointnet2.py (continued)

class PointNet2Encoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # PointNet++ is a hierarchy of Set Abstraction modules.
        # We define a sequence of such modules.

        # --- Feature dimensions ---
        # The output dimension of the final layer will be `out_channels`.
        # We can define intermediate channel dimensions.
        c_in = 3  # Input is just position
        c1 = 32
        c2 = 64
        c3 = out_channels

        # --- Layer 1 ---
        # Samples 512 points, finds neighbors in a 0.2 radius.
        # MLP maps (c_in + 3) -> c1
        self.sa1_module = SAModule(0.5, 0.2,
                                   nn=nn.Sequential(nn.Linear(c_in + 3, c1), nn.ReLU(), nn.Linear(c1, c1)))

        # --- Layer 2 ---
        # Samples 128 points from the 512, finds neighbors in a 0.4 radius.
        # MLP maps (c1 + 3) -> c2
        self.sa2_module = SAModule(0.25, 0.4,
                                   nn=nn.Sequential(nn.Linear(c1 + 3, c2), nn.ReLU(), nn.Linear(c2, c2)))

        # --- Layer 3 (Global) ---
        # Samples all remaining points (ratio=1.0 means no downsampling if not needed)
        # It aggregates features from all 128 points.
        self.sa3_module = SAModule(1.0, 1.0,  # r=1.0 and ratio=1.0 to group all points
                                   nn=nn.Sequential(nn.Linear(c2 + 3, c3), nn.ReLU(), nn.Linear(c3, c3)))

    def forward(self, pos, batch):
        # Layer 1
        x1, pos1, batch1 = self.sa1_module(pos, pos, batch)

        # Layer 2
        x2, pos2, batch2 = self.sa2_module(x1, pos1, batch1)

        # Layer 3
        x3, pos3, batch3 = self.sa3_module(x2, pos2, batch2)  # SAModule returns 3 values

        # Return final features, positions, and corresponding batch indices
        return x3, pos3, batch3