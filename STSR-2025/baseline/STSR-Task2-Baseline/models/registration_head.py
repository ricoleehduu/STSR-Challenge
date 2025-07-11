import torch
import torch.nn as nn

class SVDHead(nn.Module):
    def __init__(self, in_dim_src, in_dim_tgt):
        super().__init__()
        # Simple attention mechanism
        self.attention = nn.Linear(in_dim_src, in_dim_tgt)

    def forward(self, p_src, f_src, p_tgt, f_tgt):
        # p_src: (B, N_s, 3), f_src: (B, N_s, C)
        # p_tgt: (B, N_t, 3), f_tgt: (B, N_t, C)

        # Compute soft correspondence
        f_src_att = self.attention(f_src)  # (B, N_s, C_t)
        # bmm: batch matrix multiplication
        affinity = torch.bmm(f_src_att, f_tgt.transpose(1, 2))  # (B, N_s, N_t)
        prob = torch.softmax(affinity, dim=2)

        # Compute expected corresponding points
        p_corr = torch.bmm(prob, p_tgt)  # (B, N_s, 3)

        # Compute weighted centroids
        p_src_centroid = torch.mean(p_src, dim=1, keepdim=True)
        p_corr_centroid = torch.mean(p_corr, dim=1, keepdim=True)

        p_src_centered = p_src - p_src_centroid
        p_corr_centered = p_corr - p_corr_centroid

        # Build cross-covariance matrix
        H = torch.bmm(p_src_centered.transpose(1, 2), p_corr_centered)  # (B, 3, 3)

        # SVD decomposition
        try:
            U, _, V = torch.svd(H, some=False, compute_uv=True)
        except torch.linalg.LinAlgError:
            # SVD may not converge, return identity matrix
            identity = torch.eye(3).to(H.device).unsqueeze(0).repeat(H.shape[0], 1, 1)
            return identity, torch.zeros(H.shape[0], 3).to(H.device)

        R = torch.bmm(V, U.transpose(1, 2))

        # Fix possible reflections
        det = torch.det(R)
        diag = torch.tensor([1.0, 1.0, -1.0], device=R.device)
        V_prime = V * diag
        R_det_neg = torch.bmm(V_prime, U.transpose(1, 2))
        R = torch.where(det.view(-1, 1, 1) < 0, R_det_neg, R)

        t = p_corr_centroid.squeeze(1) - torch.bmm(R, p_src_centroid.transpose(1, 2)).squeeze(-1)

        # Assemble 4x4 transformation matrix
        transform = torch.eye(4, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t

        return transform