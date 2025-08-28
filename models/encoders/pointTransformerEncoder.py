import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_utils import PointNetFeaturePropagation, PointNetSetAbstraction, index_points, square_distance
import numpy as np
#from .transformer import TransformerBlock

#The Point Transformer Encoder
#The encoder  follow a standard hierarchical structure:
#Initial Embedding: A small MLP (fc1) to lift the input features (e.g., 3D coordinates) into a higher-dimensional space.
# Downsampling & Feature Aggregation: A series of TransitionDown and TransformerBlock modules to hierarchically downsample the point cloud and capture features at different scales.
# Global Feature Aggregation: A final pooling layer (e.g., global average or max pooling) to get a single feature vector for the entire point cloud.
# Latent Space Mapping: Two separate fully connected networks to map the global feature vector to the mean and log-variance vectors.


class PointTransformerEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, cfg=None):
        super().__init__()
        # We'll use a hardcoded config for demonstration if none is provided.
        if cfg is None:
            class DummyConfig:
                def __init__(self):
                    self.num_point = 500
                    self.input_dim = 3
                    self.nblocks = 4
                    self.nneighbor = 16
                    self.transformer_dim= 128 #512 kill my gpu, this could be a bottleneck to overcome
                    self.input_dim = input_dim
            cfg = DummyConfig()
        npoints, nblocks, nneighbor, transformer_dim, d_points = cfg.num_point, cfg.nblocks, cfg.nneighbor, cfg.transformer_dim, cfg.input_dim

        self.zdim = zdim
        
        # Initial feature embedding
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        # Hierarchical downsampling and feature learning
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(
                npoints // 4 ** (i + 1),
                nneighbor,
                [channel // 2 + 3, channel, channel]
            ))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

        # Final layers to map to the latent space (mean and log-variance)
        # The output feature size from the last TransformerBlock will be 32 * 2**nblocks
        final_feature_dim = 32 * 2 ** nblocks
        
        # Networks for latent mean (m)
        self.fc_m1 = nn.Linear(final_feature_dim, 256)
        self.fc_m2 = nn.Linear(256, zdim)
        
        # Networks for latent log-variance (v)
        self.fc_v1 = nn.Linear(final_feature_dim, 256)
        self.fc_v2 = nn.Linear(256, zdim)

    def forward(self, x):
        # x: (B, N, 3)
        xyz = x
        
        # 1. Initial feature embedding
        # We pass the same coordinates for xyz and features to the first TransformerBlock
        points = self.transformer1(xyz, self.fc1(xyz))[0]

        # 2. Hierarchical downsampling
        for i, (td, tb) in enumerate(zip(self.transition_downs, self.transformers)):
            xyz, points = td(xyz, points)
            points = tb(xyz, points)[0]

        # 3. Global feature aggregation
        # Global max pooling across all points
        global_features = torch.max(points, dim=1)[0]
        
        # 4. Map to latent space
        m = F.relu(self.fc_m1(global_features))
        m = self.fc_m2(m)
        
        v = F.relu(self.fc_v1(global_features))
        v = self.fc_v2(v)
        
        return m, v


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        # xyz: bxnx3, features: bxnx32 = xyz*linear(3,32)
        #dist: bxnxn = bx(n_{i,j}= square distance n_i to n_j) (why not square-root?)
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k = bx(firs_k(distance n_i to k n_j ))
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3 = bx(3 dim coordinates of k closest points to n_i)
        # in simplest words, each point n_i has attached to it is closest k neightbors
        pre = features # (xyz: bxnxd * linear(d,32)=bxnx32)
        #projection to attention dim: bxnx32*linear(32xd_model:512)=bxnxd_model
        x = self.fc1(features) 
        # q : (x:bxnx512)*(linear(d_model:d_model,d_model)) = bxnxd_model
        # k : index_points(w_ks(x):bxnx512, knn_idx) = bxnx(k indexes of closes k points to poin n_i)xd_model
        # v isomorphic k
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        #self.dc_delta iso linear(3,d_model:512)
        # xyzk :=(xyz[:,:,None]:bxnx1x3-knn_xyz):bxnxkx3= bxnxkx3 (Per each point position, substracts the position of k closer points (?))
        #self.dc_delta(xyzk): bxnxkx3* linear(3,512)=bxnxkxd_model. Projection of xyzk into attention dim
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model
        # pos_enc is pretty much the projection of knn_xyz into attention dim. 
        # self.fc_gamma iso linear( d_model, d_model)
        # q[:, :, None]:bxnx1xk - k:bxnx[k]xd_model + pos_enc:b x n x k x d_model = b x n x k x d_model
        # attn = bxnxkxd_model*linear(d_model, d_model): bxnxkxd_model
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # einstein sumation over k dim: bxnxkxd_model*bxnxkxd_model-> bxnxd_model
        # res attn*(v+pos_enc)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # res:bxnxd_model*linear(d_model,d_points) = bxnx(d_points=32)
        res = self.fc2(res) + pre
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)



class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2

