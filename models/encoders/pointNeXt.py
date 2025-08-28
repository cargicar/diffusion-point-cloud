import torch
import torch.nn as nn
from torch_geometric.nn import PointNeXt
import torch.nn.functional as F

class PointNeXtEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, num_classes=None):
        super().__init__()
        self.zdim = zdim
        
        # PointNeXt is a complete classification network. We'll repurpose its
        # feature extraction layers. Here, we define the PointNeXt model
        # and then extract the layers we need.
        self.pointnext = PointNeXt(
            in_channels=input_dim,
            out_channels=zdim,  # We'll use the latent space as the final output dim
            num_classes=num_classes, # Pass None if not doing classification directly
            # The following are default values from the original PointNeXt paper
            num_layers=4,
            sa_channels=[32, 64, 128, 256],
            fp_channels=[256, 256]
        )
        
        # PointNeXt's final output is a global feature vector. We will use this
        # to generate the mean and log-variance for our VAE latent space.
        
        # Linear layers for mapping to latent space mean (m)
        self.fc1_m = nn.Linear(self.pointnext.out_channels, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        
        # Linear layers for mapping to latent space log-variance (v)
        self.fc1_v = nn.Linear(self.pointnext.out_channels, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        
    def forward(self, data):
        # PointNeXt typically uses a torch_geometric.data.Data object
        # which can handle the x (features) and pos (positions)
        # If your data is just a tensor (B, N, 3), you'll need to create this object.
        x, pos = data.x, data.pos
        
        # The PointNeXt forward pass returns both the global features and per-point features
        global_features = self.pointnext(x, pos)
        
        # Use the global features to get mean and log-variance
        m = F.relu(self.fc1_m(global_features))
        m = F.relu(self.fc2_m(m))
        m = self.fc3_m(m)
        
        v = F.relu(self.fc1_v(global_features))
        v = F.relu(self.fc2_v(v))
        v = self.fc3_v(v)
        
        return m, v