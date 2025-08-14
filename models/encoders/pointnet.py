import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)
        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v

""" 
This PyTorch module is a PointNet encoder, a neural network architecture designed to process point cloud data (sets of 3D points). Its primary function is to transform an unordered set of points into a compressed, meaningful feature vector.

The code implements a common variant of PointNet used in generative models like Variational Autoencoders (VAEs), where the encoder outputs both a mean and a log-variance vector for a latent space distribution.

How the Encoder Works

The forward pass of the network can be broken down into a few key steps:

    Input Transformation: The input x is a point cloud with shape (batch_size, num_points, 3). The first thing the model does is x.transpose(1, 2), which changes the shape to (batch_size, 3, num_points). This is a crucial step because the nn.Conv1d layers expect the channels (the 3 coordinates) to be the second dimension.

    Point-wise Feature Extraction:

        The network applies a series of four 1D convolutional layers (conv1 through conv4). A 1D convolution on a point cloud is applied to each point individually and independently. This means it learns features for each point using its (x, y, z) coordinates, but without considering the relationships between points yet.

        Each convolution is followed by a Batch Normalization (bn) layer and a ReLU activation function, which helps stabilize training and introduce non-linearity.

        This process progressively increases the number of channels (feature dimensions) from 3 to 128, then to 256, and finally to 512, building a rich set of features for each point.

    Symmetric Pooling (Global Feature Aggregation):

        The key innovation of PointNet is its ability to handle unordered point clouds. It achieves this by applying a symmetric function to aggregate the point-wise features.

        The line x = torch.max(x, 2, keepdim=True)[0] performs max-pooling across all points. It takes the maximum value from each of the 512 feature channels, effectively aggregating the most important feature from the entire point cloud.

        The result is a single global feature vector of shape (batch_size, 512), which is invariant to the order of the input points.

    Latent Space Mapping:

        After the max-pooling step, the global feature vector is flattened into a shape of (batch_size, 512).

        The code then splits into two identical branches: one for calculating the mean (m) and one for the log-variance (v) of a distribution.

        Each branch uses a multi-layer perceptron (a series of nn.Linear layers with Batch Normalization and ReLU activations) to map the 512-dimensional global feature vector down to a lower-dimensional latent space of size zdim.

    Output: The final output of the module is a tuple (m, v), representing the mean and log-variance of the point cloud's distribution in the latent space. This is a typical output format for the encoder in a Variational Autoencoder (VAE), which can then be used to sample and reconstruct new point clouds.

"""
