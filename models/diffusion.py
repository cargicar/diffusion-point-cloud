import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .common import *


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

"""
This module, VarianceSchedule, is a crucial component of a Denoising Diffusion Probabilistic Model (DDPM). Its sole purpose is to pre-compute and manage the noise levels for each time step in the diffusion process. It's a static part of the model that doesn't learn anything during training; it just provides the necessary parameters for both adding noise (the forward process) and removing noise (the reverse process). 📈

The __init__ Method (Pre-computation)

The constructor (__init__) is where all the work happens. It takes the number of steps (num_steps), the starting variance (beta_1), and the ending variance (beta_T) and pre-computes a schedule of various parameters that will be used throughout the diffusion process.

    Beta (βt​): The core of the schedule. The code creates a linear schedule of betas, which are the noise variances added at each time step t. The variance starts at beta_1 and increases linearly to beta_T. A zero is padded at the beginning so the indexing aligns with the time steps (from 1 to T).

    Alpha (αt​): The code calculates αt​=1−βt​. This value represents the signal retention at each time step.

    Alpha Bar (αˉt​): This is the cumulative product of all alphas up to time t, i.e., αˉt​=∏i=1t​αi​. This is a critical value used in the forward process to add noise to a clean image in a single step. The code computes this efficiently using log_alphas.

    Sigmas (σt​): The sigmas are the variances used in the reverse (denoising) process. The code computes two types:

        sigmas_flex: Represents the maximum possible variance for the reverse step.

        sigmas_inflex: Represents the minimum possible variance.

        The flexibility parameter in the get_sigmas method allows for a smooth interpolation between these two values during sampling, which can influence the quality and diversity of the generated samples.

These computed schedules are stored as buffers (self.register_buffer), which are tensors that are part of the model's state but are not considered trainable parameters.

The uniform_sample_t Method

This is a utility function used during training. It simply returns a batch of randomly sampled time steps t from 1 to num_steps. This ensures the model learns to predict noise at all stages of the diffusion process, not just at the beginning or end.

The get_sigmas Method

This method is used during the generation process (sampling). It takes a time step t and a flexibility parameter and returns a variance value that is a weighted average between the minimum (sigmas_inflex) and maximum (sigmas_flex) variances. This gives some control over the stochasticity of the reverse diffusion process.


"""
class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out
"""This module, PointwiseNet, is a neural network designed to transform point cloud data in the context of a denoising diffusion model. Its primary job is to predict the noise that has been added to a point cloud at a specific time step.

The network processes each point in the cloud individually, making it a "point-wise" network, and uses information about the current time step and a shape latent vector to inform its prediction.

How the Module Works

The network's behavior can be broken down into these key steps:

    Input Processing: The forward method takes three inputs:

        x: The point cloud itself, with shape (Batch_size, Num_points, Point_dimension).

        beta: A scalar representing the current time step t. The network uses this to understand how much noise has been added.

        context: A shape latent vector, likely from an encoder, with shape (Batch_size, Latent_dimension). This vector tells the network about the overall shape it should be working with.

    Time and Context Embedding:

        The beta (time) and context (shape latent) inputs are reshaped and then combined.

        The beta value is converted into a time embedding by concatenating beta, sin(beta), and cos(beta). This gives the network a richer, periodic representation of time instead of just a single scalar.

        This time embedding is then concatenated with the context vector to create a single conditional embedding (ctx_emb) that contains all the necessary information about the current time step and the target shape.

    Point-wise Processing with Conditional Layers:

        The core of the network is a series of ConcatSquashLinear layers. This is a special type of linear layer that takes not only the input points (x) but also the conditional embedding (ctx_emb).

        The ConcatSquashLinear layer processes each point independently. It takes a point from x and uses the ctx_emb to "modulate" its weights, allowing the network's behavior to be conditioned on the time step and shape.

        Each layer's output is passed through a Leaky ReLU activation function, adding non-linearity to the model.

    Residual Connection:

        The network has an option to use a residual connection, determined by the self.residual flag.

        If enabled, the final output of the network is added back to the original input x. This is a common technique in deep learning that helps the network learn the change or noise that needs to be added or removed, rather than learning the entire output from scratch. This makes training more stable and effective.

In summary, this PointwiseNet is a conditional network that takes a noisy point cloud and uses a time signal and a shape latent vector to predict the noise that needs to be removed. It processes each point independently, making it a highly parallelizable and efficient architecture for point cloud tasks.
 
"""
class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

"""
This module, DiffusionPoint, is an implementation of a denoising diffusion probabilistic model (DDPM) for point clouds. Its core function is to learn a reverse diffusion process to generate new point clouds. The module contains two primary methods: get_loss for training and sample for generation.

The module works by progressively adding noise to a point cloud and then learning to reverse that process step-by-step. During training, it learns to predict the noise added at a random time step. During generation, it starts with pure random noise and iteratively removes the predicted noise until a coherent point cloud emerges.

1. The __init__ Method

The __init__ method initializes the two core components of the diffusion model:

    self.net: This is the noise prediction network (likely the PointwiseNet you saw before). Its job is to take a noisy point cloud and predict the amount of noise that was added to it.

    self.var_sched: This is the variance schedule. It defines the noise levels (βt​ and αt​) for each time step t in the diffusion process. It is a crucial component that controls the amount of noise added or removed at each step.

2. The get_loss Method (Training)

This method defines a single step of the training process. The goal is to train the self.net to accurately predict noise.

    Noise Addition (Forward Process): It takes a clean point cloud x_0 and a random time step t. The var_sched is used to get the appropriate alpha_bar and beta values for this time step.

    Creating a Noisy Point Cloud: A random noise tensor e_rand is generated. The x_t noisy point cloud is then created using the formula: xt​=αˉt​​x0​+1−αˉt​​ϵ, where ϵ is the random noise. This is a key step in diffusion models; it adds a specific amount of noise to the clean point cloud.

    Noise Prediction: The noisy point cloud x_t is passed to the noise prediction network self.net, which attempts to predict the noise that was just added.

    Calculating Loss: A Mean Squared Error (MSE) loss is calculated between the predicted noise (e_theta) and the actual random noise (e_rand). The model is trained to minimize this loss, which teaches it to predict the noise component.

3. The sample Method (Generation)

This method outlines the reverse diffusion process, where a new point cloud is generated from scratch.

    Start with Noise: The process begins with a batch of pure random noise x_T, representing a point cloud at the final time step T.

    Denoising Loop: The code then iterates backward from t = T down to t = 1. In each step, it performs the following:

        It uses the noise prediction network (self.net) to predict the noise e_theta in the current noisy point cloud x_t.

        It uses a mathematical formula (the reverse diffusion step) to subtract the predicted noise and get a slightly cleaner point cloud x_{t-1}.

        This formula effectively calculates a less noisy version of the point cloud based on the noise prediction.

        It also adds a new, small amount of random noise z (unless it's the last step) to ensure the generated samples are diverse.

    Final Output: After iterating through all the time steps, the final tensor traj[0] is a clean, generated point cloud that resembles the training data. The ret_traj option allows you to save and visualize the entire denoising process.


"""
