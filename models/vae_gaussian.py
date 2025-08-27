import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *


class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)

        self.cat_embed = nn.Sequential(
            nn.Embedding(args.num_classes, args.latent_dim),
            MLP(
                args.latent_dim,
                int(args.mlp_ratio * args.latent_dim),
                act_layer=args.latent_dim,
                drop=0.1,
            ),
        )
        

        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        
    def get_loss(self, x, y, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        #Get category embeddings and add to z
        cat_embed = self.cat_embedding(y)  # (B, F)
        z_conditioned = z + cat_embed # Adding is a common approach
        
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()
        #Pass the conditioned latent code to the diffusion model
        #loss_recons = self.diffusion.get_loss(x, z)
        loss_recons = self.diffusion.get_loss(x, z_conditioned)
        

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    # def sample(self, z, y, num_points, flexibility, truncate_std=None):
    #     """
    #     Args:
    #         z:  Input latent, normal random samples with mean=0 std=1, (B, F)
    #     """
    #     if truncate_std is not None:
    #         z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
    #     samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
    #     return samples

    def sample(self, y, num_points, truncate_std=None):
        """
        Args:
            num_points: Number of points to sample.
            y: A tensor of category indices, (B,).
        """
        batch_size = y.size(0)

        # Sample a standard normal latent vector z
        z = torch.randn(batch_size, self.args.latent_dim, device=y.device)
        
        # Get category embeddings and add to z
        category_embed = self.cat_embedding(y)  # (B, F)
        z_conditioned = z + category_embed

        if truncate_std is not None:
            z_conditioned = truncated_normal_(z_conditioned, mean=category_embed, std=1, trunc_std=truncate_std)

        samples = self.diffusion.sample(num_points, context=z_conditioned, flexibility=self.args.flexibility)
        return samples