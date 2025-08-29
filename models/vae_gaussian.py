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

        self.cat_embedding =nn.Embedding(
            num_embeddings=args.num_classes,
            embedding_dim=args.latent_dim
        )
        #NOTE omnilearn cat embedding
        # self.cat_embedding = nn.Sequential(
        #     nn.Embedding(args.num_classes, args.latent_dim),
        #     MLP(
        #         args.latent_dim,
        #         int(args.mlp_ratio * args.latent_dim),
        #         act_layer=args.latent_dim,
        #         drop=0.1,
        #     ),
        # )
        

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
            y:  categories int labels (B,)
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

    def sample(self, y, num_points, flexibility, truncate_std=None):
        """
        Args:
            num_points: Number of points to sample.
            y: A tensor of category indices, (B,).
        """
        try:
            batch_size = y.size(0)
        except: 
            batch_size = 1

        # Sample a standard normal latent vector z
        z = torch.randn(batch_size, self.args.latent_dim, device=y.device)
        
        # Get category embeddings and add to z
        category_embed = self.cat_embedding(y)  # (B, F)
        z_conditioned = z + category_embed

        if truncate_std is not None:
            z_conditioned = truncated_normal_(z_conditioned, mean=category_embed, std=1, trunc_std=truncate_std)

        samples = self.diffusion.sample(num_points, context=z_conditioned, flexibility=flexibility)
        return samples
    
    """ 
    Understanding the kl_weight

    The kl_weight is a hyperparameter that balances two components of the variational autoencoder (VAE) loss function:

        loss_recons: The reconstruction loss, which measures how accurately the model can reconstruct the input point cloud from the latent space.

        loss_prior: The KL divergence loss, which measures how much the learned latent distribution deviates from a standard normal distribution. This term is a regularizer that prevents the encoder from collapsing into a single point, ensuring the latent space is well-structured and easy to sample from.

    The total loss is calculated as: loss = kl_weight * loss_prior + loss_recons.

        A high kl_weight forces the model to prioritize making the latent distribution Gaussian. This can lead to a phenomenon known as posterior collapse, where the model ignores the latent variable and the encoder learns to output a fixed distribution, as it is heavily penalized for any deviation.

        A low kl_weight prioritizes reconstruction, allowing the latent space to become more complex and potentially less Gaussian. While this might improve reconstruction quality, it makes the latent space harder to navigate and sample from, which is bad for generating new data.

    KL Annealing

    Instead of using a fixed kl_weight, a common and effective technique is KL annealing. This involves gradually increasing the value of kl_weight during training.

    A typical KL annealing schedule looks like this:

        Warm-up phase: Start with kl_weight = 0 for the initial epochs. This allows the model to first focus solely on learning a good reconstruction, establishing a stable representation.

        Gradual increase: Slowly ramp up kl_weight from 0 to a target value (e.g., 1.0) over a set number of epochs. This gently encourages the latent distribution to conform to the prior without collapsing.

        Constant phase: Keep kl_weight at a fixed value (e.g., 1.0) for the rest of the training.

    By using KL annealing, you get the best of both worlds: the model first learns to reconstruct well and then is regularized to maintain a structured latent space, avoiding posterior collapse and enabling high-quality generation.
    """