import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_airplane.pt')
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_0.000000_1000.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
#parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--dataset_path', type=str, default='/pscratch/sd/c/ccardona/datasets/shapenetCore/')
parser.add_argument('--batch_size', type=int, default=128)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
args = parser.parse_args()

def pad_tensors_to_max_size(tensor_list: list) -> list:
    """
    Pads a list of tensors to the same size based on the largest tensor in the list.
    
    The padding is applied along dimension 1 with zeros.
    
    Args:
        tensor_list: A list of tensors with shape (1, N, 3), where N can vary.
        
    Returns:
        A new list of tensors where all tensors have the same size.
    """
    # 1. Find the maximum size along the second dimension (N)
    max_size = 0
    for tensor in tensor_list:
        if tensor.size(1) > max_size:
            max_size = tensor.size(1)

    # 2. Pad each tensor to the max_size
    padded_tensors = []
    for tensor in tensor_list:
        current_size = tensor.size(1)
        padding_size = max_size - current_size
        
        # If padding is needed, create a zero tensor and concatenate
        if padding_size > 0:
            # Create a tensor of zeros with shape (1, padding_size, 3)
            padding = torch.zeros(1, padding_size, 3, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the original tensor with the padding tensor along dimension 1
            padded_tensor = torch.cat([tensor, padding], dim=1)
            padded_tensors.append(padded_tensor)
        else:
            # No padding needed, just add the original tensor
            padded_tensors.append(tensor)
            
    return padded_tensors


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_batch_3d(batch_of_point_clouds: torch.Tensor):
    """
    Plots each individual point cloud from a batch in a separate 3D scatter plot.

    Args:
        batch_of_point_clouds: A PyTorch tensor of shape (B, N, 3), where:
            - B is the batch size (e.g., 128)
            - N is the number of points (e.g., 2048)
            - 3 represents the (x, y, z) coordinates
    """
    # Get the batch size
    batch_size = batch_of_point_clouds.shape[0]

    # Loop through each point cloud in the batch
    #for i in range(batch_size):
    for i in range(3):
        # Extract the current point cloud tensor
        # .detach() is used to remove it from the computation graph.
        # .cpu() ensures the tensor is on the CPU.
        # .numpy() converts the tensor to a NumPy array, which matplotlib requires.
        point_cloud = batch_of_point_clouds[i].detach().cpu().numpy()

        # Separate the coordinates for plotting
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        # Create a new figure and a 3D subplot for the current point cloud
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        ax.scatter(x, y, z, s=1)  # s is the marker size

        # Set axis labels and a title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud {i+1} of {batch_size}')
        
        # Display the plot
        plt.savefig(f"results/gen_{i}.png")


# --- Example Usage ---
# Create a dummy tensor that matches your batch size and shape
dummy_batch = torch.randn(128, 2048, 3)

# Call the function to plot each individual point cloud
plot_batch_3d(dummy_batch)

# Logging
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt, weights_only=False)#, map_location='cpu')
seed_all(args.seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=args.normalize,
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])

# Reference Point Clouds
ref_pcs = []
for i, data in enumerate(test_dset):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
ref_pcs_padded = pad_tensors_to_max_size(ref_pcs)
ref_pcs = torch.cat(ref_pcs_padded, dim=0)

# Generate Point Clouds
gen_pcs = []
for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
plot_batch_3d(gen_pcs)
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}def plot_batch_3d(batch_of_poindef plot_batch_3d(batch_of_point_clouds: torch.Tensor):t_clouds: torch.Tensor):
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
