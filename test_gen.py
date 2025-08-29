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


cats = ['Airplane', 'Bag', 'Basket', 'Bathtub', 'Bed', 'Bench', 'Bottle', 'Bowl', 'Bus', 
        'Cabinet', 'Can', 'Camera', 'Cap', 'Car', 'Chair', 'Clock', 'Dishwasher', 'Monitor', 
        'Table', 'Telephone', 'Tin_can', 'Tower', 'Train', 'Keyboard', 'Earphone', 'Faucet', 
        'File', 'Guitar', 'Helmet', 'Jar', 'Knife', 'Lamp', 'Laptop', 'Speaker', 'Mailbox', 
        'Microphone', 'Microwave', 'Motorcycle', 'Mug', 'Piano', 'Pillow', 'Pistol', 'Pot', 
        'Printer', 'Remote_control', 'Rifle', 'Rocket', 'Skateboard', 'Sofa', 'Stove',
        'Vessel', 'Washer', 'Cellphone', 'Birdhouse', 'Bookshelf']

int__to_classes = {
    0: 'Airplane',
    1: 'Bag',
    2: 'Basket',
    3: 'Bathtub',
    4: 'Bed',
    5: 'Bench',
    6: 'Bottle',
    7: 'Bowl',
    8: 'Bus',
    9: 'Cabinet',
    10: 'Can',
    11: 'Camera',
    12: 'Cap',
    13: 'Car',
    14: 'Chair',
    15: 'Clock',
    16: 'Dishwasher',
    17: 'Monitor',
    18: 'Table',
    19: 'Telephone',
    20: 'Tin_can',
    21: 'Tower',
    22: 'Train',
    23: 'Keyboard',
    24: 'Earphone',
    25: 'Faucet',
    26: 'File',
    27: 'Guitar',
    28: 'Helmet',
    29: 'Jar',
    30: 'Knife',
    31: 'Lamp',
    32: 'Laptop',
    33: 'Speaker',
    34: 'Mailbox',
    35: 'Microphone',
    36: 'Microwave',
    37: 'Motorcycle',
    38: 'Mug',
    39: 'Piano',
    40: 'Pillow',
    41: 'Pistol',
    42: 'Pot',
    43: 'Printer',
    44: 'Remote_control',
    45: 'Rifle',
    46: 'Rocket',
    47: 'Skateboard',
    48: 'Sofa',
    49: 'Stove',
    50: 'Vessel',
    51: 'Washer',
    52: 'Cellphone',
    53: 'Birdhouse',
    54: 'Bookshelf'
}
# Arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_airplane.pt')
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_0.000000_4000_55cate.pt')
parser.add_argument('--categories', type=str_list, default=cats)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--flexibility', type=float, default=0.0)
# Datasets and loaders
#parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--dataset_path', type=str, default='/pscratch/sd/c/ccardona/datasets/shapenetCore/')
parser.add_argument('--batch_size', type=int, default=128)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=1000)
parser.add_argument('--normalize', type=str, default='shape_unit', choices=[None, 'shape_unit', 'shape_bbox'])
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

def plot_batch_3d(batch_of_point_clouds: torch.Tensor, cates):
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
    for i in range(batch_size):
    #for i in range(num_samples):
        # Extract the current point cloud tensor
        # .detach() is used to remove it from the computation graph.
        # .cpu() ensures the tensor is on the CPU.
        # .numpy() converts the tensor to a NumPy array, which matplotlib requires.
        point_cloud = batch_of_point_clouds[i].detach().cpu().numpy()
        category = cates[i]
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
        ax.set_title(f'Point Cloud {i+1} of {int__to_classes[category]}')
        
        # Display the plot
        plt.savefig(f"results/gen_cate:{int__to_classes[category]}.png")
        plt.close()



# Logging
#save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
save_dir = os.path.join(args.save_dir, 'GEN_Ours_%s_%d' % ('_'.join("55_cats"), int(time.time())) )
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

# Mode
logger.info('Loading model...')
if ckpt['args'].model == 'gaussian':
    model = GaussianVAE(ckpt['args']).to(args.device)
elif ckpt['args'].model == 'flow':
    model = FlowVAE(ckpt['args']).to(args.device)
logger.info(repr(model))
# if ckpt['args'].spectral_norm:
#     add_spectral_norm(model, logger=logger)
model.load_state_dict(ckpt['state_dict'])
#test_size = len(test_dset)
test_size = 100
# Reference Point Clouds
ref_pcs = []
ref_cats = []
for i, data in enumerate(test_dset):
    if i >= test_size:
            break
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_cats.append(data['cate'])
ref_pcs = torch.cat(ref_pcs, dim=0)

# Generate Point Clouds
num_class = len(args.categories)
gen_pcs = []
# for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
#     with torch.no_grad():
#         #z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
#         y = torch.randint(0,num_class,(args.val_batch_size,)).to(args.device)
#         x = model.sample(y, args.sample_num_points, flexibility=args.flexibility)
#         gen_pcs.append((x.detach().cpu(),y))
for y in tqdm(ref_cats, 'Generate'):
    with torch.no_grad():
        y = torch.tensor(y, dtype=torch.int32).to(args.device)
        x = model.sample(y, args.sample_num_points, flexibility=args.flexibility)
        gen_pcs.append(x.detach().cpu())
gen_pcs = torch.cat(gen_pcs, dim=0)

plot_batch_3d(gen_pcs, ref_cats)
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

# Save
logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'out.npy'), gen_pcs.numpy())

# Compute metrics
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}
    jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

for k, v in results.items():
    logger.info('%s: %.12f' % (k, v))
