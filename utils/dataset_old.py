import os
import random
from copy import copy
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm

my_dataset = ['02691156',  '02954340',  '03001627',  '03467517',  '03636649',  '03790512',  '03948459',  '04225987', '02773838',  '02958343',  '03261776', '03624134',  '03642806',  '03797390',  '04099429', '04379243'] 

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items() if k in my_dataset}

#train_split_data = json.load(open('/kaggle/input/d/jeremy26/shapenet-core/Shapenetcore_benchmark/train_split.json', 'r'))
class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        #if 'all' in cates:
        cates = [k for (k, v) in cate_to_synsetid.items()]
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None
        self.get_statistics()
        self.load()
    def get_statistics(self):
        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        # with h5py.File(self.path, 'r') as f:
        #     pointclouds = []
        #     for synsetid in self.cate_synsetids:
        #         for split in ('train', 'val', 'test'):
        #             pointclouds.append(torch.from_numpy(f[synsetid][split][...]))
        pointclouds = []
        #for synsetid in self.cate_synsetids:
        for split in ('train', 'val', 'test'):
            pc =  json.load(open(f"{self.path}/{split}_split.json", 'r'))
            for item in pc:
                file = os.path.join(self.path, item[2])
                point_cloud_array = np.load(file)
                pc_torch = torch.from_numpy(point_cloud_array)
                pointclouds.append(pc_torch)

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        #B, N, _ = all_points.size()
        N, _ = all_points.size()
        #mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        mean = all_points.view(N, -1).mean(dim=0) # (1, 3)
        #std = all_points.view(-1).std(dim=0)        # (1, )
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f): 
            # for j, pc in enumerate(f[synsetid][self.split]):
            #     yield torch.from_numpy(pc), j, cate_name
            for j, pc in enumerate(f):
                file_array = os.path.join(self.path, pc[2])
                pc_array = np.load(file_array)
                cate_id = pc[2][:8]
                cate_name = synsetid_to_cate[cate_id]
                yield torch.from_numpy(pc_array), j, cate_name, cate_id


        with open(os.path.join(self.path, f'{self.split}_split.json'), 'r') as f:
            self.split_data = json.load(f) 
            for pc, pc_id, cate_name, cate_id in _enumerate_pointclouds(self.split_data):
        # with h5py.File(self.path, mode='r') as f:
        #     for pc, pc_id, cate_name, cate_id in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

def collate_fn_pad_point_clouds(batch):
    """
    Pads point cloud tensors to the largest number of points in the batch.
    """
    # Find the largest number of points in the current batch
    max_num_points = max(p['pointcloud'].shape[0] for p in batch)
    
    # Pad all point clouds to max_num_points
    padded_batch = []
    for data in batch:
        point_cloud = data['pointcloud']
        num_points_to_pad = max_num_points - point_cloud.shape[0]
        # Pad with zeros
        padding = torch.zeros((num_points_to_pad, 3), dtype=point_cloud.dtype)
        padded_point_cloud = torch.cat([point_cloud, padding], dim=0)
        data['pointcloud'] = padded_point_cloud
        padded_batch.append(padded_point_cloud)
    # Stack the padded point clouds
    return torch.stack(padded_batch, dim=0)
    #return batch

# class ShapeNetDataset(Dataset):
#     def __init__(self, root_dir, split_type, scale_mode, transform=None):
#         self.root_dir = root_dir
#         self.split_type = split_type
#         with open(os.path.join(root_dir, f'{self.split_type}_split.json'), 'r') as f:
#             self.split_data = json.load(f)       
    
#     def __getitem__(self, index):
#         # read point cloud data
#         class_id, class_name, point_cloud_path = self.split_data[index]        
#         point_cloud_path = os.path.join(self.root_dir, point_cloud_path)
#         pc_data = np.load(point_cloud_path)
        
        
#         # return variable
#         data_dict= {}
#         data_dict['points'] = pc_data 
#         data_dict['num_points'] = pc_data.shape[0]
#         data_dict['class_id'] = class_id
#         data_dict['class_name'] = class_name
#         return data_dict        
    
#     @staticmethod
#     def collate_batch(batch_list, _unused=False):
#         ret = {}
#         ret['class_id'] = np.array([x['class_id'] for x in batch_list])
#         ret['class_name'] = np.array([x['class_name'] for x in batch_list])
#         ret['num_points'] = np.array([x['num_points'] for x in batch_list])
#         ret['num_voxels'] = np.array([x['num_voxels'] for x in batch_list])
#         ret['voxels'] = np.concatenate([x['voxels'] for x in batch_list], axis=0)
#         ret['voxel_num_points'] = np.concatenate([x['voxel_num_points'] for x in batch_list], axis=0)
        
#         for key in ['points', 'voxel_coords']:
#             val = [x[key] for x in batch_list]
#             coors = []
#             for i, coor in enumerate(val):
#                 coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
#                 coors.append(coor_pad)
#             ret[key] = np.concatenate(coors, axis=0)
#         ret['batch_size'] = len(batch_list)
#         return ret
                    
#     def __len__(self):
#         return len(self.split_data)