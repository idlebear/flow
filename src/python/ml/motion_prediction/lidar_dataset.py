# PyTorch DataLoader for Autonomous vehicle motion prediction

import torch

from torch.utils.data import Dataset

import numpy as np
from os import listdir
from os.path import join
import re


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, static, dynamic, motion = sample['data'], sample['static'], sample['dynamic'], sample['motion']

        # move from image centric to channels first
        data = torch.from_numpy(np.moveaxis(data, -1, 0).astype(np.float32))

        # target data is already in the right order, and only needs conversion to a tensor
        if static is not None:
            static = torch.from_numpy(static.astype(np.float32)).unsqueeze(0)

        if dynamic is not None:
            dynamic = dynamic.astype(np.float32)
            motion = motion.astype(np.float32)

        return {'data': data,
                'static': static,
                'dynamic': dynamic,
                'motion': motion}


class LidarDataSet(Dataset):
    def __init__(self, data_dir, data_prefix='lidar-', static_target=None, dynamic_target=None, transform=None,
                 target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        file_list = listdir(data_dir)
        self.data_files = [f for f in file_list if f.startswith(data_prefix)]
        self.data_files.sort()
        self.static_target = static_target
        self.dynamic_target = dynamic_target

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        lidar_path = join(self.data_dir, self.data_files[index])
        lidar_data = np.load(lidar_path)
        parts = re.split('\W', self.data_files[index])
        if len(parts) < 4:
            raise ValueError('Format of input datafile name is incorrect -- should be xxxxx-lidar-yyyyy.npy')

        if self.dynamic_target is not None:
            target_name = parts[0]+'-'+self.dynamic_target+'-'+parts[2]+'.npy'
            target_path = join(self.data_dir, target_name)
            target_data = np.load(target_path)
            dynamic = target_data[:, 0, :, :].astype(np.float32)
            motion = target_data[:, 1:, :, :].astype(np.float32)
        else:
            dynamic = None
            motion = None

        if self.static_target is not None:
            target_name = parts[0]+'-'+self.static_target+'-'+parts[2]+'.npy'
            target_path = join(self.data_dir, target_name)
            target_data = np.load(target_path)
            static = target_data.astype(np.float32)
        else:
            static = None

        sample = {'data': lidar_data, 'static': static, 'dynamic': dynamic, 'motion': motion}
        if self.transform:
            sample = self.transform(sample)

        return sample
