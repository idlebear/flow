import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

from flow import FlowGrid
from lidar_dataset import LidarDataSet, ToTensor


if __name__ == '__main__':
    path = 'data/lidar_static_dynamic'

    # add a transform to convert the data from image format to friendly channel first format
    transform = transforms.Compose([ToTensor()])
    ds = LidarDataSet(path, data_prefix='town4-lidar-', static_target='static-occ', dynamic_target='dynamic-occ', transform=transform)
    print('Length: {}'.format(len(ds)))
    print('Shape: {}'.format(ds[2]['data'].shape))

    batch_size = 4

    ds_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    for idx, sample in enumerate(ds_dataloader):
        print("Shape of data [N, C, H, W]: ", sample['data'].shape)
        print("Shape of static occupancy label: ", sample['static'].shape, sample['static'].dtype)
        print("Shape of dynamic occupancy label: ", sample['dynamic'].shape, sample['dynamic'].dtype)
        print("Shape of motion label: ", sample['motion'].shape, sample['motion'].dtype)
        break

    plt.ion()

    T = 10

    for batch, sample in enumerate(ds_dataloader):

        X = sample['data']
        y_static_occ = sample['static']
        y_dynamic_occ = sample['dynamic']
        y_motion = sample['motion']

        # move everything to numpy for display
        y_dynamic_occ = y_dynamic_occ.cpu().detach().numpy()
        y_static_occ = y_static_occ.cpu().detach().numpy()
        y_motion = y_motion.cpu().detach().numpy()

        lidar_data = X.cpu().detach().numpy()

        for i in range(batch_size):

            lidar_img = lidar_data[i, 0, :, :]
            for j in range(25):
                for k in range(10):
                    lidar_img = np.logical_or(lidar_img, lidar_data[i, j*10 + k, :, :])

            gt_flow_grid = FlowGrid(y_dynamic_occ[i, 0, :, :], None, np.expand_dims(y_motion[i, :, :, :, :], axis=1), scale=0.4)
            gt_flow = gt_flow_grid.flow(mode='bilinear', dt=0.5)

            fig, ax = plt.subplots(5, T+1, num=1, figsize=(25, 15))

            row_labels = ['LIDAR (->T=0)', 'GT STATIC', 'X motion', 'Y motion', 'Pred X motion', 'Pred Y motion', 'GT Flow', 'Pred Flow']
            for _ax, label in zip(ax[:, 0], row_labels[:]):
                _ax.set_ylabel(label, size='x-large')

            col_labels = ['T = 0', 'T = 0.5', 'T = 1.0', 'T = 1.5', 'T = 2.0', 'T = 2.5', 'T = 3.0', 'T = 3.5', 'T = 4.0', 'T = 4.5']  # , 'T = 5.0', ]
            for _ax, label in zip(ax[0, :], col_labels):
                _ax.set_title(label, size='x-large')

            ax[0, 0].imshow(lidar_img)

            ax[4, 0].imshow(y_dynamic_occ[i, 0, :, :])

            for j in range(T):

                gt_xmo_img = np.squeeze(y_motion[i, j, 0, :, :])
                gt_ymo_img = np.squeeze(y_motion[i, j, 1, :, :])

                ax[1, j].imshow(y_dynamic_occ[i, j, :, :])
                ax[2, j].imshow(gt_xmo_img)
                ax[3, j].imshow(gt_ymo_img)
                ax[4, j+1].imshow(gt_flow[j, :, :])

            for k in range(0, 5):
                for j in range(T+1):
                    ax[k, j].set_xticklabels([])
                    ax[k, j].set_yticklabels([])

            # plt.draw()
            # plt.pause(0.1)
            plt.show(block=True)
