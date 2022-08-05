
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import block
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import time

from backbone_torch import BackBone
from flow import FlowGrid
from lidar_dataset import LidarDataSet, ToTensor

if __name__ == '__main__':
    path = 'data/lidar_static_dynamic'

    # setting device on GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # instantiate the model
    model = BackBone(3, 10).float().to(device)

    # add a transform to convert the data from image format to friendly channel first format
    transform = transforms.Compose([ToTensor()])
    ds = LidarDataSet(path, data_prefix='town4-lidar-', static_target='static-occ', dynamic_target='dynamic-occ', transform=transform)
    print('Length: {}'.format(len(ds)))
    print('Shape: {}'.format(ds[2]['data'].shape))

    test_len = len(ds) // 10                    # 10% test data
    train_len = len(ds) - test_len              # and the remainder for training

    # hyperparameters
    learning_rate = 1e-3
    batch_size = 4
    epochs = 15

    datasets = torch.utils.data.random_split(ds, (train_len, test_len), generator=torch.Generator().manual_seed(5))
    train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False)

    ds_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    lidar_model_file = './full_occupation_model'
    if isfile(lidar_model_file):
        model.load_state_dict(torch.load(lidar_model_file))

    plt.ion()

    T = 10
    K = 3

    for batch, sample in enumerate(test_dataloader):

        T = model.T
        K = model.K

        X = sample['data'].to(device)
        y_static_occ = sample['static'].to(device)
        y_dynamic_occ = sample['dynamic'].to(device)
        y_motion = sample['motion'].to(device)

        pred_static_occ, pred_dyn_occ, pred_score, pred_motion = model(X)

        # push the occupancy probs through a sigmoid first to make sure they're in the right range
        pred_static_occ = torch.sigmoid(pred_static_occ)
        pred_dyn_occ = torch.sigmoid(pred_dyn_occ)

        # move everything to numpy for display
        pred_static_occ = pred_static_occ.cpu().detach().numpy()
        pred_dyn_occ = pred_dyn_occ.cpu().detach().numpy()
        y_dynamic_occ = y_dynamic_occ.cpu().detach().numpy()
        y_static_occ = y_static_occ.cpu().detach().numpy()
        y_motion = y_motion.cpu().detach().numpy()

        # score data is ( N, T, K, 1, H, W )
        pred_shape = pred_score.shape
        pred_score = pred_score.view(-1, T, K, 1, pred_shape[-2], pred_shape[-1])
        for t in range(T):
            pred_score[:, t, :, :, :, :] = nn.Softmax(dim=1)(pred_score[:, t, :, :, :, :])
        pred_score = pred_score.cpu().detach().numpy()

        # motion data is (N, T, K, 2, H, W )
        pred_shape = pred_motion.shape
        pred_motion = pred_motion.view(-1, T, K, 2, pred_shape[-2], pred_shape[-1])
        pred_motion = pred_motion.cpu().detach().numpy()

        for i in range(batch_size):
            lidar_data = sample['data'].cpu().detach().numpy()
            lidar_img = lidar_data[i, 0, :, :]
            for j in range(25):
                for k in range(10):
                    lidar_img = np.logical_or(lidar_img, lidar_data[i, j*10 + k, :, :])

            # proportion the vectors by their score contribution
            pred_xmo = np.zeros((10, 200, 200))
            pred_ymo = np.zeros((10, 200, 200))
            for t in range(model.T):
                for k in range(model.K):
                    pred_xmo[t, :, :] += pred_score[i, t, k, 0, :, :] * pred_motion[i, t, k, 0, :, :]
                    pred_ymo[t, :, :] += pred_score[i, t, k, 0, :, :] * pred_motion[i, t, k, 1, :, :]

            gt_flow_grid = FlowGrid(y_dynamic_occ[i, 0, :, :], None, np.expand_dims(y_motion[i, :, :, :, :], axis=1), scale=0.4)
            filtered_pred_dyn_occ = np.array(pred_dyn_occ[i, 0, :, :])
            filtered_pred_dyn_occ[filtered_pred_dyn_occ < 0.1] = 0  # filter out some of the noise

            flow_grid = FlowGrid(filtered_pred_dyn_occ, pred_score[i, :, :, :, :, :], pred_motion[i, :, :, :, :, :], scale=0.4)
            gt_flow = gt_flow_grid.flow(mode='bilinear', dt=0.5)
            flow = flow_grid.flow(mode='bilinear', dt=0.5)

            fig, ax = plt.subplots(8, T+1, num=1, figsize=(25, 15))

            row_labels = ['LIDAR (->T=0)', 'GT STATIC', 'X motion', 'Y motion', 'Pred X motion', 'Pred Y motion', 'GT Flow', 'Pred Flow']
            for _ax, label in zip(ax[:, 0], row_labels[:]):
                _ax.set_ylabel(label, size='x-large')

            col_labels = ['T = 0', 'T = 0.5', 'T = 1.0', 'T = 1.5', 'T = 2.0', 'T = 2.5', 'T = 3.0', 'T = 3.5', 'T = 4.0', 'T = 4.5']  # , 'T = 5.0', ]
            for _ax, label in zip(ax[0, :], col_labels):
                _ax.set_title(label, size='x-large')

            ax[0, 0].imshow(lidar_img)
            ax[0, 1].imshow(pred_static_occ[i, 0, :, :])

            filtered_occ = np.array(pred_static_occ[i, 0, :, :])
            filtered_occ[filtered_occ < 0.1] = 0
            ax[0, 2].imshow(filtered_occ)
            filtered_occ[filtered_occ < 0.2] = 0
            ax[0, 3].imshow(filtered_occ)
            filtered_occ[filtered_occ < 0.3] = 0
            ax[0, 4].imshow(filtered_occ)

            ax[6, 0].imshow(y_dynamic_occ[i, 0, :, :])
            ax[7, 0].imshow(pred_dyn_occ[i, 0, :, :])

            gt_img = np.squeeze(y_static_occ[i, 0:, :])
            ax[1, 0].imshow(gt_img)

            for j in range(T):

                gt_xmo_img = np.squeeze(y_motion[i, j, 0, :, :])
                gt_ymo_img = np.squeeze(y_motion[i, j, 1, :, :])

                pred_xmo_img = np.squeeze(pred_xmo[j, :, :])
                pred_ymo_img = np.squeeze(pred_ymo[j, :, :])

                ax[2, j].imshow(gt_xmo_img)
                ax[3, j].imshow(gt_ymo_img)
                ax[4, j].imshow(pred_xmo_img)
                ax[5, j].imshow(pred_ymo_img)
                ax[6, j+1].imshow(gt_flow[j, :, :])
                ax[7, j+1].imshow(flow[j, :, :])

            for k in range(0, 7):
                for j in range(T+1):
                    ax[k, j].set_xticklabels([])
                    ax[k, j].set_yticklabels([])

            # plt.draw()
            # plt.pause(0.1)
            plt.show(block=True)
