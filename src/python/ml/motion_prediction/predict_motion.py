
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import time

from ml.motion_prediction.backbone_torch import BackBone
from ml.motion_prediction.flow import FlowGrid


class MotionNN:
    def __init__(self, model_data_path, T=10, K=3, device='cuda') -> None:
        self.model = BackBone(K, T).float().to(device)
        self.model.load_state_dict(torch.load(model_data_path))
        self.device = device
        self.T = T
        self.K = K

    def predict_motion(self, lidar_data, scale, dt, filter=None):
        # move from image centric to channels first
        lidar_data = torch.from_numpy(np.moveaxis(lidar_data, -1, 0).astype(np.float32)).to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred_static_occ, pred_dyn_occ, pred_score, pred_motion = self.model(lidar_data)
            pred_static_occ = torch.sigmoid(pred_static_occ).cpu().detach().numpy()
            pred_dyn_occ = torch.sigmoid(pred_dyn_occ).cpu().detach().numpy()

            if filter is not None:
                pred_static_occ[pred_static_occ < filter] = 0
                pred_dyn_occ[pred_dyn_occ < filter] = 0

            # score data is ( 1, T, K, 1, H, W )
            pred_shape = pred_score.shape
            pred_score = pred_score.view(-1, self.T, self.K, 1, pred_shape[-2], pred_shape[-1])
            for t in range(self.T):
                pred_score[:, t, :, :, :, :] = nn.Softmax(dim=1)(pred_score[:, t, :, :, :, :])
            pred_score = pred_score.cpu().detach().numpy()

            # motion data is (1, T, K, 2, H, W )
            pred_shape = pred_motion.shape
            pred_motion = pred_motion.view(-1, self.T, self.K, 2, pred_shape[-2], pred_shape[-1])
            pred_motion = pred_motion.cpu().detach().numpy()

            flow_grid = FlowGrid(pred_dyn_occ[0, 0, :, :], pred_score[0, :, :, :, :, :], pred_motion[0, :, :, :, :, :], scale=scale)
            dynamic_flow = flow_grid.flow(mode='bilinear', dt=dt)

        return pred_static_occ, pred_dyn_occ, dynamic_flow
