import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.normalization import GroupNorm
import torchvision
import matplotlib.pyplot as plt
import numpy as np

GROUP_NORMALIZATION_SIZE = 32
KERNAL_SIZE = 3
STD_PADDING = int(KERNAL_SIZE // 2)
DILATED_PADDING = int(2 * STD_PADDING)

INPUT_CHANNELS = 250

C1_CHANNELS = 32
C1_GROUPS = int(C1_CHANNELS // GROUP_NORMALIZATION_SIZE)
C2_CHANNELS = 64
C2_GROUPS = int(C2_CHANNELS // GROUP_NORMALIZATION_SIZE)
C4_CHANNELS = 128
C4_GROUPS = int(C4_CHANNELS // GROUP_NORMALIZATION_SIZE)
C8_CHANNELS = 256
C8_GROUPS = int(C8_CHANNELS // GROUP_NORMALIZATION_SIZE)

PREDICTION_CHANNELS = 128
PREDICTION_GROUPS = int(PREDICTION_CHANNELS // GROUP_NORMALIZATION_SIZE)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, group_norm=None) -> None:
        super().__init__()

        if group_norm is None:
            use_bias = True
            self.norm = None
        else:
            use_bias = False
            self.norm = GroupNorm(*group_norm)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=use_bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.norm is None:
            return self.relu(self.conv(x))
        else:
            return self.relu(self.norm(self.conv(x)))


class BackBone(nn.Module):
    def __init__(self, K, T):
        super().__init__()

        self.K = K
        self.T = T

        # Layer 1 -- image size must be evenly divisible by 8
        # Image input shape ( ?, 250, 480, 480)
        self.C1 = nn.Sequential(
            ConvBlock(in_channels=INPUT_CHANNELS, out_channels=C1_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C1_GROUPS, C1_CHANNELS)),
            ConvBlock(in_channels=C1_CHANNELS, out_channels=C1_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C1_GROUPS, C1_CHANNELS))
        )

        self.C2 = nn.Sequential(
            ConvBlock(in_channels=C1_CHANNELS, out_channels=C2_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C2_GROUPS, C2_CHANNELS)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(in_channels=C2_CHANNELS, out_channels=C2_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C2_GROUPS, C2_CHANNELS)),
        )

        self.C4 = nn.Sequential(
            ConvBlock(in_channels=C2_CHANNELS, out_channels=C4_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C4_GROUPS, C4_CHANNELS)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(in_channels=C4_CHANNELS, out_channels=C4_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C4_GROUPS, C4_CHANNELS)),
            ConvBlock(in_channels=C4_CHANNELS, out_channels=C4_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C4_GROUPS, C4_CHANNELS)),
        )

        self.C8 = nn.Sequential(
            ConvBlock(in_channels=C4_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.average_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.C = nn.Sequential(
            ConvBlock(in_channels=int(C1_CHANNELS+C2_CHANNELS+C4_CHANNELS+C8_CHANNELS), out_channels=C8_CHANNELS,
                      kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
            ConvBlock(in_channels=C8_CHANNELS, out_channels=C8_CHANNELS, kernel_size=KERNAL_SIZE, padding=STD_PADDING, group_norm=(C8_GROUPS, C8_CHANNELS)),
        )

        self.pred_layer = nn.Sequential(
            ConvBlock(in_channels=C2_CHANNELS, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
            ConvBlock(in_channels=PREDICTION_CHANNELS, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
        )

        self.dilation_layer = nn.Sequential(
            ConvBlock(in_channels=C8_CHANNELS, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=DILATED_PADDING, dilation=2, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
            ConvBlock(in_channels=PREDICTION_CHANNELS, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=DILATED_PADDING, dilation=2, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.static_occ_layer = nn.Sequential(
            ConvBlock(in_channels=PREDICTION_CHANNELS * 2, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
        )

        self.static_occ_score = nn.Sequential(
            nn.Conv2d(in_channels=PREDICTION_CHANNELS, out_channels=1, kernel_size=KERNAL_SIZE, padding=STD_PADDING),
        )

        self.dyn_occ_layer = nn.Sequential(
            ConvBlock(in_channels=PREDICTION_CHANNELS * 2, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
        )

        self.dyn_occ_score = nn.Sequential(
            nn.Conv2d(in_channels=PREDICTION_CHANNELS, out_channels=1, kernel_size=KERNAL_SIZE, padding=STD_PADDING),
        )

        self.motion_layer = nn.Sequential(
            ConvBlock(in_channels=PREDICTION_CHANNELS * 2, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
            ConvBlock(in_channels=PREDICTION_CHANNELS, out_channels=PREDICTION_CHANNELS, kernel_size=KERNAL_SIZE,
                      padding=STD_PADDING, group_norm=(PREDICTION_GROUPS, PREDICTION_CHANNELS)),
        )

        self.motion_score = nn.Sequential(
            nn.Conv2d(in_channels=PREDICTION_CHANNELS, out_channels=K*T, kernel_size=KERNAL_SIZE, padding=STD_PADDING),
        )

        self.motion_vectors = nn.Sequential(
            nn.Conv2d(in_channels=PREDICTION_CHANNELS, out_channels=2*K*T, kernel_size=KERNAL_SIZE, padding=STD_PADDING),
        )

    def forward(self, x):
        C1 = self.C1(x)
        C2 = self.C2(C1)
        C4 = self.C4(C2)
        C8 = self.C8(C4)

        # tap the backbone and combine the results to construct C
        C = torch.cat((self.average_pool(C1), C2), dim=1)
        C = torch.cat((self.average_pool(C), C4, C8), dim=1)
        C = self.C(C)

        static_pred = self.pred_layer(C2)
        static_dilated = self.dilation_layer(C)
        static_pred = torch.cat((static_pred, static_dilated), dim=1)
        static_occ = self.static_occ_layer(static_pred)
        static_occ = self.static_occ_score(static_occ)

        dyn_pred = self.pred_layer(C2)
        dyn_dilated = self.dilation_layer(C)
        dyn_pred = torch.cat((dyn_pred, dyn_dilated), dim=1)
        dyn_occ = self.dyn_occ_layer(dyn_pred)
        dyn_occ = self.dyn_occ_score(dyn_occ)

        motion = self.motion_layer(dyn_pred)
        motion_score = self.motion_score(motion)
        motion_vectors = self.motion_vectors(motion)

        return static_occ, dyn_occ, motion_score, motion_vectors
