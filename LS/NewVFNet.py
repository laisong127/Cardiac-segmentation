import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class VFN(nn.Module):
    def __init__(self):
        super(VFN, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.downsample2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.downsample3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(16, 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(2),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res0 = x
        x = self.downsample1(x)
        print(x.shape)
        res1 = x
        x = self.downsample2(x)
        print(x.shape)
        res2 = x
        x = self.downsample3(x)
        print(x.shape)
        x = self.upsample1(x)
        x = res2 + x
        x = self.upsample2(x)
        x = res1 + x
        x = self.upsample3(x)
        x = res0 + x
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 4, 64, 64, 64)
    Input = torch.randn(1, 2, 6, 128, 128)
    label = torch.randint(0, 4, (1, 10, 128, 128))
    model = VFN()
    out = model(Input)
    # print(model)
    print(out.shape)
    # print(label.shape)
    # out = torch.argmax(out, dim=1).float()
    # print(out)
