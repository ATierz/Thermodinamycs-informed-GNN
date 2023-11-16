"""dataLoader.py"""

import os
import numpy as np
import itertools
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class GraphDataset(Dataset):
    def __init__(self, datasetInfo, dset_dir, len=0, short=True):
        'Initialization'
        self.short = short
        self.dset_dir = dset_dir
        self.datasetInfo = datasetInfo
        self.dims = {'z': 8, 'q': 2, 'q_0': 0, 'n': 1, 'f': 1, 'g': 0}
        self.samplingFactor = datasetInfo['samplingFactor']
        self.dt = datasetInfo['dt'] * self.samplingFactor
        self.data = []
        # self.data = torch.load(os.path.join(self.dset_dir))
        self.data = torch.load(dset_dir)
        if len != 0:
            self.data = self.data[:len]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)

    def get_stats(self):
        total_tensor = None
        # sample = random.sample(self.data, k=round(len(self.data) * 0.3))
        sample = self.data
        for data in sample:
            if total_tensor is not None:
                total_tensor = torch.cat((total_tensor, data.x), dim=0)
            else:
                total_tensor = data.x

        scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler = StandardScaler()

        scaler.fit(total_tensor)
        # apply transform
        # standardized = scaler.transform(total_tensor)

        return scaler


if __name__ == '__main__':
    pass
