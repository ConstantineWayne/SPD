# -*- coding: utf-8 -*-
"""
EventDataset classes
"""



import os
import random
import glob
import numpy as np
import torch
from os.path import join
from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(self, base_folder, event_folder, start_time=0, stop_time=0,
                 transform=None, normalize=True,
                 noise=None, severity=None):

        self.base_folder = base_folder
        self.event_folder = join(self.base_folder, event_folder)
        self.transform = transform
        self.start_time = start_time
        self.stop_time = stop_time
        self.normalize = normalize

        # ===== noise 控制 =====
        self.noise = noise
        self.severity = severity

        self.use_mvsec = True
        self.read_timestamps()
        self.parse_event_folder()

    def read_timestamps(self):
        raw_stamps = np.loadtxt(join(self.event_folder, 'timestamps.txt'))
        if raw_stamps.size == 0:
            raise IOError('Dataset is empty')

        if len(raw_stamps.shape) == 1:
            raw_stamps = raw_stamps.reshape((1, 2))

        self.stamps = raw_stamps[:, 1]
        assert(np.alltrue(np.diff(self.stamps) > 0)), "timestamps not increasing"

        self.initial_stamp = self.stamps[0]
        self.stamps = self.stamps - self.initial_stamp

        self.first_valid_idx = 0
        self.last_valid_idx = len(self.stamps) - 1

        self.length = self.last_valid_idx - self.first_valid_idx + 1

    def parse_event_folder(self):
        raise NotImplementedError

    def __len__(self):
        return self.length

    def num_channels(self):
        raise NotImplementedError

    def get_last_stamp(self):
        return self.stamps[self.last_valid_idx]

    def get_index_at(self, i):
        return self.first_valid_idx + i

    def get_stamp_at(self, i):
        return self.stamps[self.get_index_at(i)]

    def __getitem__(self, i):
        raise NotImplementedError



class VoxelGridDataset(EventDataset):

    def parse_event_folder(self):
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def add_event_noise(self, event_tensor):
        if self.noise not in ['noise'] or self.noise is None:
            return event_tensor

        event_tensor = event_tensor.copy().astype(np.float32)
        C, H, W = event_tensor.shape



        noise_ratios = [0.05, 0.15, 0.25, 0.5, 0.8]

        noise_ratio = noise_ratios[self.severity - 1]


        if self.noise in ['noise']:
            num_noise = int(noise_ratio * C * H * W)

            c = np.random.randint(0, C, num_noise)
            y = np.random.randint(0, H, num_noise)
            x = np.random.randint(0, W, num_noise)


            noise_polarities = np.random.choice([-1.0, 1.0], num_noise)


            event_tensor[c, y, x] = noise_polarities


        event_tensor = np.clip(event_tensor, -1.0, 1.0)

        return event_tensor

    def __getitem__(self, i, transform_seed=None):
        assert 0 <= i < self.length

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        if self.use_mvsec:
            path = join(self.event_folder,
                        'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i))
            event_tensor = np.load(path)
        else:
            path_event = glob.glob(self.event_folder + '/*_{:04d}_voxel.npy'.format(self.first_valid_idx + i))
            event_tensor = np.load(path_event[0])

        if self.normalize:
            mask = np.nonzero(event_tensor)
            if mask[0].size > 0:
                mean = event_tensor[mask].mean()
                std = event_tensor[mask].std()
                if std > 0:
                    event_tensor[mask] = (event_tensor[mask] - mean) / std

        event_tensor = self.add_event_noise(event_tensor)


        self.num_bins = event_tensor.shape[0]
        events = torch.from_numpy(event_tensor).float()

        if self.transform:
            random.seed(transform_seed)
            events = self.transform(events)

        return {'events': events}


