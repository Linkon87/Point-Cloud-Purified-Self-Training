#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset




def load_data(test_data,test_label):

    all_data = np.load(test_data)
    all_label = np.load(test_label)

    print('all_data: ',test_data)
    print('all_label: ', test_label)

    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40_difussion_source(Dataset):
    def __init__(self, test_data,test_label,  partition='train'):
        self.test_data = test_data
        self.test_label = test_label
        self.data, self.label = load_data(self.test_data,self.test_label)
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return [pointcloud, item], label.item()

    def __len__(self):
        return self.data.shape[0]


