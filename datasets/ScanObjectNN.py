import os
import sys
import glob
import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"




class ScanObjectNN(Dataset):
    def __init__(self, num_points=2048, partition='training'):
        self.partition = partition
        if partition=='train':
            self.partition = 'training'
        self.data, self.label = load_scanobjectnn_data(self.partition)
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            # pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return  [pointcloud, item], label.item()
    def __len__(self):
        return self.data.shape[0]

def load_scanobjectnn_data(partition):
    # download()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.abspath(os.path.join(current_directory, '..'))
    sys.path.append(project_path)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    # h5_name = project_path + '/scanobjectnn/main_split_nobg/' + partition + '_objectdataset.h5'
    h5_name = project_path + '/pointdp/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud