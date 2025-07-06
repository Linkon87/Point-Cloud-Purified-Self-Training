
import os
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from rand_augs import RandAug





def load_data(test_data,test_label):

    all_data = np.load(test_data)
    all_label = np.load(test_label)

    print('all_data: ',test_data)
    print('all_label: ', test_label)

    return all_data, all_label



class ScanObjectNN_test(Dataset):
    def __init__(self, test_data,test_label,data_augment=False,data_augment_strong=False):
     self.test_data = test_data
     self.test_label = test_label
     self.data, self.label = load_data(self.test_data,self.test_label)
     self.data_augment = data_augment
     self.data_augment_strong = data_augment_strong

    def __getitem__(self, item):
        pointcloud = self.data[item]
        pointcloud2 = self.data[item]
        label = self.label[item]
        if self.data_augment:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        if self.data_augment_strong:
            randaug_new = RandAug(n=5)
            pointcloud2 = randaug_new(pointcloud2)
            del randaug_new

            return [pointcloud, pointcloud2, item], label.item()

        return  [pointcloud, item], label.item()

    def __len__(self):
        return self.data.shape[0]

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud
