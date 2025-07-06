import numpy as np
from torch.utils.data import Dataset
from rand_augs import RandAug


def load_data(test_data,test_label):

    try:
        npz = np.load(test_data)
        all_data = npz['test_pc'][...,:3]  # [..., K, 3]
        all_label = npz['test_label']
        print('all_data: ',test_data)
    except:
        all_data = np.load(test_data)
        all_label = np.load(test_label)
        print('all_data: ',test_data)
        print('all_label: ', test_label)



    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def pointcloud_strong_aug(pointcloud):
    pass


class ModelNet40C(Dataset):
    def __init__(self, split, test_data,test_label,data_augment=False,data_augment_strong=False):
        assert split == 'test'
        self.split = split
        self.test_data = test_data
        self.test_label = test_label
        self.data, self.label = load_data(self.test_data,self.test_label)
        self.partition = 'test'
        self.data_augment = data_augment
        self.data_augment_strong = data_augment_strong

    def __getitem__(self, item):
        pointcloud = self.data[item]  # [:self.num_points]
        pointcloud2 = self.data[item]
        label = self.label[item]
        if self.data_augment:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        if self.data_augment_strong:
            # pointcloud2 = pointcloud_strong_aug(pointcloud2)
            #
            # randaug = RandAugment(1,2)
            # pointcloud2 = randaug(pointcloud2)
            randaug_new = RandAug(n=5)
            pointcloud2 = randaug_new(pointcloud2)
            del randaug_new
            return [pointcloud, pointcloud2, item], label.item()

        return [pointcloud, item], label.item()

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    test = ModelNet40C('test')
    print(len(test))
    print(test.__getitem__(0)['pc'].shape)