import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


from pc_utils import (rotate_point_cloud, PointcloudScaleAndTranslate)
import rs_cnn.data.data_utils as rscnn_d_utils
from rs_cnn.data.ModelNet40Loader import ModelNet40Cls as rscnn_ModelNet40Cls
import PCT_Pytorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils
from pointnet2_tf.modelnet_h5_dataset import ModelNetH5Dataset as pointnet2_ModelNetH5Dataset
from dgcnn.pytorch.data import ModelNet40 as dgcnn_ModelNet40
from dgcnn.pytorch.data import load_data as dgcnn_load_data
from dgcnn.pytorch.data import load_si_adv_data


# distilled from the following sources:
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/data/ModelNet40Loader.py
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/train_cls.py

class ScanObjectNN(Dataset):
    def __init__(self, num_points=2048, spilt='training'):
        self.partition = spilt
        if spilt=='train':
            self.partition = 'training'
        self.data, self.label = load_scanobjectnn_data(self.partition)
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return {'pc': pointcloud, 'label': label.item()}
    def __len__(self):
        return self.data.shape[0]

class ScanObjectNN_diffusion(Dataset):
    def __init__(self, num_points=2048, spilt='training'):
        self.partition = spilt
        if spilt=='train':
            self.partition = 'training'
        self.data, self.label = load_scanobjectnn_data(self.partition)
        self.num_points = num_points

    def __getitem__(self, item):
        pc = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            # pc = translate_pointcloud(pc)
            np.random.shuffle(pc)

        shift = pc.mean(axis=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
        pc = (pc - shift) / scale
            
        # scale = torch.ones([1, 1])
        # shift = torch.zeros([1, 3])
        return {'pc': pc, 'label': label.item(), 'shift': shift, 'scale': scale}

    def __len__(self):
        return self.data.shape[0]
# def download():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, 'data')
#     if not os.path.exists(DATA_DIR):
#         os.mkdir(DATA_DIR)
#     if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
#         # note that this link only contains the hardest perturbed variant (PB_T50_RS).
#         # for full versions, consider the following link.
#         www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
#         # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
#         zipfile = os.path.basename(www)
#         os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
#         os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#         os.system('rm %s' % (zipfile))

def load_scanobjectnn_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    # h5_name = BASE_DIR + '/data/h5_files/main_split_nobg/' + partition + '_objectdataset.h5'
    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
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
class ModelNet40Rscnn(Dataset):
    def __init__(self, split, data_path, train_data_path,
                 valid_data_path, test_data_path, num_points):

        self.split = split
        self.num_points = num_points
        _transforms = transforms.Compose([rscnn_d_utils.PointcloudToTensor()])
        rscnn_params = {
            'num_points': 1024,  # although it does not matter
            'root': data_path,
            'transforms': _transforms,
            'train': (split in ["train", "valid"]),
            'data_file': {
                'train': train_data_path,
                'valid': valid_data_path,
                'test':  test_data_path
            }[self.split]
        }
        self.rscnn_dataset = rscnn_ModelNet40Cls(**rscnn_params)
        self.PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()

    def __len__(self):
        return self.rscnn_dataset.__len__()

    def __getitem__(self, idx):
        point, label = self.rscnn_dataset.__getitem__(idx)
        # for compatibility with the overall code
        point = np.array(point)
        label = label[0].item()

        return {'pc': point, 'label': label}

    def batch_proc(self, data_batch, device):
        point = data_batch['pc'].to(device)
        if self.split == "train":
            # (B, npoint)
            fps_idx = pointnet2_utils.furthest_point_sample(point, 1200)
            fps_idx = fps_idx[:, np.random.choice(1200, self.num_points,
                                                  False)]
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            point.data = self.PointcloudScaleAndTranslate(point.data)
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(
                point, self.num_points)  # (B, npoint)
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()
        # to maintain compatibility
        point = point.cpu()
        return {'pc': point, 'label': data_batch['label']}


# distilled from the following sources:
# https://github.com/charlesq34/pointnet2/blob/7961e26e31d0ba5a72020635cee03aac5d0e754a/modelnet_h5_dataset.py
# https://github.com/charlesq34/pointnet2/blob/7961e26e31d0ba5a72020635cee03aac5d0e754a/train.py
class ModelNet40PN2(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.dataset_name = 'modelnet40_pn2'
        data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]
        pointnet2_params = {
            'list_filename': data_path,
            # this has nothing to do with actual dataloader batch size
            'batch_size': 32,
            'npoints': num_points,
            'shuffle': False
        }

        # loading all the pointnet2data
        self._dataset = pointnet2_ModelNetH5Dataset(**pointnet2_params)
        all_pc = []
        all_label = []
        while self._dataset.has_next_batch():
            # augmentation here has nothing to do with actual data_augmentation
            pc, label = self._dataset.next_batch(augment=False)
            all_pc.append(pc)
            all_label.append(label)
        self.all_pc = np.concatenate(all_pc)
        self.all_label = np.concatenate(all_label)

    def __len__(self):
        return self.all_pc.shape[0]

    def __getitem__(self, idx):
        return {'pc': self.all_pc[idx], 'label': np.int64(self.all_label[idx])}

    def batch_proc(self, data_batch, device):
        if self.split == "train":
            point = np.array(data_batch['pc'])
            point = self._dataset._augment_batch_data(point)
            # converted to tensor to maintain compatibility with the other code
            data_batch['pc'] = torch.tensor(point)
        else:
            pass

        return data_batch


class ModelNet40Dgcnn(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]

        dgcnn_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            'num_points': num_points,
            "data_path":  self.data_path
        }
        self.dataset = dgcnn_ModelNet40(**dgcnn_params)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        pc, label = self.dataset.__getitem__(idx)
        # shift = pc.mean(axis=0).reshape(1, 3)
        # scale = pc.flatten().std().reshape(1, 1)
        # pc = (pc - shift) / scale
        return {'pc': pc, 'label': label.item()}

def load_data_customized(data_path):
    npz = np.load(data_path)
    test_pc = npz['test_pc'][...,:3]  # [..., K, 3]
    test_label = npz['test_label']
    return test_pc, test_label


def load_data_customized(data_path):
    npz = np.load(data_path)
    test_pc = npz['test_pc'][...,:3]  # [..., K, 3]
    test_label = npz['test_label']
    return test_pc, test_label

class ModelNet40Noise(Dataset):
    def __init__(self,  split, train_data_path,
                 valid_data_path, test_data_path, num_points, noise_level=0.12):
        self.noise_level = noise_level
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]
        self.num_points = num_points
        self.data, self.label = dgcnn_load_data(self.data_path)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.split == 'train':
            pointcloud = pointcloud + np.random.randn(*pointcloud.shape).astype('float32') * self.noise_level
        label = self.label[item]

        return {'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_Customized(Dataset):
    def __init__(self, split, num_points, data_path, diffusion=False):
        assert split == 'test'
        self.diffusion = diffusion
        self.data, self.label = load_data_customized(data_path)
        self.num_points = num_points
        self.partition = split

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if not self.diffusion:
            return {'pc': pointcloud, 'label': label.item()}
        else:
            shift = pointcloud.mean(axis=0).reshape(1, 3)
            scale = pointcloud.flatten().std().reshape(1, 1)
            pointcloud = (pointcloud - shift) / scale
            return {'pc': pointcloud, 'label': label.item(), 'shift': shift, 'scale': scale}

    def __len__(self):
        return self.data.shape[0]




class ModelNet40Diffusion(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.partition = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.partition]
        self.data, self.label = dgcnn_load_data(self.data_path)
        self.num_points = num_points


    def __getitem__(self, item):
        pc = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pc)
        shift = pc.mean(axis=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
        pc = (pc - shift) / scale
        return {'pc': pc, 'label': label.item(), 'shift': shift, 'scale': scale}
        
    def __len__(self):
        return self.data.shape[0]


class ModelNet40Diffusion_si_adv(Dataset):
    def __init__(self):
        self.data, self.label = load_si_adv_data()

    def __getitem__(self, item):
        pc = self.data[item]
        label = self.label[item]
        shift = pc.mean(axis=0).reshape(1, 3)
        scale = pc.flatten().std().reshape(1, 1)
        pc = (pc - shift) / scale
        return {'pc': pc, 'label': label.item(), 'shift': shift, 'scale': scale}
        #return {'pc': pc, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]
        


def load_data(data_path,corruption,severity):

    if data_path.endswith('-Opt'):
        DATA_DIR = os.path.join(data_path, 'convonet_opt-data_' + corruption + '_' +str(severity) + '.npz')
        npz = np.load(DATA_DIR)
        test_pc = npz['test_pc'][...,:3]  # [..., K, 3]
        test_label = npz['test_label']
        return test_pc, test_label
    else:
        DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
        # if corruption in ['occlusion']:
        #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
        LABEL_DIR = os.path.join(data_path, 'label.npy')
        all_data = np.load(DATA_DIR)
        all_label = np.load(LABEL_DIR)
        return all_data, all_label

class ModelNet40C(Dataset):
    def __init__(self, split, test_data_path,corruption,severity):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return {'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]

class ModelNet40C_Diffusion(Dataset):
    def __init__(self, split, test_data_path,corruption,severity):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        shift = pointcloud.mean(axis=0).reshape(1, 3)
        scale = pointcloud.flatten().std().reshape(1, 1)
        pointcloud = (pointcloud - shift) / scale
        return {'pc': pointcloud, 'label': label.item(),'shift': shift, 'scale': scale}

    def __len__(self):
        return self.data.shape[0]


def create_dataloader(split, cfg):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {
        "split": split
    }

    if cfg.EXP.DATASET == "ScanObjectNN":
        dataset = ScanObjectNN(spilt=dataset_args['split'])
    elif cfg.EXP.DATASET == "ScanObjectNN_diffusion":
        dataset = ScanObjectNN_diffusion(spilt=dataset_args['split'])
    elif cfg.EXP.DATASET == "modelnet40_rscnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_RSCNN))
        dataset = ModelNet40Rscnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_pn2":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_PN2))
        dataset = ModelNet40PN2(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_c_diffusion":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C_Diffusion(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_diffusion":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Diffusion(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_customized":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_CUSTOMIZED))
        dataset = ModelNet40_Customized(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_customized_diffusion":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_CUSTOMIZED))
        dataset_args['diffusion'] = True
        dataset = ModelNet40_Customized(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_noise":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_Noise))
        # dataset_args['diffusion'] = True
        dataset = ModelNet40Noise(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_si_adv_diffusion":
        dataset = ModelNet40Diffusion_si_adv()
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    )

def main():
    dataset = ScanObjectNN(num_points=2048, spilt='test')

    data = dataset.data
    label = dataset.label

    np.save('scan_clean_data.npy', data)
    np.save('scan_clean_label.npy', label)

if __name__ == "__main__":
    main()