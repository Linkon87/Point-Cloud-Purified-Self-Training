import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets.dgcnn_modelnet40_dataset import ModelNet40
# from datasets.modelnet40_c_dataset import ModelNet40C
from datasets.attack_recons_dataset import ModelNet40C
from datasets.difussion_source import ModelNet40_difussion_source
from datasets.ScanObjectNN import ScanObjectNN
from datasets.ScanObjectNN_test import ScanObjectNN_test


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# def prepare_test_data(args, with_da=False):
#     dataset = ModelNet40C(split="test", test_data_path=os.path.join(args.dataroot, 'modelnet40_c'), corruption=args.corruption, severity=args.severity, data_augment=with_da)
#     return dataset

def prepare_test_data(args, with_da=False,with_strong=False):
    dataset = ModelNet40C(split="test",test_data=args.test_data,test_label=args.test_label,data_augment=with_da,data_augment_strong=with_strong)
    return dataset

def prepare_train_data(args):
    dataset = ModelNet40(num_points=1024, data_path=os.path.join(args.dataroot, 'modelnet40_ply_hdf5_2048', 'train_files.txt'))
    return dataset

def prepare_t_data(args):
    dataset = ModelNet40(num_points=1024, data_path=os.path.join(args.dataroot, 'modelnet40_ply_hdf5_2048', 'test_files.txt'))
    return dataset

def create_dataloader(dataset, args, shuffle=False, drop_last=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker)

def prepare_source_train_data(args):
    dataset = ModelNet40_difussion_source(test_data=args.source_data,test_label=args.source_label)
    return dataset

def prepare_scanobjectnn_train_data(args):
    dataset = ScanObjectNN()
    return dataset

def prepare_scanobjectnn_test_data(args,with_da=False,with_strong=False):
    dataset = ScanObjectNN_test(test_data=args.test_data,test_label=args.test_label,data_augment=with_da,data_augment_strong=with_strong)
    return dataset
