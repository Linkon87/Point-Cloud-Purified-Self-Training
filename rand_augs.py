import os
import random

import numpy as np
import torch
import math
from tensorboardX import SummaryWriter


def jittering(data, prob=0.9):
    if np.random.uniform() < prob:
        data[:,:3] += np.random.normal(0, 0.01, size=data[:,:3].shape)
    return data

def scaling(data, prob=0.9):
    if np.random.uniform() < prob:
        data[:,:3] *= np.random.uniform(0.9, 1.1)
    return data


def shifting(data, prob=0.9):
    if np.random.uniform() < prob:
        data[:,:2] += np.random.normal(0., 0.01, size=data[:,:2].shape)
    return data


# def rotating(data, prob=0.4):
#     if np.random.uniform() < prob:
#         angle = np.random.uniform(0, 360)
#         rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#         data[:,:2] = data[:,:2].dot(rotation_matrix)
#     return data

def rotating(data, prob=0.9):
    if np.random.uniform() < prob:
        angle = np.random.uniform(0, 360) *np.pi / 180 # Convert degrees to radians
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        data[:, :2] = data[:, :2].dot(rotation_matrix)
    return data



def random_horizontal_flip(x, p=0.9):
    ''' randomly apply horizontal flip to pointcloud tensor'''
    if np.random.uniform() < p:
        x[:, 1] = x[:, 1] * (-1)
    return x


def random_vertical_flip(x, p=0.5):
    ''' randomly apply vertical flip to pointcloud tensor'''
    if np.random.uniform() < p:
        x[:, 0] = x[:, 0] * (-1)
    return x


# def translate_point_cloud(pcd, trans=(0.1, 0.2, 0.3)):
#     '''translates the given point cloud numpy array'''
#     assert len(trans) == 3, "you must pass 3 values in trans"
#     assert isinstance(trans, tuple), "you must pass a tuple of size 3"
#
#     # Create an identity matrix and add the translation vector as the last column
#     trans_m = np.eye(4)
#     trans_m[:3, 3] = np.array(trans)
#
#     # Convert the point cloud to homogeneous coordinates by adding a column of ones
#     homogeneous_pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
#
#     # Apply the translation to the point cloud in homogeneous coordinates
#     translated_pcd = np.dot(homogeneous_pcd, trans_m)
#
#     # Convert back to Cartesian coordinates by dividing by the last column
#     translated_pcd = translated_pcd[:, :3] / translated_pcd[:, 3][:, np.newaxis]
#
#     return translated_pcd

def translate_point_cloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def shear_point_cloud(pcd, shear_deg=0.5):
    '''shears the given point cloud numpy array'''
    shear_value = np.tan(np.radians(shear_deg))  # Use tangent for shear value

    shear_m = np.array([[1, shear_value, 0],  # Only shear in one direction for simplicity
                        [0, 1, 0],
                        [0, 0, 1]])
    return np.dot(pcd, shear_m)


def rescale_point_cloud(pcd, rescale=(2.0, 1.0, 0.5)):
    """Rescales the given point cloud numpy array"""
    assert len(rescale) == 3, "You must pass 3 values in rescale"

    rescale_m = np.array([[rescale[0], 0, 0], [0, rescale[1], 0], [0, 0, rescale[2]]])
    return np.dot(pcd, rescale_m)


# def rotate_point_cloud(pcd, rot_deg=10.0): #45.0
#     """Rotates the given point cloud numpy array"""
#     alpha = np.radians(rot_deg)
#     cos = np.cos(alpha)
#     sin = np.sin(alpha)
#
#     b1 = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
#     pc1 = np.dot(pcd, b1)
#     b2 = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
#     pc2 = np.dot(pc1, b2)
#     b3 = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
#     rot_m = np.dot(pc2, b3)
#
#     return rot_m
def rotate_point_cloud(pcd, rot_deg=1.0):  # 45.0
    """Rotates the given point cloud numpy array"""
    alpha = np.radians(rot_deg)
    cos = np.cos(alpha)
    sin = np.sin(alpha)

    b1 = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    b2 = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    b3 = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    combined_rot_m = np.dot(np.dot(b1, b2), b3)  # Combine the rotation matrices
    rot_pcd = np.dot(pcd, combined_rot_m)  # Apply the combined rotation to the point cloud

    return rot_pcd

def normalize_point_cloud(data):
    """Normalize a point cloud feature. Scales features between [0,1]. Should only be applied on a dataset-level."""
    for i in range(3):  # assuming the point cloud has 3 features
        data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())
    return data

def ScaleX(pts,v=0.4): # (0 , 2)
    pts = torch.from_numpy(pts)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 0 ] *= scaler
    pts = pts.numpy()
    return pts

def ScaleY(pts,v=0.4): # (0 , 2)
    pts = torch.from_numpy(pts)
    assert 0 <= v <= 0.5
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 1 ] *= scaler
    pts = pts.numpy()
    return pts

def ScaleZ(pts,v=0.4): # (0 , 2)
    assert 0 <= v <= 0.5
    pts = torch.from_numpy(pts)
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 2 ] *= scaler
    pts = pts.numpy()
    return pts
def NonUniformScale(pts,v=0.4): # Resize in [0.5 , 1.5]
    assert 0 <= v <= 0.5
    pts = torch.from_numpy(pts)
    scaler = np.random.uniform( low = 1 - v, high =  1 + v, size = 3 )
    pts[:, 0:3] *= torch.from_numpy(scaler).float()
    pts = pts.numpy()
    return pts

def Resize(pts,v=0.4):
    assert 0 <= v <= 0.5
    pts = torch.from_numpy(pts)
    scaler = np.random.uniform(low = 1-v, high =  1 + v)
    pts[:, 0:3 ] *= scaler
    pts = pts.numpy()
    return pts
def UniformTranslate(pts ,v=0.5):
    assert 0 <= v <= 1
    pts = torch.from_numpy(pts)
    translation = (2 * np.random.random() - 1 ) * v
    pts[:, 0:3] += translation
    pts = pts.numpy()
    return pts

def NonUniformTranslate(pts ,v=0.5):
    assert 0 <= v <= 1
    pts = torch.from_numpy(pts)
    translation = (2 * np.random.random(3) - 1 ) * v
    pts[:, 0:3] +=  torch.from_numpy(translation).float()
    pts = pts.numpy()
    return pts
def RandomDropout(pts,v):
    assert 0.3 <= v <= 0.875
    pts = torch.from_numpy(pts)
    dropout_rate = v
    drop = torch.rand(pts.size(0)) < dropout_rate
    save_idx = np.random.randint(pts.size(0))
    save_point = pts[save_idx].clone()
    pts[drop] = save_point
    pts = pts.numpy()
    return pts
def RandomErase(pts,v=0.1):
    assert 0 <= v <= 0.5
    pts = torch.from_numpy(pts)
    "v : the radius of erase ball"
    random_idx = np.random.randint(pts.size(0))
    erase_point = pts[random_idx].clone()
    mask = torch.sum((pts[:,0:3] - pts[random_idx,0:3]).pow(2), dim = 1) < v ** 2
    pts[mask] = erase_point
    pts = pts.numpy()
    return pts


def ShearXY(points, v=0.1):
    assert 0 <= v <= 0.5
    points = torch.from_numpy(points).float()
    a, b = v * (2 * np.random.random(2) - 1)
    shear_matrix = torch.tensor([[1, 0, 0],
                                 [0, 1, 0],
                                 [a, b, 1]], dtype=torch.float32)
    points[:, 0:3] = torch.matmul(points[:, 0:3], shear_matrix.t())
    points = points.numpy()
    return points
def ShearYZ(points,v=0.1):
    assert 0 <= v <= 0.5
    points = torch.from_numpy(points).float()
    b , c  = v * (2 * np.random.random(2) - 1)
    shear_matrix = np.array([[1, b, c],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=torch.float32)
    shear_matrix = torch.from_numpy(shear_matrix).float()

    points[:,0:3] = points[:, 0:3] @ shear_matrix.t()
    points = points.numpy()
    return points
def ShearXZ(points,v=0.1):
    assert 0 <= v <= 0.5
    points = torch.from_numpy(points).float()

    a , c  = v * (2 * np.random.random(2) - 1)
    shear_matrix = np.array([[1, 0, 0],
                             [a, 1, c],
                             [0, 0, 1]], dtype=torch.float32)
    shear_matrix = torch.from_numpy(shear_matrix).float()
    points[:,0:3] = points[:, 0:3] @ shear_matrix.t()
    points = points.numpy()

    return points
def GlobalAffine(points, v=0.1):
    assert 0 <= v <= 1
    points = torch.from_numpy(points).float()

    affine_matrix = torch.from_numpy(np.eye(3) + np.random.randn(3, 3) * v).float()
    affine_matrix = affine_matrix.float()
    points[:, 0:3] = torch.matmul(points[:, 0:3], affine_matrix.t())
    points = points.numpy()
    return points
def PointToNoise(points,v):
    assert 0 <= v <= 0.5
    points = torch.from_numpy(points)
    mask = np.random.random(points.size(0)) < v
    noise_idx = [idx for idx in range(len(mask)) if mask[idx] == True]
    pts_rand = 2 * (np.random.random([len(noise_idx), 3]) - 0.5) + np.mean(points[:,0:3].numpy(), axis= 0)

    points[noise_idx, 0:3] = torch.from_numpy(pts_rand).float()
    points = points.numpy()

    return points

def aug_list():
    l = [
        (jittering, 1),
        (scaling, 1),
        (shear_point_cloud, 1),
        (rotate_point_cloud, 1),
        (translate_point_cloud, 1),
        (shifting, 1),
        (ScaleX, 1),
        (ScaleY, 1),
        (ScaleZ, 1),
        (NonUniformScale, 1),
        (RandomErase, 1),
        (ShearXY, 1),
        (GlobalAffine, 1),
    ]
    return l
class RandAug:
    def __init__(self, n):
        """
        N : The number of augmentation choice
        """
        self.n = n
        self.augment_list = aug_list()
        # self.augment_list2 = [jittering, scaling,shear_point_cloud,rotate_point_cloud,translate_point_cloud,
        #                      shifting,ScaleX,ScaleY,ScaleZ,NonUniformScale,RandomErase,ShearXY,GlobalAffine]
        # self.augment_counts = {augment.__name__: 0 for augment in self.augment_list2}


    def __call__(self, pc):
        augment_list = aug_list()
        ops, weights = zip(*augment_list)
        ops = random.choices(ops, weights=weights, k=self.n)
        points = pc
        # ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
                points = op(points)
                # self.augment_counts[op.__name__] += 1
        return points



