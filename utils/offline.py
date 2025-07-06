import torch
import copy
import statistics
import numpy as np

from .discrepancy import *
import os

def offline(trloader, ext, num_classes=40):
    # if os.path.exists('offline.pth'):
    #     data = torch.load('offline.pth')
    #     return data

    ext.eval()

    feat_stack = []
    label_stack = []

    with torch.no_grad():
        print('offline ...')
        for batch_idx, (inputs, labels) in enumerate(trloader):

            label_stack.append(labels.cuda())
            feat = ext(inputs[0].cuda())
            feat_stack.append(feat)
            # print('offline process rate: %.2f%%\r' % ((batch_idx + 1) / len(trloader) * 100.), end='')


    feat_all = torch.cat(feat_stack)
    label_all = torch.cat(label_stack)
    feat_cov = covariance(feat_all)
    feat_mean = feat_all.mean(dim=0)

    n_, d = feat_all.shape[:2]

    num_categories = torch.zeros(num_classes, dtype=torch.int).cuda()
    num_categories.scatter_add_(dim=0, index=label_all, src=torch.ones_like(label_all, dtype=torch.int))
    #
    feat_categories = torch.zeros(num_classes, n_, d).cuda()
    feat_categories.scatter_add_(dim=0, index=label_all[None, :, None].expand(-1, -1, d), src=feat_all[None, :, :])
    #
    feat_mean_categories = feat_categories.sum(dim=1) # K, D
    feat_mean_categories /= num_categories[:, None]  ## (40 , 256)

    feat_corr = feat_categories.permute(0, 2, 1) @ feat_categories # K, D, D
    feat_corr /= num_categories[:, None, None]

    feat_cov_categories = feat_corr - feat_mean_categories[:, :, None] @ feat_mean_categories[:, None, :]

    # torch.save((feat_mean, feat_cov, feat_mean_categories, feat_cov_categories), 'offline.pth')
    return feat_mean, feat_cov, feat_mean_categories, feat_cov_categories
