import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from networks.PointNetCls import PointNetfeat

class POINTNET(nn.Module):
    def __init__(self, k=40):
        super(POINTNET, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1).float()

        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        logit = self.fc3(x) # 256 ,k
        x = F.log_softmax(logit, dim=1)
        # return x
        return logit