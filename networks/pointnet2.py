import torch
import torch.nn as nn
from networks.pointnet2_msg_cls import Pointnet2MSG


class PointNet2(nn.Module):

    def __init__(self, num_class=40, version_cls=1.0):
        super(PointNet2,self).__init__()
        num_class = num_class
        self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True, version=version_cls)

    def forward(self, pc):
        pc = pc.to(next(self.parameters()).device)
        # pc = pc.permute(0, 2, 1).contiguous()

        logit = self.model(pc)

        return logit
