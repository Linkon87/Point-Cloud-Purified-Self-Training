import torch.nn as nn
import torch.nn.functional as F
from networks.cuvenet_ori import CurveNet as CurveNet_og



class CurveNet(nn.Module):

    def __init__(self,num_classes=40):
        super(CurveNet,self).__init__()

        num_classes = num_classes
        self.model = CurveNet_og(num_classes = num_classes)

    def forward(self, pc):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)

        return logit

