import torch.nn as nn
from networks.pointMLP_ori import pointMLP as pointMLP_original


class pointMLP(nn.Module):

    def __init__(self):
        super(pointMLP,self).__init__()

        num_classes = 40
        self.model = pointMLP_original(num_classes=num_classes)


    def forward(self, pc):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)
        return logit
