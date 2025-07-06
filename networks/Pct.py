import torch.nn as nn
from networks.PctCls import Pct as Pct_original


class PCT(nn.Module):

    def __init__(self,num_classes=40):
        super(PCT, self).__init__()
        num_classes = num_classes
            # default arguments
        class Args:
                def __init__(self):
                    self.dropout = 0.
        args = Args()
        self.model = Pct_original(args, output_channels=num_classes)

    def forward(self, pc):
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)
        return logit
