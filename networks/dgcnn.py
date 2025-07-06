
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn_original import DGCNN as DGCNN_original

class DGCNN(nn.Module):

    def __init__(self,num_classes=40, task="cls"):
        super().__init__()
        self.task = task

        if task == "cls":
            num_classes = num_classes
            # default arguments
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    # self.dropout = 0.0
                    self.dropout = 0.0
                    self.leaky_relu = 1
            args = Args()
            self.model = DGCNN_original(args, output_channels=num_classes)
        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
        else:
            assert False

        return logit