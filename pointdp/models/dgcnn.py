import torch.nn as nn
import torch.nn.functional as F
from dgcnn.pytorch.model import DGCNN as DGCNN_original
from all_utils import DATASET_NUM_CLASS
from .autoencoder import Autoencoder

class DGCNN(nn.Module):

    def __init__(self, task, dataset, ae=None):
        super().__init__()
        self.task = task
        self.dataset = dataset
        self.is_ae = ae.IS

        if task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            if task in ['si_adv_attack_cls']: num_classes = 40
            else : num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
            args = Args()
            self.model = DGCNN_original(args, output_channels=num_classes)

        else:
            assert False
        if self.is_ae:
            self.ae = Autoencoder(encoder=ae.encoder, decoder=ae.decoder, truncate=ae.truncate, t=ae.t)

    def forward(self, pc, scale=None, shift=None, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            assert cls is None
            if self.is_ae:
                feature, pc = self.ae(pc)
                if self.ae.decoder_name == 'diffusion':# and self.task !='si_adv_attack_cls' :
                    pc = pc * scale + shift  #归一化的点云拿去扩散，重建后的点云再取消归一化，再拿去分类 对于si_adv，怎么取消归一化未知，scale和shift未知
                pc = pc.permute(0, 2, 1).contiguous()
            logit = self.model(pc)
            pc = pc.permute(0, 2, 1).contiguous()
            # pc = (pc - shift) / scale
            out = {'logit': logit,'recons':pc}
        else:
            assert False
        return out

    def classification_forward(self, pc, scale=None, shift=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc * scale + shift
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)
        out = {'logit': logit}
        return out

