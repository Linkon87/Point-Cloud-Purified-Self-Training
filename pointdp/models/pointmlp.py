import torch.nn as nn
from pointMLP.classification_ModelNet40.models.pointmlp import pointMLP as pointMLP_original
from all_utils import DATASET_NUM_CLASS
from .autoencoder import Autoencoder

class pointMLP(nn.Module):

    def __init__(self, task, dataset, ae = None):
        super().__init__()
        self.task = task
        self.dataset = dataset
        self.is_ae = ae.IS

        if task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            if task in ['si_adv_attack_cls']: num_classes = 40
            else:
             num_classes = DATASET_NUM_CLASS[dataset]
            self.model = pointMLP_original(num_classes=num_classes)
        else:
            assert False
        if self.is_ae:
            self.ae = Autoencoder(encoder = ae.encoder, decoder = ae.decoder, truncate = ae.truncate, t=ae.t)


    def forward(self, pc, scale = None, shift = None, cls = None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            assert cls is None
            if self.is_ae:
                feature, pc = self.ae(pc)
                if self.ae.decoder_name == 'diffusion':
                    pc = pc * scale + shift
                pc = pc.permute(0, 2, 1).contiguous()
            logit = self.model(pc)
            pc = pc.permute(0, 2, 1).contiguous()
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
