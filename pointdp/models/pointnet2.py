import torch
import torch.nn as nn
from pointnet2_pyt.pointnet2.models.pointnet2_msg_cls import Pointnet2MSG
from pointnet2_scan import PointNet2_scan
from all_utils import DATASET_NUM_CLASS
from .autoencoder import Autoencoder

class PointNet2(nn.Module):

    def __init__(self, task, dataset, version_cls, ae=None):
        super().__init__()
        self.task =  task
        self.is_ae = ae.IS
        if task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            if task in ['si_adv_attack_cls']: num_class = 40
            else: 
                num_class = DATASET_NUM_CLASS[dataset]
            # if num_class == 15:
            #     self.model = PointNet2_scan()
            # else:
            self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True, version=version_cls)
        else:
            assert False

        if self.is_ae:
            self.ae = Autoencoder(encoder=ae.encoder, decoder=ae.decoder, truncate=ae.truncate, t=ae.t)

    def forward(self, pc, scale=None, shift=None,normal=None, cls=None):
        pc = pc.to(next(self.parameters()).device)
        # pc = pc.permute(0, 2, 1).contiguous()
        if self.task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls']:
            assert cls is None
            assert normal is None
            if self.is_ae:
                feature, pc = self.ae(pc)
                if self.ae.decoder_name == 'diffusion':
                    pc = pc * scale + shift
            logit = self.model(pc)
            out = {'logit': logit,'recons':pc}
        else:
            assert False
        return out

    def classification_forward(self, pc, scale=None, shift=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc * scale + shift
        logit = self.model(pc)
        out = {'logit': logit}
        return out