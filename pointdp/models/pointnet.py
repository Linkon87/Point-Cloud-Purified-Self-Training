# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model import PointNetCls
from all_utils import DATASET_NUM_CLASS
from .autoencoder import Autoencoder

class PointNet(nn.Module):

    def __init__(self, dataset, task,  ae=None):
        super().__init__()
        self.task = task
        self.is_ae = ae.IS
        
        # num_class = DATASET_NUM_CLASS[dataset]
        if task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls','cls_trans']:
            if task in ['si_adv_attack_cls']: num_class = 40
            else: 
                num_class = DATASET_NUM_CLASS[dataset]
                
            self.model = PointNetCls(k=num_class, feature_transform=True)

        else:
            assert False
        if self.is_ae:
            self.ae = Autoencoder(encoder=ae.encoder, decoder=ae.decoder, truncate=ae.truncate, t=ae.t)

    def forward(self, pc, scale=None, shift=None, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.transpose(2, 1).float()
        if self.task in ['cls','ae','diffusion','attack_cls','non_adaptive_attack_cls','sde_attack_cls','si_adv_attack_cls','cls_trans']:
            assert cls is None
            if self.is_ae:
                feature, pc = self.ae(pc)
                if self.ae.decoder_name == 'diffusion':
                    pc = pc * scale + shift
                pc = pc.permute(0, 2, 1).contiguous()
            logit, _,  trans_feat = self.model(pc)
            pc = pc.permute(0, 2, 1).contiguous()
            out = {'logit': logit, 'trans_feat': trans_feat,'recons':pc}
        else:
            assert False

        return out

    def classification_forward(self, pc,scale=None, shift=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc * scale + shift
        pc = pc.permute(0, 2, 1).contiguous()
        logit,_, trans_feat = self.model(pc)
        out = {'logit': logit, 'trans_feat': trans_feat}
        return out