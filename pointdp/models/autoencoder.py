from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .diffusion import *



class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, a, b):
        x, y = a, b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).to(a).long()
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P.min(1)[0], P.min(2)[0]

    def forward(self, preds, gts):
        dl, dr = self.batch_pairwise_dist(gts, preds)
        # mins, _ = torch.min(P, 1)
        # loss_1 = torch.mean(mins)
        # mins, _ = torch.min(P, 2)
        # loss_2 = torch.mean(mins)
        # print(dl.mean() + dr.mean())
        return dl.mean() + dr.mean()


class DeepSymEncoder(nn.Module):
    def __init__(self):
        super(DeepSymEncoder,self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1, bias=False)

        self.conv6 = nn.Conv1d(512, 128, kernel_size=1, bias=False)
        self.conv7 = nn.Conv1d(128, 32, kernel_size=1, bias=False)
        self.conv8 = nn.Conv1d(32, 1, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(512, 512, bias=False)
        self.linear2 = nn.Linear(512, 256, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(128)
        self.bn7 = nn.BatchNorm1d(32)
        self.bn8 = nn.BatchNorm1d(1)

        self.bn9 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(256)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.2)
        x = self.bn5(self.conv5(x))
        ### x -> (batch_size, number_points, dim)
        
        x = torch.sort(x, dim = 1, stable = True)

        ## TODO: interpolation to make number_points consistent
        x = x.permute(0, 2, 1).contiguous()

        x = F.leaky_relu(self.bn6(self.conv6(x)),0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)),0.2)
        x = self.bn8(self.conv8(x))

        x = x.permute(0, 2, 1).contiguous().squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.linear2(x)
        return torch.squeeze(x)


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder,self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(256, 1024, kernel_size=1, bias=False)

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)

    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.2)
        x = self.bn5(self.conv5(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.linear2(x)
        return torch.squeeze(x)

class MLPDecoder(nn.Module):
    def __init__(self, feat_dims=256):
        super(MLPDecoder,self).__init__()
        self.linear1 = nn.Linear(feat_dims, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 1024, bias=False)
        self.linear3 = nn.Linear(1024, 1024 * 3, bias=False)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
    
    def forward(self,input):
        batch_size = input.size(0)
        x = F.relu(self.bn1(self.linear1(input)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        return x.reshape(batch_size, 1024, 3)


class FoldNetDecoder(nn.Module):
    def __init__(self,feat_dims=256):
        super(FoldNetDecoder,self).__init__()
        self.sphere = np.load("sphere.npy")
        self.m = 2025
        self.folding1 = nn.Sequential(
            nn.Conv1d(feat_dims+3, feat_dims, 1),
            nn.BatchNorm1d(feat_dims),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims//2, 1),
            nn.BatchNorm1d(feat_dims//2),
            nn.ReLU(),
            nn.Conv1d(feat_dims//2, 3, 1),
        )  
        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dims+3, feat_dims, 1),
            nn.BatchNorm1d(feat_dims),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims//2, 1),
            nn.BatchNorm1d(feat_dims//2),
            nn.ReLU(),
            nn.Conv1d(feat_dims//2, 3, 1),
        )

    def build_grid(self, batch_size):
        points = self.sphere
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, input):
        x = input.repeat(1, 1, self.m)      # (batch_size, feat_dims, num_points)
        # print(x.shape)
        points = self.build_grid(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        # print(points.shape)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)            # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)   # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)


class Autoencoder(nn.Module):
    def __init__(self,encoder='pointnet',decoder='diffusion',truncate=False,t=200):
        super(Autoencoder,self).__init__()
        
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.truncate = truncate
        self.t = t
        if encoder == 'pointnet':
            self.encoder = PointNetEncoder()
        elif encoder == 'deepsym':
            pass 
            #TODO
        else:
            assert False
        
        if decoder == 'foldnet':
            self.decoder = FoldNetDecoder()
        elif decoder == 'diffusion':
            self.decoder = DiffusionPoint(
                net = PointwiseNet(point_dim=3, context_dim=256, residual=True),
                var_sched = VarianceSchedule()
            )   
        elif decoder == 'mlp':
            self.decoder = MLPDecoder()
        else:
            assert False

        self.loss = ChamferLoss()

    def forward(self,pc):
        pc = pc.to(next(self.parameters()).device)
        # print(pc.requires_grad)
        if pc.shape[2] == 3:
            pc = pc.permute(0, 2, 1).contiguous()
            feature = self.encoder(pc)
        else:
            feature = self.encoder(pc)

        if self.decoder_name == 'diffusion':
            if self.truncate:
                recon = self.denoiser(pc.permute(0, 2, 1),t=self.t,context=feature)    
            else:    
                recon = self.decode(feature,pc.shape[2])
        else:
            recon = self.decoder(feature)
        return feature, recon

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, d, N).
        """
        x = x.to(next(self.parameters()).device)
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1).contiguous()
        code = self.encoder(x)
        return code 

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        assert hasattr(self.decoder,'sample')
        return self.decoder.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)
    
    def denoiser(self, pointcloud, t, flexibility=0.0, ret_traj=False, context=None):
        assert hasattr(self.decoder,'truncated_sample')
        return self.decoder.truncated_sample(pointcloud, t, context=context)

    def get_loss(self, input):
        input = input.to(next(self.parameters()).device)
        if input.shape[2] == 3:
            input_1 = input.permute(0, 2, 1).contiguous()
            if isinstance(self.decoder,DiffusionPoint):
                code = self.encode(input_1)
                loss = self.decoder.get_loss(input, code)
                return loss
            else:
                feature, output = self.forward(input_1)
                return self.loss(input, output)
        else:
            if isinstance(self.decoder,DiffusionPoint):
                input_1 = input.permute(0, 2, 1).contiguous()
                code = self.encode(input)
                loss = self.decoder.get_loss(input_1, code)
                return loss
            else:
                feature, output = self.forward(input)
                return self.loss(input, output)

    ##### L1 Norm Distance ######
    def get_feature_loss(self, input, target):
        input = input.to(next(self.parameters()).device)
        feature = self.encode(input)
        return torch.mean(torch.abs(feature-target))
    ##############################

    # def get_feature_loss(self, input, target):
    #     input = input.to(next(self.parameters()).device)
    #     feature = self.encode(input)
    #     kl_loss = nn.KLDivLoss()
    #     input = F.log_softmax(feature, dim = -1)   
    #     return kl_loss(input,target)