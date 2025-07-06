import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_path)
from dataloader import create_dataloader
from torch.nn.utils import clip_grad_norm_
from time import time
from datetime import datetime
from progressbar import ProgressBar
import models
from collections import defaultdict
import os
import numpy as np
import argparse
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal, TrackTrain, smooth_loss, DATASET_NUM_CLASS, SORDefense,
    rscnn_voting_evaluate_cls, pn2_vote_evaluate_cls)
from configs import get_cfg_defaults
import pprint
from pointnet_pyt.pointnet.model import feature_transform_regularizer
import sys
import aug_utils
from models.common import get_linear_scheduler
from models.revdiffusion import RevGuidedDiffusion

from load_ae_model_scan import  load_model_opt_sched

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cpu':
    print('WARNING: Using CPU')




def check_inp_fmt(task, data_batch, dataset_name):  ## 
    if task in ['cls', 'cls_trans']:
        # assert set(data_batch.keys()) == {'pc', 'label'}
        # print(data_batch['pc'],data_batch['label'])
        pc, label = data_batch['pc'], data_batch['label']
        # special case made for modelnet40_dgcnn to match the
        # original implementation
        # dgcnn loads torch.DoubleTensor for the test dataset
        if dataset_name in ['modelnet40_dgcnn', 'modelnet40_noise']:
            assert isinstance(pc, torch.FloatTensor) or isinstance(
                pc, torch.DoubleTensor)
        else:
            assert isinstance(pc, torch.FloatTensor)
        assert isinstance(label, torch.LongTensor)
        assert len(pc.shape) == 3
        assert len(label.shape) == 1
        b1, _, y = pc.shape[0], pc.shape[1], pc.shape[2]
        b2 = label.shape[0]
        assert b1 == b2
        assert y == 3
        assert label.max().item() < DATASET_NUM_CLASS[dataset_name]
        assert label.min().item() >= 0
    elif task in ['ae', 'diffusion']:
        pc, label = data_batch['pc'], data_batch['label']
        if dataset_name == 'modelnet40_dgcnn':
            assert isinstance(pc, torch.FloatTensor) or isinstance(
                pc, torch.DoubleTensor)
        else:
            assert isinstance(pc, torch.FloatTensor)
        assert len(pc.shape) == 3
    elif task in ['si_adv_attack_cls']:
        pc, label = data_batch['pc'], data_batch['label']
    else:
        assert NotImplemented


def check_out_fmt(task, out, dataset_name):
    if task == 'cls':
        # assert set(out.keys()) == {'logit'}
        logit = out['logit']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    elif task in ['ae', 'diffusion']:
        pass
        # assert len(logit.shape) == 3
    elif task == 'cls_trans':
        assert set(out.keys()) == {'logit', 'trans_feat'}
        logit = out['logit']
        trans_feat = out['trans_feat']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(trans_feat, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert len(trans_feat.shape) == 3
        assert trans_feat.shape[0] == logit.shape[0]
        # 64 coming from pointnet implementation
        assert (trans_feat.shape[1] == trans_feat.shape[2]) and (trans_feat.shape[1] == 64)
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    else:
        assert NotImplemented


def get_inp(task, model, data_batch, batch_proc, dataset_name):  # 
    check_inp_fmt(task, data_batch, dataset_name)
    if not batch_proc is None:
        data_batch = batch_proc(data_batch, DEVICE)
        check_inp_fmt(task, data_batch, dataset_name)

    if isinstance(model, nn.DataParallel):
        model_type = type(model.module)
    else:
        model_type = type(model)

    if task in ['cls', 'cls_trans', 'ae', 'attack_cls', 'non_adaptive_attack_cls', 'advpc', 'sde_attack_cls','si_adv_attack_cls']:
        pc = data_batch['pc']
        inp = {'pc': pc.cuda()}
        if hasattr(model, 'ae') and model.ae.decoder_name == 'diffusion':
            inp['scale'] = data_batch['scale'].cuda()
            inp['shift'] = data_batch['shift'].cuda()
    elif task in ['diffusion']:
        inp = {
            'pc': data_batch['pc'].cuda(),
            'shift': data_batch['shift'].cuda(),
            'scale': data_batch['scale'].cuda()
        }
    # elif task in ['si_adv_attack_cls']:
    #     pc = data_batch['pc']
    #     inp = {'pc': pc.cuda()}
    else:
        assert False

    return inp


def get_loss(task, loss_name, data_batch, out, dataset_name, model):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    check_out_fmt(task, out, dataset_name)
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'], torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (
                                    1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0),
                                                                            label_2[i].unsqueeze(0).long()) * \
                                   data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'],
                                                                                                    label_2) * \
                           data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
        # source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'], torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (
                                    1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0),
                                                                            label_2[i].unsqueeze(0).long()) * \
                                   data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'],
                                                                                                    label_2) * \
                           data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
        elif loss_name == 'ce_chamfer':
            loss = smooth_loss(out['logit'], label) + model.ae.get_loss(data_batch['pc'])
        else:
            assert False
    elif task == 'cls_trans':
        label = data_batch['label'].to(out['logit'].device)
        trans_feat = out['trans_feat']
        logit = out['logit']
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'], torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (
                                    1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0),
                                                                            label_2[i].unsqueeze(0).long()) * \
                                   data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'],
                                                                                                    label_2) * \
                           data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'], torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (
                                    1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0),
                                                                            label_2[i].unsqueeze(0).long()) * \
                                   data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'],
                                                                                                    label_2) * \
                           data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            assert False
    elif task in ['ae', 'diffusion']:
        loss = model.ae.get_loss(data_batch['pc'])
    elif task in ['advpc']:
        loss = model.get_loss(data_batch['pc'])
    else:
        assert False

    return loss


def attack(task, loader, model, dataset_name, method, confusion=False):
    import attack_utils
    model.eval()

    if method.METHOD == 'advpc':
        model_ae = models.Autoencoder(
            encoder='pointnet',
            decoder='mlp')
        ##### HARDCODE #########################
        checkpoint = torch.load('./runs/point_ae_scan/model_best_test.pth')
        ########################################
        model_ae.load_state_dict(checkpoint['model_state'])
        model_ae.eval()

    if task == 'sde_attack_cls':
        model.ae.decoder = RevGuidedDiffusion(model.ae.decoder)

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    # with torch.no_grad():
    bar = ProgressBar(max_value=len(loader))
    time5 = time()
    if confusion:
        pred = []
        ground = []
        attacked = []
        recons = []
    for i, data_batch in enumerate(loader):
        if method.METHOD in ['spsa', 'nes', 'nattack'] and i % 5 != 0:
            continue
        time1 = time()
        if method.METHOD == 'spsa':
            if method.NORM == np.inf:
                data_batch['pc'] = attack_utils.spsa(data_batch, model, task, method.LOSS, dataset_name, iters=method.ITER,
                                                 alpha=method.EPS / 10, eps=method.EPS)
            else:
                data_batch['pc'] = attack_utils.l2_spsa(data_batch, model, task, method.LOSS, dataset_name,
                                                     iters=method.ITER,
                                                     alpha=method.EPS / 10, eps=method.EPS)
        if method.METHOD == 'nattack':
            if method.NORM == np.inf:
                data_batch['pc'] = attack_utils.nattack(data_batch, model, task, method.LOSS, dataset_name,
                                                    iters=method.ITER, alpha=method.EPS / 10, eps=method.EPS)
            else:
                data_batch['pc'] = attack_utils.l2_nattack(data_batch, model, task, method.LOSS, dataset_name,
                                                        iters=method.ITER, alpha=method.EPS / 10, eps=method.EPS)
        elif method.METHOD == 'pgd':
            data_batch['pc'] = attack_utils.pgd(data_batch, model, task, method.LOSS, dataset_name, step=method.ITER,
                                                alpha=method.EPS / 10, eps=method.EPS, p=method.NORM)
            # pass
        elif method.METHOD == 'cw':
            data_batch['pc'] = attack_utils.cw(data_batch, model, task, method.LOSS, dataset_name, step=method.ITER,
                                               p=method.NORM)
        elif method.METHOD == 'knn':
            data_batch['pc'] = attack_utils.knn(data_batch, model, task, method.LOSS, dataset_name, step=method.ITER)
        elif method.METHOD == 'advpc':
            data_batch['pc'] = attack_utils.advpc(data_batch, model, dataset_name, method.LOSS, model_ae, step=method.ITER,
                                                  p=method.NORM )
        elif method.METHOD == 'add':
            data_batch['pc'] = attack_utils.add(data_batch, model, task, method.LOSS, dataset_name, step=method.ITER,
                                                alpha=method.EPS / 10, eps=method.EPS, p=method.NORM)
        elif method.METHOD == 'drop':
            data_batch['pc'] = attack_utils.drop(data_batch, model, task, method.LOSS, dataset_name, step=method.ITER)
        elif method.METHOD == 'apgd':
            if method.NORM == np.inf:
                attack = attack_utils.APGDAttack(model=model, n_iter=method.ITER, eps=method.EPS)
            else:
                attack = attack_utils.APGDAttack(model=model, n_iter=method.ITER, eps=method.EPS, norm='L2')
            _, data_batch['pc'] = attack.perturb(data_batch, dataset_name)
        elif method.METHOD == 'si_adv':
            pass

        inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
        time2 = time()
        with torch.no_grad():
            out = model(**inp)
            ### single test ####
            # TODO
            # out = model.classification_forward(**inp)
            if confusion:
                if dataset_name in ['modelnet40_diffusion','ScanObjectNN_diffusion']:
                    attacked.append(
                        (inp['pc'] * inp['scale'] + inp['shift']).squeeze().cpu())  ## special case for attack impl
                else:
                    attacked.append((inp['pc']).squeeze().cpu())
                pred.append(out['logit'].squeeze().cpu())
                ground.append(data_batch['label'].squeeze().cpu())
                recons.append(out['recons'].squeeze().cpu())

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    if not confusion:
        return perf.agg()
    else:
        pred = np.argmax(torch.cat(pred).numpy(), axis=1)
        ground = torch.cat(ground).numpy()
        attacked = torch.cat(attacked).numpy()

        recons = torch.cat(recons).numpy()
        return perf.agg(), pred, ground, attacked,recons


def validate(task, loader, model, dataset_name, adapt=None, confusion=False):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0
    loss = None
    out = None

    if task == 'sde_attack_cls':
        model.ae.decoder = RevGuidedDiffusion(model.ae.decoder)
    with torch.no_grad():
        bar = ProgressBar(max_value=len(loader))
        time5 = time()
        if confusion:
            pred = []
            ground = []
            recons = []
        for i, data_batch in enumerate(loader):
            if cmd_args.sor:
                data_batch['pc'] = sor_process(data_batch['pc'])
            time1 = time()
            inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
            time2 = time()

            if adapt.METHOD == 'bn':
                model = adapt_bn(inp, model, adapt)
            elif adapt.METHOD == 'tent':
                model = adapt_tent(inp, model, adapt)
            elif adapt.METHOD == 'ttac':
                pass


            if task not in ['ae', 'diffusion', 'advpc']:
                out = model(**inp)
                # out = model.classification_forward(**inp)
            if task in ['advpc']:
                loss = model.get_loss(inp['pc'])
            if hasattr(model, 'ae'):
                if model.ae.decoder_name == 'diffusion':
                    _, recon = model.ae(inp['pc'])
                    loss = model.ae.loss(recon * inp['scale'] + inp['shift'], inp['pc'] * inp['scale'] + inp['shift']) #
                else:
                    loss = model.ae.get_loss(inp['pc'])
            if confusion:
                if hasattr(model, 'ae'):
                    if model.ae.decoder_name == 'diffusion':
                        recons.append((recon * inp['scale'] + inp['shift']).squeeze().cpu())
                    else:
                        _, recon = model.ae(inp['pc'])
                        recons.append(recon.squeeze().cpu())
                if task not in ['ae', 'diffusion']:
                    pred.append(out['logit'].squeeze().cpu())
                ground.append(data_batch['label'].squeeze().cpu())
            time3 = time()
            perf.update(data_batch=data_batch, out=out, loss=loss)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    if not confusion:
        return perf.agg()
    else:
        if task not in ['ae', 'diffusion']:
            pred = np.argmax(torch.cat(pred).numpy(), axis=1)
        ground = torch.cat(ground).numpy()
        if hasattr(model, 'ae'):
            recons = torch.cat(recons).numpy()
        return perf.agg(), pred, ground, recons


def train(task, loader, model, optimizer, loss_name, dataset_name, cfg):
    model.train()

    def get_extra_param():
        return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    out = None
    time3 = time()

    for i, data_batch in enumerate(loader):
        time1 = time()


        if cfg.AUG.NAME == 'pgd':
            import attack_utils
            data_batch['pc'] = attack_utils.pgd(data_batch, model, task, loss_name, dataset_name)
            model.train()
        if task in ['ae']:
            for para in model.model.parameters():
                para.requires_grad = False
        if task in ['cls'] and hasattr(model, 'ae'): # 
            for para in model.ae.parameters():
                para.requires_grad = False

        inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
        if task not in ['ae', 'diffusion', 'advpc']:
            if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                out = model.classification_forward(inp['pc'], inp['scale'], inp['shift'])
            else:
                out = model(**inp)
        loss = get_loss(task, loss_name, data_batch, out, dataset_name, model)
        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        if loss.ne(loss).any():
            print("WARNING: avoiding step as nan in the loss")
        else:
            optimizer.zero_grad()
            loss.backward()
            bad_grad = False
            for x in model.parameters():
                if x.grad is not None:
                    if x.grad.ne(x.grad).any():
                        print("WARNING: nan in a gradient")
                        bad_grad = True
                        break
                    if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                        print("WARNING: inf in a gradient")
                        bad_grad = True
                        break

            if bad_grad:
                print("WARNING: avoiding step as bad gradient")
            else:
                if task == 'diffusion':
                    clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 10 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    return perf.agg(), perf.agg_loss()


def save_checkpoint(id, epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"./runs/{cfg.EXP.EXP_ID}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)


# def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
#
#     ####
#     print(f'Recovering model and checkpoint from {model_path}')
#     checkpoint = torch.load(model_path)
#     try:
#         model.load_state_dict(checkpoint['model_state'])
#     except:
#         if isinstance(model, nn.DataParallel):
#             model.module.load_state_dict(checkpoint['model_state'])
#         else:
#             model = nn.DataParallel(model)
#             model.load_state_dict(checkpoint['model_state'])
#             model = model.module
#
#     optimizer.load_state_dict(checkpoint['optimizer_state'])
#     # for backward compatibility with saved models
#     if 'lr_sched_state' in checkpoint:
#         lr_sched.load_state_dict(checkpoint['lr_sched_state'])
#         if checkpoint['bnm_sched_state'] is not None:
#             bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
#     else:
#         print("WARNING: lr scheduler and bnm scheduler states are not loaded.")
#
#     return model




def load_model(model, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    return model


def get_model(cfg):

    if cfg.EXP.MODEL_NAME == 'pointnet2':
        model = models.PointNet2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.PN2,
            ae=cfg.AE)
    elif cfg.EXP.MODEL_NAME == 'dgcnn':
        model = models.DGCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ae=cfg.AE)
    elif cfg.EXP.MODEL_NAME == 'pointnet':
        model = models.PointNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ae=cfg.AE)
    elif cfg.EXP.MODEL_NAME == 'pct':
        model = models.Pct(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ae=cfg.AE)
    elif cfg.EXP.MODEL_NAME == 'pointMLP':
        model = models.pointMLP(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ae=cfg.AE)
    elif cfg.EXP.MODEL_NAME == 'pointMLP2':
        model = models.pointMLP2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'curvenet':
        model = models.CurveNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ae=cfg.AE)

    elif cfg.EXP.MODEL_NAME == 'pointnetae':
        model = models.Autoencoder(
            encoder='pointnet',
            decoder='mlp')
    else:
        assert False

    return model


def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans', 'attack_cls', 'non_adaptive_attack_cls', 'sde_attack_cls']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    elif task in ['ae', 'diffusion', 'advpc']:
        assert metric_name in ['chamfer']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(optim_name, tr_arg, model):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=tr_arg.lr_decay_factor,
            patience=tr_arg.lr_reduce_patience,
            verbose=True,
            min_lr=tr_arg.lr_clip)
        bnm_sched = None
    elif optim_name == 'pct':
        pass
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            tr_arg.num_epochs,
            eta_min=tr_arg.learning_rate)
        bnm_sched = None
    elif optim_name == 'diffusion':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = get_linear_scheduler(
            optimizer,
            start_epoch=150,
            end_epoch=300,
            start_lr=tr_arg.learning_rate,
            end_lr=1e-4
        )
        bnm_sched = None
    elif optim_name == 'step':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 250], gamma=0.1)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg, resume=False, model_path="",model_name=''):
    loader_train = create_dataloader(split='train', cfg=cfg)
    loader_valid = create_dataloader(split='train', cfg=cfg)
    loader_test = create_dataloader(split='test', cfg=cfg)


    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)

    if resume:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path,model_name)
    if cfg.EXP.TASK in ['cls', 'cls_trans'] and cfg.AE and model_path != '':
        model = load_model(model, model_path)
    # else:
    #     assert model_path == ""

    log_dir = f"./runs/{cfg.EXP.EXP_ID}/debug"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    track_train = TrackTrain(early_stop_patience=cfg.TRAIN.early_stop)

    if cfg.EXP.TASK in ['cls'] and hasattr(model, 'ae') and model.ae.decoder_name == 'diffusion':
        _, _, ground, recons = validate('diffusion', loader_train, model, cfg.EXP.DATASET, adapt=cfg.ADAPT,
                                        confusion=True)
        loader_train.dataset.data = recons #
        loader_train.dataset.label = ground

    for epoch in range(cfg.TRAIN.num_epochs):
        print(f'Epoch {epoch}')

        print('Training..')
        train_perf, train_loss = train(cfg.EXP.TASK, loader_train, model, optimizer, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET,
                                       cfg)
        pprint.pprint(train_perf, width=80)
        tb.update('train', epoch, train_perf)

        if (not cfg.EXP_EXTRA.no_val) and epoch % cfg.EXP_EXTRA.val_eval_freq == 0:
            print('\nValidating..')
            val_perf = validate(cfg.EXP.TASK, loader_valid, model, cfg.EXP.DATASET, cfg.ADAPT)
            pprint.pprint(val_perf, width=80)
            tb.update('val', epoch, val_perf)
        else:
            val_perf = defaultdict(float)

        if (not cfg.EXP_EXTRA.no_test) and (epoch % cfg.EXP_EXTRA.test_eval_freq == 0):
            print('\nTesting..')
            test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
            pprint.pprint(test_perf, width=80)
            tb.update('test', epoch, test_perf)
        else:
            test_perf = defaultdict(float)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf(cfg.EXP.TASK, train_perf, cfg.EXP.METRIC),
            val_metric=get_metric_from_perf(cfg.EXP.TASK, val_perf, cfg.EXP.METRIC),
            test_metric=get_metric_from_perf(cfg.EXP.TASK, test_perf, cfg.EXP.METRIC))

        if (not cfg.EXP_EXTRA.no_val) and track_train.save_model(epoch_id=epoch, split='val'):
            print('Saving best model on the validation set')
            save_checkpoint('best_val', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_test) and track_train.save_model(epoch_id=epoch, split='test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_val) and track_train.early_stop(epoch_id=epoch):
            print(f"Early stopping at {epoch} as val acc did not improve for {cfg.TRAIN.early_stop} epochs.")
            break

        if (not (cfg.EXP_EXTRA.save_ckp == 0)) and (epoch % cfg.EXP_EXTRA.save_ckp == 0):
            save_checkpoint(f'{epoch}', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        if cfg.EXP.OPTIMIZER == 'vanilla':
            assert bnm_sched is None
            lr_sched.step(train_loss)
        else:
            lr_sched.step()

    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

    print('\nTesting on the final model..')
    last_test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
    pprint.pprint(last_test_perf, width=80)

    tb.close()


def entry_test(cfg, test_or_valid, model_path="", confusion=False,model_name=''):
    split = "test" if test_or_valid else "valid"
    loader_test = create_dataloader(split=split, cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)
    model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path,model_name)
    model.eval()
    if confusion:
        test_perf, pred, ground, recons = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT,
                                                   confusion)
        # print(pred.shape, ground.shape)
        #### some hardcoding #######
        np.save('./output/' + cfg.EXP.EXP_ID + '_recon.npy', recons)
        np.save('./output/' + cfg.EXP.EXP_ID + '_pred.npy', pred)
        np.save('./output/' + cfg.EXP.EXP_ID + '_ground.npy', ground)
        #### #### #### #### #### ####
    else:
        test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT, confusion)
    pprint.pprint(test_perf, width=80)
    return test_perf


def entry_attack(cfg, model_path="", confusion=False,model_name=''):
    # split = "test" if test_or_valid else "valid"
    loader_test = create_dataloader(split='test', cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)
    model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path,model_name)
    model.eval()
    if confusion:
        test_perf, pred, ground, attacked,recons = attack(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, method=cfg.ATTACK,
                                                   confusion=confusion)
        #### some hardcoding #######
        np.save('./output/' + cfg.EXP.EXP_ID + '_attacked.npy', attacked)
        np.save('./output/' + cfg.EXP.EXP_ID + '_pred.npy', pred)
        np.save('./output/' + cfg.EXP.EXP_ID + '_ground.npy', ground)
        np.savez('./output/' + cfg.EXP.EXP_ID, test_pc=attacked, test_label=ground, pred=pred)
        np.save('./output/' + cfg.EXP.EXP_ID + '_recons.npy', recons)
        #### #### #### #### #### ####
    else:
        test_perf = attack(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, method=cfg.ATTACK, confusion=confusion)
    pprint.pprint(test_perf, width=80)
    return test_perf




parser = argparse.ArgumentParser()
parser.set_defaults(entry=lambda cmd_args: parser.print_help())
parser.add_argument('--entry', type=str, default="attack")
parser.add_argument('--exp-config', type=str, default=
                                                        'configs/scan/pct_truncate_diffusion_pgd_naive.yaml')
parser.add_argument('--model-path', type=str, default=
'')

parser.add_argument('--resume', action="store_true", default=False)  #

parser.add_argument('--output', type=str, default='output/debug.out',
                    help="path to output file")

parser.add_argument('--confusion', action="store_true", default=False,
                    help="whether to output confusion matrix data")


parser.add_argument("--DATALOADER_batch_size", type=int, default=-1, help="Batch size for the data loader")
parser.add_argument("--DATALOADER_num_workers", type=int, default=0, help="Number of workers for the data loader")
parser.add_argument("--load_model_name", type=str, default='', choice=['PCT','Curvenet','pointnet2',''],help='pretrained backbone to load')

cmd_args = parser.parse_args()

if __name__ == '__main__':

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)

        if cmd_args.DATALOADER_batch_size != -1:
            cfg.DATALOADER.batch_size = cmd_args.DATALOADER_batch_size
        if cmd_args.DATALOADER_num_workers != -1:
            cfg.DATALOADER.num_workers = cmd_args.DATALOADER_num_workers

        if cfg.EXP.EXP_ID == "":
            cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        entry_train(cfg, cmd_args.resume, cmd_args.model_path,cmd_args.load_model_name)

    elif cmd_args.entry in ["test", "valid", "attack"]:
        file_object = open(cmd_args.output, 'a')
        assert not cmd_args.exp_config == ""
        # assert not cmd_args.model_path == ""

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)
        if cfg.EXP.DATASET in ["modelnet40_c", "modelnet40_c_diffusion"]:
            cfg.DATALOADER.MODELNET40_C.corruption = cmd_args.corruption
            cfg.DATALOADER.MODELNET40_C.severity = cmd_args.severity
        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        if cmd_args.entry in ["test", "valid"]:
            test_or_valid = cmd_args.entry == "test"
            entry_test(cfg, test_or_valid, cmd_args.model_path, cmd_args.confusion,cmd_args.load_model_name)

        if cmd_args.entry in ["attack"]:
            # test_or_valid = cmd_args.entry == "test"
            entry_attack(cfg, cmd_args.model_path, cmd_args.confusion,cmd_args.load_model_name)

    else:
        assert False
