from __future__ import print_function
import argparse
import random
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# ----------------------------------
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_path)

from utils.prepare_dataset import prepare_scanobjectnn_test_data, prepare_scanobjectnn_train_data, create_dataloader
from thresholding.scanobject import Adapt_thres # 15 class

from utils.offline import offline




# ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--iters', default=6, type=int)
    ########################################################################
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--test_data', type=str,
                        default='scanobjectnn/make_data/pct/4changing/changing_inf_batch_attack.npy')
    parser.add_argument('--test_label', type=str,
                        default='scanobjectnn/make_data/pct/4changing/label.npy')

    parser.add_argument('--backbone', default='pct', type=str,choices=['pointnet2','pct','curvenet'])
    parser.add_argument('--is_ada_thres', default=True, type=bool)
    parser.add_argument('--is_fix', default=False, type=bool)



    args = parser.parse_args()

    print(args)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)



    ########### build and load model #################

    if args.backbone == 'pointnet2':
        from networks.test_pointnet2_scan import build_model, load_model
        mini_batch_length = 512#512
        fc_out = 1024
        q_1 = 511
        args.batch_size = 16
    elif args.backbone == 'pct':
        from networks.scan_pct import build_model, load_model
        mini_batch_length =2882 #2468
        fc_out = 256
        q_1 = 1280
        args.batch_size = 32

    elif args.backbone == 'curvenet':
        from networks.test_curvernet_scan import build_model, load_model
        mini_batch_length = 512
        fc_out = 512
        q_1 = 511
        args.batch_size = 32

    net, ext, classifier = build_model()

    load_model(net, args)

    ########### create dataset and dataloader #################
    te_dataset = prepare_scanobjectnn_test_data(args)  ###  adv_points
    tr_dataset = prepare_scanobjectnn_train_data(args)
    tr_extra_dataset = prepare_scanobjectnn_test_data(args, with_da=True, with_strong=True)

    te_dataloader = create_dataloader(te_dataset, args, shuffle=True, drop_last=False)
    tr_dataloader = create_dataloader(tr_dataset, args, True, True)

    ########### summary offline features #################
    ext_mean, ext_cov, ext_mean_categories, ext_cov_categories = offline(tr_dataloader, ext, num_classes=15)
    bias = ext_cov.max().item() / 300.
    print('ext covariance bias:', bias)
    template_ext_cov = torch.eye(fc_out).cuda() * bias

    ########### create optimizer #################
    # optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


    ########### test before TTT #################


    correct = []
    correct_adv = []
    correct_advpc = []
    correct_cw = []
    correct_pgd = []


    correct_clean = []
    for te_batch_idx, (te_inputs, te_labels) in enumerate(te_dataloader):
        net.eval()
        with torch.no_grad():
            outputs = net(te_inputs[0].cuda())
            _, predicted = outputs.max(1)
            correct.append(predicted.cpu().eq(te_labels))

            te_inputs_idx = te_inputs[1]
            advpc_idx = ((te_inputs_idx // 64) % 4) == 0
            cw_idx = ((te_inputs_idx // 64) % 4 )== 1
            pgd_idx = ((te_inputs_idx // 64) % 4) == 2
            clean_idx =((te_inputs_idx // 64) % 4 )== 3

            correct_advpc.append(predicted[advpc_idx].cpu().eq(te_labels[advpc_idx]))
            correct_cw.append(predicted[cw_idx].cpu().eq(te_labels[cw_idx]))
            correct_pgd.append(predicted[pgd_idx].cpu().eq(te_labels[pgd_idx]))
            correct_clean.append(predicted[clean_idx].cpu().eq(te_labels[clean_idx]))

           

    print('clean : {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
    print('advpc : {:.2f}'.format(torch.cat(correct_advpc).numpy().mean() * 100))
    print('cw : {:.2f}'.format(torch.cat(correct_cw).numpy().mean() * 100))
    print('pgd : {:.2f}'.format(torch.cat(correct_pgd).numpy().mean() * 100))
    print('Mix : {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))

    print('---------------')


    del correct ,correct_clean, correct_cw,correct_advpc,correct_pgd,correct_adv
    ###########  #################

    torch.cuda.empty_cache()
    ########### TTT #################
    class_num = 15


    sample_alpha = torch.ones(len(tr_extra_dataset), dtype=torch.float)

    ema_ext_total_mu = ext_mean.clone()
    ema_ext_total_cov = ext_cov.clone()

    ema_total_n = 0
    class_ema_length = 128  # Nclip
    loss_scale = 0.05
    # mini_batch_length = 2468

    mini_batch_indices = []  #

    correct = []
    correct_adv = []
    correct_advpc = []
    correct_cw = []
    correct_pgd = []
    correct_clean = []


    ada_threshold = Adapt_thres()

    tic = time.time()
    for te_batch_idx, (te_inputs, te_labels) in enumerate(te_dataloader):
        mini_batch_indices.extend(te_inputs[-1].tolist())
        mini_batch_indices = mini_batch_indices[-mini_batch_length:]
        print('mini_batch_length:', len(mini_batch_indices))
        try:
            del tr_extra_subset
            del tr_extra_dataloader
        except:
            pass

        tr_extra_subset = data.Subset(tr_extra_dataset, mini_batch_indices)
        tr_extra_dataloader = create_dataloader(tr_extra_subset, args, shuffle=True, drop_last=True)

        net.train()

        for iter_id in range(0, min(args.iters, int(len(mini_batch_indices) / 128) + 1) + 2):
            if iter_id > 1:
                sample_alpha = torch.where(sample_alpha < 1, sample_alpha + 0.2, torch.ones_like(sample_alpha))

            for batch_idx, (inputs, labels) in enumerate(tr_extra_dataloader):
                # optimizer.zero_grad()
                optimizer2.zero_grad()

                ####### feature alignment ###########
                loss = 0.
                inputs, inputs_strong_aug, indexes = inputs

                inputs_strong_aug = inputs_strong_aug.cuda().float()
                inputs = inputs.cuda()
                
                feat_ext = ext(inputs)
                # logit = classifier(feat_ext).cpu()

                if args.is_ada_thres:
                    # ada_threshold
                    predict_logit_strong = net(inputs_strong_aug)
                    predict_logit = net(inputs)
                    loss_st = ada_threshold.train_step(weak_logit=predict_logit, strong_logit=predict_logit_strong)
                    loss_st = 0.1 * loss_st
                    # loss += loss_st
                    # del loss_st
                
                if args.is_fix:
                    loss_st =0.
                    predict_logit_strong = net(inputs_strong_aug)
                    predict_logit = net(inputs)
                    softmax_logit = predict_logit.softmax(dim=1).cpu()
                    pro_u, pseudo_label_u = softmax_logit.max(dim=1)  #  without ema
                    pseudo_label_mask2 = (pro_u > 0.6).clone()
                    Lu = (F.cross_entropy(predict_logit_strong, pseudo_label_u.cuda(),
                                       reduction='none') * pseudo_label_mask2.cuda()).mean()
                    loss_st += 0.1 * Lu



                # Gaussian Distribution Alignment
                b = feat_ext.shape[0]
                ema_total_n += b
                # alpha =  1. / ema_total_n ##
                alpha = 1. / (1280) if ema_total_n > (1280) else 1. / ema_total_n
                delta = alpha * (feat_ext - ema_ext_total_mu).sum(dim=0)
                tmp_mu = ema_ext_total_mu + delta

                tmp_cov = ema_ext_total_cov + alpha * ((feat_ext - ema_ext_total_mu).t() @ (
                        feat_ext - ema_ext_total_mu) - b * ema_ext_total_cov) - delta[:, None] @ delta[None, :]

                with torch.no_grad():
                    ema_ext_total_mu = tmp_mu.detach()
                    ema_ext_total_cov = tmp_cov.detach()
                source_domain = torch.distributions.MultivariateNormal(ext_mean, ext_cov + template_ext_cov)
                # print("tmp_mu",tmp_mu)
                # print("tmp_cov",tmp_cov)
                target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
                loss += (torch.distributions.kl_divergence(source_domain,
                                                           target_domain) + torch.distributions.kl_divergence(
                    target_domain, source_domain)) * loss_scale



                loss.backward()
                if args.is_ada_thres or args.is_fix:
                    loss_st.backward()
                if iter_id > 1 and len(mini_batch_indices) > 256:
                    # optimizer.step()
                    if args.is_ada_thres or args.is_fix:
                        optimizer2.step()
                    # optimizer.zero_grad()
                    if args.is_ada_thres or args.is_fix:
                        optimizer2.zero_grad()
                del loss
                if args.is_ada_thres or args.is_fix:
                    del loss_st

        #### Test ####
        # del ada_threshold
        net.eval()
        with torch.no_grad():
            outputs = net(te_inputs[0].cuda())
            _, predicted = outputs.max(1)
            correct.append(predicted.cpu().eq(te_labels))

            te_inputs_idx = te_inputs[1]

            advpc_idx = ((te_inputs_idx // 64) % 4) == 0
            cw_idx = ((te_inputs_idx // 64) % 4 )== 1
            pgd_idx = ((te_inputs_idx // 64) % 4) == 2
            clean_idx =((te_inputs_idx // 64) % 4 )== 3
            
            correct_advpc.append(predicted[advpc_idx].cpu().eq(te_labels[advpc_idx]))
            correct_cw.append(predicted[cw_idx].cpu().eq(te_labels[cw_idx]))
            correct_pgd.append(predicted[pgd_idx].cpu().eq(te_labels[pgd_idx]))
            correct_clean.append(predicted[clean_idx].cpu().eq(te_labels[clean_idx]))

           

            print('real time clean : {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
            print('real time advpc : {:.2f}'.format(torch.cat(correct_advpc).numpy().mean() * 100))
            print('real time cw : {:.2f}'.format(torch.cat(correct_cw).numpy().mean() * 100))
            print('real time pgd : {:.2f}'.format(torch.cat(correct_pgd).numpy().mean() * 100))
            print('real time Mix : {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))
           

            

            print('---------------')
            net.train()

    print('test time traing result')
    print('clean : {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
    print(' advpc : {:.2f}'.format(torch.cat(correct_advpc).numpy().mean() * 100))
    print(' cw : {:.2f}'.format(torch.cat(correct_cw).numpy().mean() * 100))
    print(' pgd : {:.2f}'.format(torch.cat(correct_pgd).numpy().mean() * 100))
    print(' Mix : {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))



if __name__ == '__main__':
    main()