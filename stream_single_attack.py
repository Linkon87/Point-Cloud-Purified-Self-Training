from __future__ import print_function
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
# ----------------------------------
from utils.prepare_dataset import prepare_test_data, prepare_train_data, create_dataloader
from thresholding.our.adaptive import Adapt_thres

from utils.offline import offline
from  test_protocal import  single_attack

# ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='modelnet40')
    parser.add_argument('--dataroot', default='data')
    ########################################################################
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--iters', default=6, type=int)
    ########################################################################
    parser.add_argument('--resume', default='save/dgcnn_pretrain/diffusion', help='directory of pretrained model')
    parser.add_argument('--ckpt', default=None, type=int)
    ########################################################################
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--test_data', type=str,
                        default='make_data/non_adaptive_dgcnn_truncated_diffusion_cw_inf_1__recons.npy')
    parser.add_argument('--test_label', type=str,
                        default='make_data/non_adaptive_dgcnn_truncated_diffusion_cw_inf_1__ground.npy')
    
    parser.add_argument('--backbone', default='pointnet', type=str)



    args = parser.parse_args()

    print(args)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)


    ########### build and load model #################
    if args.backbone == 'dgcnn': 
        from utils.test_helpers import build_model,load_model
        mini_batch_length = 2468
        fc_out = 256
        q_1 = 1280
        args.batch_size = 32
    elif args.backbone == 'pointnet': 
        from networks.test_pointnet import build_model, load_model
        mini_batch_length = 2468
        fc_out = 256
        q_1 = 1280
        args.batch_size = 32
    elif args.backbone == 'pointnet2': 
        from networks.test_pointnet2 import build_model, load_model
        mini_batch_length = 512
        fc_out = 1024
        q_1 = 511
        args.batch_size = 16
    elif args.backbone == 'pct': 
        from networks.test_pct import build_model, load_model
        mini_batch_length = 2468
        fc_out = 256
        q_1 = 1280
        args.batch_size = 32
    elif args.backbone == 'pointmlp': 
        from networks.test_pointmlp import build_model, load_model
        mini_batch_length = 512
        fc_out = 1024
        q_1 = 511
        args.batch_size = 16
    elif args.backbone == 'curvenet': 
        from networks.test_curvernet import build_model, load_model
        mini_batch_length = 2468
        fc_out = 512
        q_1 = 1280
        args.batch_size = 32

    net, ext, classifier = build_model()

    load_model(net, args)

    ########### create dataset and dataloader #################
    te_dataset = prepare_test_data(args)  ###  adv_points
    tr_dataset = prepare_train_data(args)
    tr_extra_dataset = prepare_test_data(args, with_da=True, with_strong=True)

    te_dataloader = create_dataloader(te_dataset, args, shuffle=True,drop_last=False)
    tr_dataloader = create_dataloader(tr_dataset, args, True, True)

    ########### summary offline features #################
    ext_mean, ext_cov, ext_mean_categories, ext_cov_categories = offline(tr_dataloader, ext)
    bias = ext_cov.max().item() / 30.
    print('ext covariance bias:', bias)
    template_ext_cov = torch.eye(fc_out).cuda() * bias

    ########### create optimizer #################
    # optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)



    ########### test before TTT #################
    single_attack(te_dataloader,net)
    ###########  #################




    torch.cuda.empty_cache()
    ########### TTT #################
    class_num = 40


    sample_alpha = torch.ones(len(tr_extra_dataset), dtype=torch.float)

    ema_ext_total_mu = ext_mean.clone()
    ema_ext_total_cov = ext_cov.clone()

    ema_total_n = 0
    class_ema_length = 128  # Nclip
    loss_scale = 0.05

    mini_batch_indices = []  #

    correct = []
    correct_clean = []
    correct_adv = []

    adapt_thres = Adapt_thres()

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
        tr_extra_dataloader = create_dataloader(tr_extra_subset, args, shuffle=True, drop_last = True)

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


                # eq.4
                predict_logit_strong = net(inputs_strong_aug)
                predict_logit = net(inputs)
                loss_st = adapt_thres.train_step(weak_logit=predict_logit, strong_logit=predict_logit_strong)
                loss_st = 0.1 * loss_st
                # loss += loss_st
                # del loss_st


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
                target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
                loss += (torch.distributions.kl_divergence(source_domain,
                                                           target_domain) + torch.distributions.kl_divergence(
                    target_domain, source_domain)) * loss_scale


                loss.backward()
                loss_st.backward()
                if iter_id > 1 and len(mini_batch_indices) > 256:
                    # optimizer.step()
                    optimizer2.step()
                    # optimizer.zero_grad()
                    optimizer2.zero_grad()
                del loss
                del loss_st

        #### Test ####
        net.eval()
        with torch.no_grad():
            outputs = net(te_inputs[0].cuda())
            _, predicted = outputs.max(1)
            correct.append(predicted.cpu().eq(te_labels))

            te_inputs_idx = te_inputs[1]
            batch_number = te_inputs_idx // 16
            batch_number_mod2 = batch_number % 2
            correct_clean.append(predicted[batch_number_mod2 == 0].cpu().eq(te_labels[batch_number_mod2 == 0]))
            correct_adv.append(predicted[batch_number_mod2 == 1].cpu().eq(te_labels[batch_number_mod2 == 1]))

            print('real time mixed acc: {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))
            print('clean acc: {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
            print(' adv acc: {:.2f}'.format(torch.cat(correct_adv).numpy().mean() * 100))
            print('---------------')
            net.train()

    print('Test time training result: {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))
    print('clean result: {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
    print('adv result: {:.2f}'.format(torch.cat(correct_adv).numpy().mean() * 100))



if __name__ == '__main__':
    main()