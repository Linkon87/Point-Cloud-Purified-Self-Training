import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import time

def si_adv():
# TODO
    pass

def pgd(data_batch,model, task, loss_name, dataset_name, step= 7, eps=0.05, alpha=0.01, p=np.inf):
    model.eval()
    # keep data_og as original
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1,data.shape[1],data.shape[2])
        alpha = (alpha / scale).expand(-1,data.shape[1],data.shape[2])
    else:
        scale = torch.ones([1, 1]) 
        shift = torch.zeros([1, 3])
    if task in ['non_adaptive_attack_cls']:
        pass
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)
    adv_data=data.clone().cuda()
    # initialize random perturbation, use 0.05 by default 
    adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05) 
    adv_data.detach()
    adv_data_batch = {}
    batchsize = data.shape[0]

    for _ in range(step):
        adv_data.requires_grad=True
        adv_data_batch['pc'] = adv_data
        adv_data_batch['label'] = data_batch['label'].cuda()
        model.zero_grad()
        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data_batch['pc'],scale=scale, shift=shift)
            loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')
        else:
            if loss_name == 'chamfer': # ?
                assert model.ae.decoder_name != 'diffusion'
                loss = model.ae.get_loss(adv_data_batch['pc'])
            elif loss_name == 'feature':
                loss = model.ae.get_feature_loss(adv_data_batch['pc'],target)  
            else:
                out = model(adv_data_batch['pc'],scale=scale, shift=shift)
                loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')
        loss.backward()
        with torch.no_grad():
            if p==np.inf:
                adv_data = adv_data + alpha * adv_data.grad.sign()
                delta = adv_data-data_og 
                delta = torch.clamp(delta, -eps,eps) # inf
            else:
                adv_data = adv_data + alpha * adv_data.grad
                delta = adv_data-data_og
                normVal = torch.norm(delta.view(batchsize, -1), p, 1).view(batchsize, 1, 1) # l2
                ### eps[:,0,0] = eps[:,any,any] ###
                if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                    mask = normVal<=eps[:,0,0].view(batchsize, 1, 1)
                    scaling = eps[:,0,0].view(batchsize, 1, 1) / normVal
                else:
                    mask = normVal<=eps
                    scaling = eps / normVal
                scaling[mask] = 1
                delta = delta*scaling
            # print(delta)
            adv_data = (data+delta).detach_()
    # print('finishing one batch...')
    return adv_data.type(torch.FloatTensor)
    
def nattack(data_batch,model, task, loss_name, dataset_name, eps=0.05, alpha=0.001, iters=100, variance=0.01, samples=512):
    # Black box attack
    assert loss_name == 'cross_entropy'
    model.eval()
    data = data_batch['pc'].cuda()
    label = data_batch['label']
    final_adv = []
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1,data.shape[1],data.shape[2]).cuda()
        alpha = (alpha / scale).expand(-1,data.shape[1],data.shape[2]).cuda()
    else:
        scale = torch.ones([1, 1]) .cuda()
        shift = torch.zeros([1, 3]).cuda()
    with torch.no_grad():
        batch_size = data.shape[0]
        for b in range(batch_size):
            adv_data_batch = {}
            if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                adv_data_batch['scale'] = torch.ones_like(scale) * scale[b]
                adv_data_batch['shift'] = shift[b].reshape(1,1,3).expand(batch_size,-1,-1)
                eps_cur = eps[b].cuda()
                alpha_cur = alpha[b].cuda()
            else:
                adv_data_batch['scale'] = None
                adv_data_batch['shift'] = None
                eps_cur = eps
                alpha_cur = alpha
            adv_data_batch['label'] = torch.ones_like(label).cuda() * label[b]
            adv_data=torch.squeeze(data[b].clone())
            adv_data_og = adv_data.clone().cuda()
            adv_data=(adv_data+(torch.rand_like(adv_data)*eps_cur*2-eps_cur)).cuda()
            adv_data.detach()
            mu = torch.zeros_like(adv_data_og).cuda()

            for _ in range(iters):
                est_gradient = torch.zeros_like(adv_data)
                loss_all = []
                loss_sum = 0
                pert_all = []

                for j in range(samples // batch_size):
                    adv_data_repeat = adv_data.repeat([batch_size,1,1]).cuda()

                    pert = torch.normal(0.0,1.0,size=adv_data_repeat.shape).cuda()
                    mu_perts = mu.repeat([batch_size,1,1]) + pert * variance

                    #####
                    adv_data_repeat = torch.clamp(adv_data_repeat, -1 + 1e-6, 1 - 1e-6)
                    ######
                    arctanh_x = torch.atanh(adv_data_repeat)
                    delta = torch.tanh(arctanh_x + mu_perts) - adv_data_repeat
                    delta = torch.clamp(delta,-eps_cur,eps_cur)
                    adv_data_repeat_1 = adv_data_repeat + delta

                    out = model(adv_data_repeat_1,scale=adv_data_batch['scale'],shift=adv_data_batch['shift'])
                    loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce=False)
                    loss_sum += torch.sum(loss,0)
                    loss_all.append(loss)
                    pert_all.append(pert)
                
                loss_all = torch.cat(loss_all)
                pert_all = torch.cat(pert_all)
                normalize_loss = (loss_all - (loss_sum / samples))/(torch.std(loss_all) + 1e-7)
                est_gradient = normalize_loss.reshape(-1, 1, 1) * pert_all 
                est_gradient = torch.mean(est_gradient, dim = 0) / variance
                # est_gradient = ( - (loss_sum / samples) * (perts_sum / samples) + (losses_perts_sum / samples)) / ((torch.std(loss_all)+1e-7) * variance)
                
                mu = mu + alpha_cur * est_gradient.sign()

                #####
                adv_data = torch.clamp(adv_data, -1 + 1e-6, 1 - 1e-6)
                ######
                delta = torch.tanh(torch.atanh(adv_data) + mu)-adv_data_og
                delta = torch.clamp(delta,-eps_cur,eps_cur)
                adv_data = adv_data_og+delta

            final_adv.append(adv_data)
            print('finishing one example...')
        print('finishing one batch...')
        final_adv = torch.stack(final_adv)
    return final_adv.cpu()

def l2_nattack(data_batch, model, task, loss_name, dataset_name, eps=1.25, alpha=0.125, iters=100, variance=0.01, samples=512):
    assert loss_name == 'cross_entropy'
    model.eval()
    data = data_batch['pc'].cuda()
    label = data_batch['label']
    final_adv = []

    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1, data.shape[1], data.shape[2]).cuda()
        alpha = (alpha / scale).expand(-1, data.shape[1], data.shape[2]).cuda()
    else:
        scale = torch.ones([1, 1]).cuda()
        shift = torch.zeros([1, 3]).cuda()

    with torch.no_grad():
        batch_size = data.shape[0]

        for b in range(batch_size):
            adv_data_batch = {}

            if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                adv_data_batch['scale'] = torch.ones_like(scale) * scale[b]
                adv_data_batch['shift'] = shift[b].reshape(1, 1, 3).expand(batch_size, -1, -1)
                eps_cur = eps[b].cuda()
                alpha_cur = alpha[b].cuda()
            else:
                adv_data_batch['scale'] = None
                adv_data_batch['shift'] = None
                eps_cur = eps.cuda()
                alpha_cur = alpha.cuda()

            adv_data_batch['label'] = torch.ones_like(label).cuda() * label[b]
            adv_data = torch.squeeze(data[b].clone())
            adv_data_og = adv_data.clone().cuda()

            # Perturbation initialization
            adv_data = (adv_data + torch.randn_like(adv_data) * eps_cur).cuda()

            mu = torch.zeros_like(adv_data_og).cuda()

            for _ in range(iters):
                est_gradient = torch.zeros_like(adv_data)
                loss_all = []
                loss_sum = 0
                pert_all = []

                for j in range(samples // batch_size):
                    adv_data_repeat = adv_data.repeat([batch_size, 1, 1]).cuda()
                    pert = torch.randn_like(adv_data_repeat).cuda()
                    mu_perts = mu.repeat([batch_size, 1, 1]) + pert * variance
                    
                    #####
                    adv_data_repeat = torch.clamp(adv_data_repeat, -1 + 1e-6, 1 - 1e-6)
                    ######
                    arctanh_x = torch.atanh(adv_data_repeat)
                    delta = torch.tanh(arctanh_x + mu_perts) - adv_data_repeat
                    delta = delta * (eps_cur / torch.norm(delta, dim=(1, 2), keepdim=True))
                    adv_data_repeat_1 = adv_data_repeat + delta

                    out = model(adv_data_repeat_1, scale=adv_data_batch['scale'], shift=adv_data_batch['shift'])
                    loss = F.cross_entropy(out['logit'], adv_data_batch['label'], reduce=False)
                    loss_sum += torch.sum(loss, 0)
                    loss_all.append(loss)
                    pert_all.append(pert)

                loss_all = torch.cat(loss_all)
                pert_all = torch.cat(pert_all)
                normalize_loss = (loss_all - (loss_sum / samples)) / (torch.std(loss_all) + 1e-7)
                est_gradient = normalize_loss.reshape(-1, 1, 1) * pert_all
                est_gradient = torch.mean(est_gradient, dim=0) / variance

                mu = mu + alpha_cur * est_gradient
                #####
                adv_data = torch.clamp(adv_data, -1 + 1e-6, 1 - 1e-6)
                ######
                # Perturbation update
                delta = torch.tanh(torch.atanh(adv_data) + mu) - adv_data_og
                delta = delta * (eps_cur / torch.norm(delta, dim=(-2, -1), keepdim=True))
                adv_data = adv_data_og + delta

            final_adv.append(adv_data)
            print('finishing one example...')

        print('finishing one batch...')
        final_adv = torch.stack(final_adv)

    return final_adv.cpu()


def spsa(data_batch, model, task, loss_name, dataset_name, eps=0.05, alpha=0.005, scaling=0.005, iters=100,
         samples=512):
    assert loss_name == 'cross_entropy'
    model.eval()
    data = data_batch['pc']
    label = data_batch['label']
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1, data.shape[1], data.shape[2])
        alpha = (alpha / scale).expand(-1, data.shape[1], data.shape[2])
    else:
        scale = None
        shift = None
    final_adv = []
    with torch.no_grad():
        batch_size = data.shape[0]
        for b in range(batch_size):
            adv_data_batch = {}

            adv_data_batch['label'] = torch.ones_like(label).cuda() * label[b]
            if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                adv_data_batch['scale'] = torch.ones_like(scale) * scale[b]
                adv_data_batch['shift'] = shift[b].reshape(1, 1, 3).expand(batch_size, -1, -1)
                eps_cur = eps[b]
                alpha_cur = alpha[b]
            else:
                adv_data_batch['scale'] = None
                adv_data_batch['shift'] = None
                eps_cur = eps
                alpha_cur = alpha
            adv_data = torch.squeeze(data[b].clone()).cuda()
            adv_data_og = adv_data.clone().cuda()
            adv_data = adv_data + (torch.rand_like(adv_data) * eps_cur * 2 - eps_cur)
            adv_data.detach()
            for _ in range(iters):
                est_gradient = torch.zeros_like(adv_data)
                for j in range(samples // batch_size):
                    adv_data_repeat = adv_data.repeat([batch_size, 1, 1])

                    pert = torch.rand_like(adv_data_repeat).cuda() - 0.5

                    adv_data_repeat_1 = adv_data_repeat + pert.sign() * scaling
                    out = model(adv_data_repeat_1, scale=adv_data_batch['scale'], shift=adv_data_batch['shift'])
                    loss_1 = F.cross_entropy(out['logit'], adv_data_batch['label'], reduce=False)

                    adv_data_repeat_2 = adv_data_repeat - pert.sign() * scaling
                    out_2 = model(adv_data_repeat_2, scale=adv_data_batch['scale'], shift=adv_data_batch['shift'])
                    loss_2 = F.cross_entropy(out_2['logit'], adv_data_batch['label'], reduce=False)

                    sub_loss = torch.reshape(loss_1 - loss_2, [-1, 1, 1]).repeat(
                        [1, adv_data_repeat.shape[1], adv_data_repeat.shape[2]])
                    est_gradient += torch.sum(sub_loss / (2 * scaling * pert.sign()), 0)

                est_gradient = est_gradient / samples
                adv_data = adv_data + alpha_cur * est_gradient.sign()
                delta = adv_data - adv_data_og
                delta = torch.clamp(delta, -eps_cur, eps_cur)
                adv_data = adv_data_og + delta

            final_adv.append(adv_data)
            print('finishing one example...')
        print('finishing one batch...')
        final_adv = torch.stack(final_adv)
    return final_adv.cpu()

def l2_spsa(data_batch, model, task, loss_name, dataset_name, eps=0.05, alpha=0.005, scaling=0.005, iters=100,
            samples=512):
    assert loss_name == 'cross_entropy'
    model.eval()
    data = data_batch['pc']
    label = data_batch['label']

    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1, data.shape[1], data.shape[2])
        alpha = (alpha / scale).expand(-1, data.shape[1], data.shape[2])
    else:
        scale = None
        shift = None

    final_adv = []

    with torch.no_grad():
        batch_size = data.shape[0]

        for b in range(batch_size):
            adv_data_batch = {}
            adv_data_batch['label'] = torch.ones_like(label).cuda() * label[b]

            if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                adv_data_batch['scale'] = torch.ones_like(scale) * scale[b]
                adv_data_batch['shift'] = shift[b].reshape(1, 1, 3).expand(batch_size, -1, -1)
                eps_cur = eps[b]
                alpha_cur = alpha[b]
            else:
                adv_data_batch['scale'] = None
                adv_data_batch['shift'] = None
                eps_cur = eps
                alpha_cur = alpha

            adv_data = torch.squeeze(data[b].clone()).cuda()
            adv_data_og = adv_data.clone().cuda()
            adv_data = adv_data + (torch.rand_like(adv_data) * eps_cur * 2 - eps_cur)
            adv_data.detach()

            for _ in range(iters):
                est_gradient = torch.zeros_like(adv_data)

                for j in range(samples // batch_size):
                    adv_data_repeat = adv_data.repeat([batch_size, 1, 1])
                    pert = torch.rand_like(adv_data_repeat).cuda() - 0.5

                    adv_data_repeat_1 = adv_data_repeat + pert.sign() * scaling
                    if task in ['non_adaptive_attack_cls']:
                        out = model.classification_forward(adv_data_repeat_1, scale=adv_data_batch['scale'],
                                                           shift=adv_data_batch['shift'])
                    else:
                        out = model(adv_data_repeat_1, scale=adv_data_batch['scale'], shift=adv_data_batch['shift'])
                    loss_1 = F.cross_entropy(out['logit'], adv_data_batch['label'], reduce=False)

                    adv_data_repeat_2 = adv_data_repeat - pert.sign() * scaling
                    if task in ['non_adaptive_attack_cls']:
                        out_2 = model.classification_forward(adv_data_repeat_2, scale=adv_data_batch['scale'],
                                                             shift=adv_data_batch['shift'])
                    else:
                        out_2 = model(adv_data_repeat_2, scale=adv_data_batch['scale'], shift=adv_data_batch['shift'])
                    loss_2 = F.cross_entropy(out_2['logit'], adv_data_batch['label'], reduce=False)

                    sub_loss = torch.reshape(loss_1 - loss_2, [-1, 1, 1]).repeat(
                        [1, adv_data_repeat.shape[1], adv_data_repeat.shape[2]])
                    est_gradient += torch.sum(sub_loss / (2 * scaling * pert.sign()), 0)

                adv_data = adv_data + alpha_cur * est_gradient.sign()
                delta = adv_data - adv_data_og
                delta_norm = torch.norm(delta, dim=(-2, -1), keepdim=True)
                scaling = torch.min(torch.ones_like(delta_norm), eps_cur / delta_norm)
                adv_data = adv_data_og + delta * scaling

            final_adv.append(adv_data)
            print('finishing one example...')

        print('finishing one batch...')
        final_adv = torch.stack(final_adv)

    return final_adv.cpu()


def eot(data_batch,model, task, loss_name, dataset_name, step= 7, eps=0.05, alpha=0.01):
    #TODO: 
    pass

def nes():
    pass


def l2_cw(data_batch, task, model, loss_name,dataset_name, c=1, step=200, alpha=0.01):

    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
        i, _ = torch.max((1-one_hot_labels)*outputs - one_hot_labels*10000, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit
        # print(i,j)
        return torch.clamp((j-i), min=0)

    def tanh_space(x):
        return torch.tanh(x)

    def inverse_tanh_space(x):
        return torch.atanh(x)

    model.eval()
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()

    if task in ['non_adaptive_attack_cls']:
        pass

    if dataset_name == ('modelnet40_diffusion' or 'ScanObjectNN_diffusion'):
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
    else:
        scale = torch.ones([1, 1]).cuda()
        shift = torch.zeros([1, 3]).cuda()
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)

    prev_cost = 1e10
    dim = len(data.shape)
    MSELoss = nn.MSELoss(reduction='none')
    Flatten = nn.Flatten()

    if dataset_name == 'ScanObjectNN_diffusion':
        w = inverse_tanh_space(torch.clamp((data * scale + shift),-1,1)).detach()
    else:
        if loss_name == 'cross_entropy':
            w = inverse_tanh_space(data * scale + shift).detach()
            # w = inverse_tanh_space(torch.clamp((data * scale + shift),-1,1)).detach()
        else:
            w = inverse_tanh_space(torch.clamp(data + 0.05 * (torch.rand_like(data)*2-1),-1,1)).detach()
    w.requires_grad = True

    best_adv = data.clone().detach()
    best_L2 = 1e10*torch.ones((len(data))).cuda()
    import torch.optim as optim
    optimizer = optim.Adam([w], lr=alpha)

    for step_i in range(step):

        adv_data = (tanh_space(w) - shift) / scale
        current_L2 = MSELoss(Flatten(adv_data),
                        Flatten(data_og)).sum(dim=1)
        L2_loss = current_L2.sum()

        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data,scale=scale, shift=shift)
            f_loss = f(out['logit'],data_batch['label']).sum()
        else:
            if loss_name == 'chamfer':
                assert model.ae.decoder_name != 'diffusion'
                f_loss = - model.ae.get_loss(adv_data)
            elif loss_name == 'feature':
                f_loss = - model.ae.get_feature_loss(adv_data,target)
            elif loss_name == 'cross_entropy':
                out = model(adv_data,scale=scale, shift=shift)
                f_loss = f(out['logit'],data_batch['label']).sum()
            else:
                assert False

        cost =L2_loss + c * f_loss
        # print(cost, L2_loss, f_loss)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        mask = best_L2 > current_L2.detach()
        mask = mask.view([-1]+[1]*(dim-1))
        best_adv = mask*adv_data.detach() + (~mask)*best_adv

        if step_i % max(step//10,1) == 0:
            if cost.item() > prev_cost:
                return best_adv
            prev_cost = cost.item()
    return best_adv

def new_inf_cw(data_batch, task, model, loss_name, dataset_name, c=1, step=200, alpha=0.01):
    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
        i, _ = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 10000, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
        return torch.clamp((j - i), min=0)

    def tanh_space(x):
        return torch.tanh(x)

    def inverse_tanh_space(x):
        return torch.atanh(x)

    model.eval()
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()

    if task in ['non_adaptive_attack_cls']:
        pass

    if dataset_name == ('modelnet40_diffusion' or 'ScanObjectNN_diffusion'):
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
    else:
        scale = torch.ones([1, 1]).cuda()
        shift = torch.zeros([1, 3]).cuda()
    if loss_name == 'feature':
        with torch.no_grad():
            target = model.ae.encode(data)

    prev_cost = 1e10
    dim = len(data.shape)
    MSELoss = nn.MSELoss(reduction='none')
    Flatten = nn.Flatten()
    if loss_name == 'cross_entropy':
        w = inverse_tanh_space(data * scale + shift).detach()
    else:
        w = inverse_tanh_space(torch.clamp(data + 0.05 * (torch.rand_like(data) * 2 - 1), -1, 1)).detach()
    w.requires_grad = True

    best_adv = data.clone().detach()
    best_inf = 1e10 * torch.ones((len(data))).cuda()
    import torch.optim as optim
    optimizer = optim.Adam([w], lr=alpha)

    for step_i in range(step):

        adv_data = (tanh_space(w) - shift) / scale
        current_Linf = torch.max(torch.abs(Flatten(adv_data) - Flatten(data_og)), dim=1)[0]
        Linf_loss = current_Linf.sum()

        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data, scale=scale, shift=shift)
            f_loss = f(out['logit'], data_batch['label']).sum()
        else:
            if loss_name == 'chamfer':
                assert model.ae.decoder_name != 'diffusion'
                f_loss = - model.ae.get_loss(adv_data)
            elif loss_name == 'feature':
                f_loss = - model.ae.get_feature_loss(adv_data, target)
            elif loss_name == 'cross_entropy':
                out = model(adv_data, scale=scale, shift=shift)
                f_loss = f(out['logit'], data_batch['label']).sum()
            else:
                assert False

        cost = Linf_loss + c * f_loss
        # print(cost, L2_loss, f_loss)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        mask = best_inf > current_Linf.detach()
        mask = mask.view([-1] + [1] * (dim - 1))
        best_adv = mask * adv_data.detach() + (~mask) * best_adv

        if step_i % max(step // 10, 1) == 0:
            if cost.item() > prev_cost:
                return best_adv
            prev_cost = cost.item()
    return best_adv


def cw(data_batch, model, task, loss_name, dataset_name, c=10, step=500, alpha=0.01, p=np.inf):
    if p == 2.:
        return l2_cw(data_batch,task, model, loss_name,dataset_name, c, step, alpha)
    if p == np.inf :
        return new_inf_cw(data_batch,task, model, loss_name,dataset_name, c, step, alpha)



class ChamferDist(nn.Module):

    def __init__(self, method='adv2ori'):
        super(ChamferDist, self).__init__()

        self.method = method

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))  # [B, K, K]
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def chamfer(self,preds, gts):
        P = self.batch_pairwise_dist(gts, preds)  # [B, N2, N1]
        mins, _ = torch.min(P, 1)  # [B, N1], find preds' nearest points in gts
        loss1 = torch.mean(mins, dim=1)  # [B]
        mins, _ = torch.min(P, 2)  # [B, N2], find gts' nearest points in preds
        loss2 = torch.mean(mins, dim=1)  # [B]
        return loss1, loss2

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,))
        loss1, loss2 = self.chamfer(adv_pc, ori_pc)  # [B], adv2ori, ori2adv
        if self.method == 'adv2ori':
            loss = loss1
        elif self.method == 'ori2adv':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.
        weights = weights.float().cuda()
        loss = loss.cuda() * weights
        if batch_avg:
            return loss.mean()
        return loss


class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        # build kNN graph
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
        # assert dist.min().item() >= -1e-6
        # assert dist.min().item() >= -5e-6 
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]).\
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss.cuda() * weights
        if batch_avg:
            return loss.mean()
        return loss
class ChamferkNNDist(nn.Module):

    def __init__(self, chamfer_method='adv2ori',
                 knn_k=5, knn_alpha=1.05,
                 chamfer_weight=5., knn_weight=3.):
        super(ChamferkNNDist, self).__init__()

        self.chamfer_dist = ChamferDist(method=chamfer_method)
        self.knn_dist = KNNDist(k=knn_k, alpha=knn_alpha)
        self.w1 = chamfer_weight
        self.w2 = knn_weight

    def forward(self, adv_pc, ori_pc,
                weights=None, batch_avg=True):
        chamfer_loss = self.chamfer_dist(
            adv_pc, ori_pc, weights=weights, batch_avg=batch_avg)
        knn_loss = self.knn_dist(
            adv_pc, weights=weights, batch_avg=batch_avg)
        loss = chamfer_loss * self.w1 + knn_loss * self.w2
        return loss


def knn(data_batch, model,task, loss_name, dataset_name, step=200, alpha=0.01):
    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
        i, _ = torch.max((1-one_hot_labels)*outputs - one_hot_labels*10000, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        return torch.clamp((j-i), min=0)

    def tanh_space(x):
        return torch.tanh(x)

    def inverse_tanh_space(x):
        return torch.atanh(x)

    model.eval()
    data = data_batch['pc'].cuda()
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
    else:
        scale = torch.ones([1, 1]).cuda()
        shift = torch.zeros([1, 3]).cuda()
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)

    prev_cost = 1e10
    dim = len(data.shape)

    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=10., knn_weight=6.)
    
    if dataset_name == 'ScanObjectNN_diffusion':
        w = inverse_tanh_space(torch.clamp((data * scale + shift),-1,1)).detach()
    else:
        if loss_name == 'cross_entropy':
            w = inverse_tanh_space(data * scale + shift).detach()
            # w = inverse_tanh_space(torch.clamp((data * scale + shift),-1,1)).detach()
        else:
            w = inverse_tanh_space(torch.clamp(data + 0.05 * (torch.rand_like(data)*2-1),-1,1)).detach()
            
    w.requires_grad = True

    best_adv = data.clone().detach()
    best_dist = 1e10*torch.ones((len(data))).cuda()
    import torch.optim as optim
    optimizer = optim.Adam([w], lr=alpha)

    for step_i in range(step):

        adv_data = (tanh_space(w) - shift) / scale
        current_dis = dist_func(adv_data,data)
        dist_loss = current_dis.mean() * 1024

        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data,scale=scale, shift=shift)
            f_loss = f(out['logit'],data_batch['label']).mean()
        else:
            if loss_name == 'chamfer':
                assert model.ae.decoder_name != 'diffusion'
                f_loss = - model.ae.get_loss(adv_data)
            elif loss_name == 'feature':
                f_loss = - model.ae.get_feature_loss(adv_data,target)
            elif loss_name == 'cross_entropy':
                out = model(adv_data,scale=scale, shift=shift)
                f_loss = f(out['logit'],data_batch['label']).mean()
            else:
                assert False

        cost =dist_loss + f_loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        mask = best_dist > dist_loss.detach()
        mask = mask.view([-1]+[1]*(dim-1))
        best_adv = mask*adv_data.detach() + (~mask)*best_adv 
        if step_i % max(step//10,1) == 0:
            if cost.item() > prev_cost:
                return best_adv.type(torch.FloatTensor)
            prev_cost = cost.item()
    
    return best_adv.type(torch.FloatTensor)
    

def add(data_batch,model, task, loss_name, dataset_name, step= 7, eps=0.05, alpha=0.01, p=np.inf, num_points=200):
    model.eval()
    # keep data_og as original
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()

    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1,num_points,data.shape[2])
        alpha = (alpha / scale).expand(-1,num_points,data.shape[2])
    else:
        scale = torch.ones([1, 1]) 
        shift = torch.zeros([1, 3])
    if task in ['non_adaptive_attack_cls']:
        pass
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)
    # initialize random perturbation
    indices = torch.Tensor(np.random.choice(data.shape[1], num_points, replace=False)).long()
    adv_data_og = data.clone()[:,indices,:]
    adv_data = adv_data_og+(torch.rand_like(adv_data_og)*eps*2-eps)
    adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05)
    
    adv_data.detach()
    adv_data_batch = {}
    batchsize = data.shape[0]

    for _ in range(step):
        adv_data.requires_grad=True
        
        adv_data_batch['pc'] = torch.cat([data,adv_data],dim=-2)
        adv_data_batch['label'] = data_batch['label'].cuda()
        model.zero_grad()
        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data_batch['pc'],scale=scale, shift=shift)
            loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')
        else:
            if loss_name == 'chamfer':
                assert model.ae.decoder_name != 'diffusion'
                loss = model.ae.get_loss(adv_data_batch['pc'])
            elif loss_name == 'feature':
                loss = model.ae.get_feature_loss(adv_data_batch['pc'],target)
            elif loss_name == 'margin':
                #TODO
                pass
            else:
                out = model(adv_data_batch['pc'],scale=scale, shift=shift)
                loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')
        loss.backward()
        with torch.no_grad():
            if p==np.inf:
                adv_data = adv_data + alpha * adv_data.grad.sign()
                delta = adv_data-adv_data_og
                delta = torch.clamp(delta, -eps,eps)
            else:
                adv_data = adv_data + alpha * adv_data.grad
                delta = adv_data-adv_data_og
                normVal = torch.norm(delta.view(batchsize, -1), p, 1).view(batchsize, 1, 1)
                ### eps[:,0,0] = eps[:,any,any] ###
                if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                    mask = normVal<=eps[:,0,0].view(batchsize, 1, 1)
                    scaling = eps[:,0,0].view(batchsize, 1, 1) / normVal
                else:
                    mask = normVal<=eps
                    scaling = eps / normVal
                scaling[mask] = 1
                delta = delta*scaling
            # print(delta)
            adv_data = (adv_data_og+delta).detach_()
    # print('finishing one batch...')
    return adv_data_batch['pc'].type(torch.FloatTensor)

def drop(data_batch,model, task, loss_name, dataset_name, step= 10, num_points=200):
    model.eval()
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
    else:
        scale = torch.ones([1, 1])
        shift = torch.zeros([1, 3])
    if task in ['non_adaptive_attack_cls']:
        pass
        # _,data = model.ae(data)
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)
    adv_data=data.clone().cuda()
    adv_data.detach()
    adv_data_batch = {}
    alpha = num_points // step

    for i in range(step):
        adv_data.requires_grad=True
        adv_data_batch['pc'] = adv_data
        adv_data_batch['label'] = data_batch['label'].cuda()
        model.zero_grad()
        if task in ['non_adaptive_attack_cls']:
            out = model.classification_forward(adv_data_batch['pc'],scale=scale, shift=shift)
            loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')
        else:
            if loss_name == 'chamfer':
                assert model.ae.decoder_name != 'diffusion'
                loss = model.ae.get_loss(adv_data_batch['pc'])
            elif loss_name == 'feature':
                loss = model.ae.get_feature_loss(adv_data_batch['pc'],target)
            elif loss_name == 'margin':
                #TODO
                pass
            else:
                out = model(adv_data_batch['pc'],scale=scale, shift=shift)
                loss = F.cross_entropy(out['logit'], adv_data_batch['label'],reduce='mean')    
        loss.backward()

        with torch.no_grad():
            sphere_core,_ = torch.median(adv_data, dim=1, keepdim=True)
            sphere_r = torch.sqrt(torch.sum(torch.square(adv_data - sphere_core), dim=2))
            sphere_axis = adv_data - sphere_core

            sphere_map = - torch.mul(torch.sum(torch.mul(adv_data.grad, sphere_axis), dim=2), torch.pow(sphere_r, 2))
            _,indice = torch.topk(sphere_map, k=adv_data.shape[1] - alpha, dim=-1, largest=False)
            tmp = torch.zeros((adv_data.shape[0], adv_data.shape[1] - alpha, 3))
            for i in range(adv_data.shape[0]):
                tmp[i] = adv_data[i][indice[i],:]
            adv_data = tmp.clone()
            
    return adv_data.type(torch.FloatTensor)

def advpc(data_batch,model, dataset_name, loss_name,model_ae, step=7, eps=0.05, alpha=0.01, p=np.inf,lamb = 0.75):
    model.eval()
    model_ae.eval()

    alpha = eps/10

    # keep data_og as original
    data_og = data_batch['pc'].cuda()
    data = data_og.clone()
    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
        scale = data_batch['scale'].cuda()
        shift = data_batch['shift'].cuda()
        eps = (eps / scale).expand(-1,data.shape[1],data.shape[2])
        alpha = (alpha / scale).expand(-1,data.shape[1],data.shape[2])
    else:
        scale = torch.ones([1, 1]).cuda() 
        shift = torch.zeros([1, 3]).cuda()
        # _,data = model.ae(data)
    if loss_name == 'feature':
        with torch.no_grad():
            target =  model.ae.encode(data)
    adv_data=data.clone().cuda()
    # initialize random perturbation
    adv_data=adv_data+(torch.rand_like(adv_data)*0.05*2-0.05)
    adv_data.detach()
    adv_data_batch = {}
    batchsize = data.shape[0]

    for _ in range(step):
        adv_data.requires_grad=True
        adv_data_batch['pc'] = adv_data
        adv_data_batch['label'] = data_batch['label'].cuda()
        model.zero_grad()
        model_ae.zero_grad()
        _, recon = model_ae(adv_data_batch['pc'])
        out_1 = model.classification_forward(recon, scale=scale, shift=shift)
        loss_1 = F.cross_entropy(out_1['logit'], adv_data_batch['label'], reduce='mean')
        if loss_name == 'feature':
            loss_2 = model.ae.get_feature_loss(adv_data_batch['pc'], target)
        else:
            out_2 = model.classification_forward(adv_data_batch['pc'],scale=scale, shift=shift)
            loss_2 = F.cross_entropy(out_2['logit'], adv_data_batch['label'],reduce='mean')

        loss = (1 - lamb) * loss_1 + lamb * loss_2
        loss.backward()
        with torch.no_grad():
            if p==np.inf:
                adv_data = adv_data + alpha * adv_data.grad.sign()
                delta = adv_data-data_og
                delta = torch.clamp(delta, -eps,eps)
            else:
                adv_data = adv_data + alpha * adv_data.grad
                delta = adv_data-data_og
                normVal = torch.norm(delta.view(batchsize, -1), p, 1).view(batchsize, 1, 1)
                ### eps[:,0,0] = eps[:,any,any] ###
                if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                    mask = normVal<=eps[:,0,0].view(batchsize, 1, 1)
                    scaling = eps[:,0,0].view(batchsize, 1, 1) / normVal
                else:
                    mask = normVal<=eps
                    scaling = eps / normVal
                scaling[mask] = 1
                delta = delta*scaling
            adv_data = (data+delta).detach_()
    return adv_data.type(torch.FloatTensor)




class APGDAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=True,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter #
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device

    def normalize(self, x):
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims)) 
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def attack_single_run(self, x_in, y_in,dataset_name,scale=None,shift=None):
        x = x_in.clone() if len(x_in.shape) == 3 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
            scale = scale.cuda()
            shift = shift.cuda()
            self.eps = (self.eps / scale).expand(-1,x.shape[1],x.shape[2])
            # alpha = (alpha / scale).expand(-1,x.shape[1],x.shape[2])
        else:
            scale = torch.ones([1, 1]).cuda() 
            shift = torch.zeros([1, 3]).cuda()
        
        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach() 
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2), keepdim=True).sqrt() + 1e-12)
        # x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter): # eot
            with torch.enable_grad():
                if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                    logits = self.model.classification_forward(x_adv,scale=scale, shift=shift)['logit']
                else:
                    logits = self.model(x_adv,scale=scale, shift=shift)['logit'] # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                    
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps)
                    x_adv_1 = torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    # x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                    #     ) * torch.min(self.eps * torch.ones_like(x).detach(),
                    #     self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x + self.normalize(x_adv_1 - x)
                                                             
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    # x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                    #     ) * torch.min(self.eps * torch.ones_like(x).detach(),
                    #     self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x + self.normalize(x_adv_1 - x)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                        logits = self.model.classification_forward(x_adv,scale=scale, shift=shift)['logit']
                    else:
                        logits = self.model(x_adv,scale=scale, shift=shift)['logit'] # 
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
              
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, data_batch, dataset_name, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x_in = data_batch['pc']
        y_in = data_batch['label']
        x = x_in.clone().cuda() if len(x_in.shape) == 3 else x_in.clone().unsqueeze(0)
        y = y_in.clone().cuda() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
            acc = self.model.classification_forward(x,scale=data_batch['scale'].cuda(),shift=data_batch['shift'].cuda())['logit'].max(1)[1] == y
        else:
            acc = self.model(x)['logit'].max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()  
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                            x_to_fool, y_to_fool, scale_to_fool, shift_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone(), data_batch['scale'][ind_to_fool].clone(),data_batch['shift'][ind_to_fool].clone()
                        else:
                            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                            best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool,dataset_name,scale_to_fool, shift_to_fool)
                        else:
                            best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool,dataset_name)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
            
            return acc, adv.type(torch.FloatTensor)
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                if dataset_name == 'modelnet40_diffusion' or 'ScanObjectNN_diffusion':
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool,dataset_name,data_batch['scale'],data_batch['shift'])
                else:
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool,dataset_name)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best.type(torch.FloatTensor)