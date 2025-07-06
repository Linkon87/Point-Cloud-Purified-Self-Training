import contextlib
import torch
import torch.nn.functional as F

from .mask import ThresholdingHook


class Adapt_thres:

    def __init__(self):
        self.thresholding_hook = ThresholdingHook()

    def train_step(self, weak_logit, strong_logit):

        with contextlib.nullcontext():

            logits_x_ulb_s = strong_logit
            logits_x_ulb_w = weak_logit
            pass

            # calculate mask
            mask, global_threshold = self.thresholding_hook.masking(logits_x_ulb=logits_x_ulb_w)


            pseudo_label = torch.argmax(logits_x_ulb_w.detach(), dim=-1)


            unsup_loss = (ce_loss(logits_x_ulb_s, pseudo_label, reduction='none') * mask).mean()

            return unsup_loss #, global_threshold

def ce_loss(logits, targets, reduction='none'):

    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
