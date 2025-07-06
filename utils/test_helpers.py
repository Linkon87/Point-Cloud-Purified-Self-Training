import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
from networks.dgcnn import DGCNN


class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))


def build_model():
    # model for modelnet40
    print('Building DGCNN...')
    dgcnn = DGCNN().cuda()
    classifier = dgcnn.model.linear3
    dgcnn.model.linear3 = nn.Identity()
    net = ExtractorHead(dgcnn, classifier)
    return net, dgcnn, classifier  


def load_model(net, args):

    filename = 'pointdp/runs/model.1024.t7'
    print('Resuming from %s...' % filename)
    ckpt = torch.load(filename)
    # state_dict = ckpt['model_state']
    state_dict = ckpt

    prefix_to_remove = 'ae.'
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix_to_remove)}

    net_dict = {}
    for k, v in state_dict.items():
        if k[:14] == "module.linear3":
        # if k[:13] == "model.linear3":
        #     k = k.replace("model.linear3.", "head.")
            k = k.replace("module.linear3.", "head.")
        else:
            # k = k.replace("model.", "ext.model.")
            k = k.replace("module.", "ext.model.")
        net_dict[k] = v

    net.load_state_dict(net_dict, strict=True)
    print('Loaded model:', filename)
    return
    
    ##
    # print('purified pretrained')
    # filename= 'save/dgcnn_pretrain/cls_ae/model_best_test.pth'
    # ckpt = torch.load(filename)
    # state_dict = ckpt['model_state']


    # prefix_to_remove = 'ae.'
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix_to_remove)}

    # net_dict = {}
    # for k, v in state_dict.items():
    #     if k[:13] == "model.linear3":
    #         k = k.replace("model.linear3.", "head.")
    #     else:
    #         k = k.replace("model.", "ext.model.")
    #     net_dict[k] = v

    # net.load_state_dict(net_dict, strict=True)
    # print('Loaded model:', filename)
    # return

   


def test(dataloader, model, print_acc=False, **kwargs):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []

    all_label = []
    all_predict = []
    probs = list()
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if type(inputs) == list:
            inputs = inputs[0]
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs, **kwargs)
            loss = criterion(outputs, labels)

            all_label.append(labels)
            all_predict.append(outputs.max(1)[1])

            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
            probs.append(outputs.softmax(dim=1))

    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()

    all_label = torch.cat(all_label)
    all_predict = torch.cat(all_predict)

    probs = torch.cat(probs)

    if print_acc:
        confusion_matrix_ = confusion_matrix(all_label.cpu(), all_predict.cpu())
        acc = confusion_matrix_.diagonal() / confusion_matrix_.sum(axis=1)
        aa = [str(np.round(i, 4)) for i in acc]
        acc = ' '.join(aa)
        print(acc)

    aacc = (all_label == all_predict).sum().item() / all_label.shape[0]
    model.train()
    return 1 - aacc, correct, losses, probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)