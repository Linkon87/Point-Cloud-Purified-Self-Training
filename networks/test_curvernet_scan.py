import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
from networks.curvenet_cls import CurveNet


class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))


def build_model():

    print('Building CURVENET...')
    curvenet = CurveNet(num_classes=15).cuda()
    classifier = curvenet.model.conv2
    curvenet.model.conv2 = nn.Identity()
    net = ExtractorHead(curvenet, classifier)
    return net, curvenet, classifier


def load_model(net, args):
    filename = 'pointdp/pretrained/scanobjectnn/curvenet/best_checkpoint.pth'
    ckpt = torch.load(filename)
    ckpt = ckpt['net']


    net_dict = {}
    for k, v in ckpt.items():
        k = k.replace("module.", "model.")
        net_dict[k] = v

    net_dict_new = {}
    for k, v in net_dict.items():
        if k[:11] == "model.conv2":
            k = k.replace("model.conv2", "head")
        else:
            k = k.replace("model.", "ext.model.")
        net_dict_new[k] = v

    net.load_state_dict(net_dict_new, strict=True)
    print('Loaded model:', filename)
    return


def test(dataloader, model, print_acc=False, **kwargs):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []

    all_label = []
    all_predict = []
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

    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()

    all_label = torch.cat(all_label)
    all_predict = torch.cat(all_predict)

    if print_acc:
        confusion_matrix_ = confusion_matrix(all_label.cpu(), all_predict.cpu())
        acc = confusion_matrix_.diagonal() / confusion_matrix_.sum(axis=1)
        aa = [str(np.round(i, 4)) for i in acc]
        acc = ' '.join(aa)
        print(acc)

    aacc = (all_label == all_predict).sum().item() / all_label.shape[0]
    model.train()
    return 1 - aacc, correct, losses




def main():
    import argparse
    from utils.prepare_dataset import prepare_test_data, prepare_train_data, create_dataloader,prepare_t_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='modelnet40')
    parser.add_argument('--dataroot', default='data')
    ########################################################################
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    ########################################################################
    parser.add_argument('--ckpt', default=None, type=int)
    ########################################################################
    ########################################################################
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--test_data', type=str, default='make_data/non_adaptive_dgcnn_truncated_diffusion_pgd_l2_1_change_dgcnn_attacked.npy')
    parser.add_argument('--test_label', type=str, default='make_data/non_adaptive_dgcnn_truncated_diffusion_cw_l2_1_change_dgcnn_ground.npy')
    args = parser.parse_args()

    print(args)
    ########### build and load model #################
    net, ext, classifier = build_model()

    load_model(net, args)
    ########### create dataset and dataloader #################
    te_dataset = prepare_t_data(args)  ###  test
    # te_dataset = prepare_test_data(args) ###


    te_dataloader = create_dataloader(te_dataset, args, False, False)

    ########### test before TTT #################
    print('Error (%)\t\ttest')
    err_cls = test(te_dataloader, net)[0]
    print(
          '%.2f\t\t' % (err_cls * 100))


if __name__ == '__main__':
    main()