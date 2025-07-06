import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
from networks.pointnet import POINTNET


class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))


# def build_model():
#     # model for modelnet40
#     print('Building pointnet...')
#     pointnet = POINTNET().cuda()
#     classifier = pointnet.fc3
#     pointnet.fc3 = nn.Identity()
#     net = ExtractorHead(pointnet, classifier)
#     return net, pointnet, classifier
def build_model():
    # model for modelnet40
    print('Building pointnet...')
    pointnet = POINTNET().cuda()
    classifier = nn.Sequential(
        pointnet.fc1,
        pointnet.bn1,
        pointnet.relu,
        pointnet.fc2,
        pointnet.dropout,
        pointnet.bn2,
        pointnet.relu,
        pointnet.fc3
    ).cuda()
    pointnet.fc1 = nn.Identity()
    pointnet.bn1 = nn.Identity()
    pointnet.bn2 = nn.Identity()
    pointnet.relu = nn.Identity()
    pointnet.dropout = nn.Identity()
    pointnet.fc2 = nn.Identity()
    pointnet.fc3 = nn.Identity()
    net = ExtractorHead(pointnet, classifier)
    return net, pointnet, classifier

def load_model(net, args):
    filename = 'save/pointnet/dgcnn_pointnet_run_1/model_best_test.pth'
    ckpt = torch.load(filename)
    net_dict = ckpt['model_state']
    prefix_to_remove = 'module.'
    net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
 #加载 clean_pretrained pointnet

    net_dict_new = {}
    for k, v in net_dict.items():
        if k[:9] == "model.fc2":
            k = k.replace("model.fc2.", "head.3.")
        elif k[:9] == "model.bn1": #
            k = k.replace("model.bn1.", "head.1.")
        elif k[:9] == "model.fc1":
            k = k.replace("model.fc1.", "head.0.")
        elif k[:9] == "model.bn2": #
            k = k.replace("model.bn2.", "head.5.")
        if k[:9] == "model.fc3":
            k = k.replace("model.fc3.", "head.7.")
        else:
            k = k.replace("model.", "ext.")
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
    # te_dataset = prepare_test_data(args) ###  改成adv_points


    te_dataloader = create_dataloader(te_dataset, args, False, False)

    ########### test before TTT #################
    print('Error (%)\t\ttest')
    err_cls = test(te_dataloader, net)[0]
    print(
          '%.2f\t\t' % (err_cls * 100))


if __name__ == '__main__':
    main()