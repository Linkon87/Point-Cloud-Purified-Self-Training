import torch
def single_attack(te_dataloader,net):
    correct = []
    correct_adv = []
    correct_clean = []
    for te_batch_idx, (te_inputs, te_labels) in enumerate(te_dataloader):
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

    print('Mix : {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))
    print('clean : {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
    print('adv : {:.2f}'.format(torch.cat(correct_adv).numpy().mean() * 100))
    del correct, correct_adv, correct_clean

def mix_attack(te_dataloader,net):
    correct = []
    correct_advpc = []
    correct_cw = []
    correct_pgd = []
    correct_knn_or_siadv = []
    correct_clean = []
    for te_batch_idx, (te_inputs, te_labels) in enumerate(te_dataloader):
        net.eval()
        with torch.no_grad():
            outputs = net(te_inputs[0].cuda())
            _, predicted = outputs.max(1)
            correct.append(predicted.cpu().eq(te_labels))

            te_inputs_idx = te_inputs[1]
            advpc_idx = (te_inputs_idx >= 0) & (te_inputs_idx <= 492)
            cw_idx = (te_inputs_idx >= 493) & (te_inputs_idx <= 985)
            pgd_idx = (te_inputs_idx >= 986) & (te_inputs_idx <= 1478)
            siadv_idx = (te_inputs_idx >= 1479) & (te_inputs_idx <= 1971)
            clean_idx = (te_inputs_idx >= 1972) & (te_inputs_idx <= 2467)

            correct_advpc.append(predicted[advpc_idx].cpu().eq(te_labels[advpc_idx]))
            correct_cw.append(predicted[cw_idx].cpu().eq(te_labels[cw_idx]))
            correct_pgd.append(predicted[pgd_idx].cpu().eq(te_labels[pgd_idx]))
            correct_knn_or_siadv.append(predicted[siadv_idx].cpu().eq(te_labels[siadv_idx]))
            correct_clean.append(predicted[clean_idx].cpu().eq(te_labels[clean_idx]))

    print('clean : {:.2f}'.format(torch.cat(correct_clean).numpy().mean() * 100))
    print('advpc : {:.2f}'.format(torch.cat(correct_advpc).numpy().mean() * 100))
    print('cw /sde : {:.2f}'.format(torch.cat(correct_cw).numpy().mean() * 100))
    print('pgd : {:.2f}'.format(torch.cat(correct_pgd).numpy().mean() * 100))
    print('siadv_or_knn /apgd : {:.2f}'.format(torch.cat(correct_knn_or_siadv).numpy().mean() * 100))
    print('Mix : {:.2f}'.format(torch.cat(correct).numpy().mean() * 100))

    del correct, correct_advpc, correct_cw, correct_pgd, correct_knn_or_siadv, correct_clean