import os

import numpy as np


advpc_recons = np.load('advpc_dgcnn_truncated_diffusion_latent_linf_1_recons.npy')
advpc_label = np.load('advpc_dgcnn_truncated_diffusion_latent_linf_1_ground.npy')

advpcl2_recons = np.load('advpcl2_dgcnn_truncated_diffusion_latent_linf_1_recons.npy')
#######
pgd_recons = np.load('dgcnn_truncate_diffusion_pgd_linf_1_t10_recons.npy')
pgdl2_recons = np.load(''
                       'dgcnn_truncate_diffusion_pgd_l2_1_t10_recons.npy')

#######
apgd_recons = np.load('non_adaptive_dgcnn_truncated_diffusion_apgd_linf_1_recons.npy')
apgdl2_recons = np.load('non_adaptive_dgcnn_truncated_diffusion_apgd_l2_1_recons.npy')
#######
bpda_recons = np.load('sde_dgcnn_truncated_diffusion_pgd_linf_1_recons.npy')
bpdal2_recons = np.load('sde_dgcnn_truncated_diffusion_pgd_l2_1_recons.npy')
#######


clean_purify = np.load('dgcnn_cls_ae_diffusion_1_recon.npy')


sorted_indices = np.argsort(advpc_label)
#######
sorted_advpc_label = advpc_label[sorted_indices]
sorted_advpc_recons = advpc_recons[sorted_indices]
sorted_advpcl2_recons = advpcl2_recons[sorted_indices]
#######
sorted_clean_recons = clean_purify[sorted_indices]
#######
sorted_apgdl2_recons = apgdl2_recons[sorted_indices]
sorted_apgd_recons = apgd_recons[sorted_indices]
#######
sorted_pgdl2_recons = pgdl2_recons[sorted_indices]
sorted_pgd_recons = pgd_recons[sorted_indices]
#######
sorted_bpdal2_recons = bpdal2_recons[sorted_indices]
sorted_bpda_recons = bpda_recons[sorted_indices]
###


advpc = []
clean1234 = []
advpc1234 = []
clean1234label = []
advpc1234label = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
        clean1234.append(batch_b)
        clean1234label.append(sorted_advpc_label[i:i + batch_size ])
    else:
        batch_b = sorted_advpc_recons[i:i + batch_size ]
        advpc1234.append(batch_b)
        advpc1234label.append(sorted_advpc_label[i:i + batch_size ])
    j += 1
    advpc.append(batch_b)
advpc = np.concatenate(advpc, axis=0)
clean1234 = np.concatenate(clean1234, axis=0)
advpc1234 = np.concatenate(advpc1234, axis=0)
clean1234label = np.concatenate(clean1234label, axis=0)
advpc1234label = np.concatenate(advpc1234label, axis=0)


advpcl2 = []
l2advpc1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_advpcl2_recons[i:i + batch_size ]
        l2advpc1234.append(batch_b)
    j += 1
    advpcl2.append(batch_b)
advpcl2 = np.concatenate(advpcl2, axis=0)
l2advpc1234 = np.concatenate(l2advpc1234, axis=0)


apgdl2 = []
l2apgd1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_apgdl2_recons[i:i + batch_size ]
        l2apgd1234.append(batch_b)

    j += 1
    apgdl2.append(batch_b)
apgdl2 = np.concatenate(apgdl2, axis=0)
l2apgd1234 = np.concatenate(l2apgd1234, axis=0)


apgdinf = []
apgd1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_apgd_recons[i:i + batch_size ]
        apgd1234.append(batch_b)
    j += 1
    apgdinf.append(batch_b)
apgdinf = np.concatenate(apgdinf, axis=0)
apgd1234 = np.concatenate(apgd1234, axis=0)


pgdl2 = []
l2pgd1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_pgdl2_recons[i:i + batch_size ]
        l2pgd1234.append(batch_b)
    j += 1
    pgdl2.append(batch_b)
pgdl2 = np.concatenate(pgdl2, axis=0)
l2pgd1234 = np.concatenate(l2pgd1234, axis=0)


pgd = []
pgd1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_pgd_recons[i:i + batch_size ]
        pgd1234.append(batch_b)
    j += 1
    pgd.append(batch_b)
pgd = np.concatenate(pgd, axis=0)
pgd1234 = np.concatenate(pgd1234, axis=0)

bpdal2 = []
l2bpda1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_bpdal2_recons[i:i + batch_size ]
        l2bpda1234.append(batch_b)
    j += 1
    bpdal2.append(batch_b)
bpdal2 = np.concatenate(bpdal2, axis=0)
l2bpda1234 = np.concatenate(l2bpda1234, axis=0)


bpda = []
bpda1234 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_bpda_recons[i:i + batch_size ]
        bpda1234.append(batch_b)
    j += 1
    bpda.append(batch_b)
bpda = np.concatenate(bpda, axis=0)
bpda1234 = np.concatenate(bpda1234, axis=0)



if not os.path.exists('single_attack'):
    os.mkdir('single_attack')

np.save('single_attack/clean_pgd_recons.npy',pgd)
np.save('single_attack/clean_pgdl2_recons.npy',pgdl2)
np.save('single_attack/clean_advpcl2_recons.npy',advpcl2)
np.save('single_attack/clean_advpc_recons.npy',advpc)
np.save('single_attack/clean_apgd_recons.npy',apgdinf)
np.save('single_attack/clean_apgdl2_recons.npy',apgdl2)
np.save('single_attack/clean_bpda_recons.npy',bpda)
np.save('single_attack/clean_bpdal2_recons.npy',bpdal2)
np.save('single_attack/label.npy',sorted_advpc_label)

# np.save('single_attack/clean_1234.npy',clean1234)
# np.save('single_attack/clean_1234_label.npy',clean1234label)
# np.save('single_attack/adv_1234_label.npy',advpc1234label)
#
# np.save('single_attack/advpc1234.npy',advpc1234)
# np.save('single_attack/l2advpc1234.npy',l2advpc1234)
# np.save('single_attack/l2pgd1234.npy',l2pgd1234)
# np.save('single_attack/pgd1234.npy',pgd1234)

