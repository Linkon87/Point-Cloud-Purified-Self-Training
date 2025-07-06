import os

import numpy as np

advpc_recons = np.load('non_adaptive_pointnet2_diffusion_advpc_ScanObjectNN_diffusion_recons.npy')
advpc_label = np.load('non_adaptive_pointMLP_diffusion_clean_ScanObjectNN_diffusion_ground.npy')

advpcl2_recons = np.load('non_adaptive_pointnet2_diffusion_advpcl2_ScanObjectNN_diffusion_recons.npy')
#######
cwl2_recons = np.load('non_adaptive_pointnet2_diffusion_cwl2_ScanObjectNN_diffusion_recons.npy')

cwinf_recons = np.load('non_adaptive_pointnet2_diffusion_cw_ScanObjectNN_diffusion_recons.npy')

#######
pgd_recons = np.load('non_adaptive_pointnet2_diffusion_pgd_ScanObjectNN_diffusion_recons.npy')
pgdl2_recons = np.load(''
                       'non_adaptive_pointnet2_diffusion_pgdl2_ScanObjectNN_diffusion_recons.npy')


knn_recons = np.load('non_adaptive_pointnet2_diffusion_knn_ScanObjectNN_diffusion_recons.npy')

#######

clean_purify = np.load('non_adaptive_pointMLP_diffusion_clean_ScanObjectNN_diffusion_recons.npy')




sorted_indices = np.argsort(advpc_label)
#######
sorted_advpc_label = advpc_label[sorted_indices]
sorted_advpc_recons = advpc_recons[sorted_indices]
sorted_advpcl2_recons = advpcl2_recons[sorted_indices]
#######
sorted_clean_recons = clean_purify[sorted_indices]

#######
sorted_cwl2_recons = cwl2_recons[sorted_indices]
sorted_cwinf_recons = cwinf_recons[sorted_indices]
#######

sorted_pgdl2_recons = pgdl2_recons[sorted_indices]
sorted_pgd_recons = pgd_recons[sorted_indices]

sorted_knn_recons = knn_recons[sorted_indices]


advpc = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_advpc_recons[i:i + batch_size ]
    j += 1
    advpc.append(batch_b)
advpc = np.concatenate(advpc, axis=0)

advpcl2 = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_advpcl2_recons[i:i + batch_size ]
    j += 1
    advpcl2.append(batch_b)
advpcl2 = np.concatenate(advpcl2, axis=0)

cwl2 = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_cwl2_recons[i:i + batch_size ]
    j += 1
    cwl2.append(batch_b)
cwl2 = np.concatenate(cwl2, axis=0)

cwinf = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_cwinf_recons[i:i + batch_size ]
    j += 1
    cwinf.append(batch_b)
cwinf = np.concatenate(cwinf, axis=0)

pgdl2 = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_pgdl2_recons[i:i + batch_size ]
    j += 1
    pgdl2.append(batch_b)
pgdl2 = np.concatenate(pgdl2, axis=0)

pgd = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_pgd_recons[i:i + batch_size ]
    j += 1
    pgd.append(batch_b)
pgd = np.concatenate(pgd, axis=0)


knn = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_knn_recons[i:i + batch_size ]
    j += 1
    knn.append(batch_b)
knn = np.concatenate(knn, axis=0)




if not os.path.exists('single_attack'):
    os.mkdir('single_attack')
np.save('single_attack/clean_pgd_recons.npy',pgd)
np.save('single_attack/clean_pgdl2_recons.npy',pgdl2)
np.save('single_attack/clean_cwinf_recons.npy',cwinf)
np.save('single_attack/clean_cwl2_recons.npy',cwl2)
np.save('single_attack/clean_advpcl2_recons.npy',advpcl2)
np.save('single_attack/clean_advpc_recons.npy',advpc)

np.save('single_attack/clean_knn_recons.npy',knn)

np.save('single_attack/label.npy',sorted_advpc_label)