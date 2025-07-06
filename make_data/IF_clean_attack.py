import numpy as np
import os
advpc_attacked = np.load('advpc_dgcnn_truncated_diffusion_pgd_linf_1_attacked.npy')
advpc_label = np.load('advpc_dgcnn_truncated_diffusion_pgd_linf_1_ground.npy')

advpcl2_attacked = np.load('advpc_dgcnn_truncated_diffusion_pgd_l2_1_attacked.npy')
#######
cwl2_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_cw_l2_1_change_dgcnn_attacked.npy')

cwinf_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_cw_inf_1__attacked.npy')

#######
pgd_attacked = np.load('advpc_dgcnn_truncated_diffusion_pgd_linf_1_attacked.npy')
pgdl2_attacked = np.load(''
                       'non_adaptive_dgcnn_truncated_diffusion_pgd_l2_1_change_dgcnn_t50_attacked.npy')

#######
knn_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_knn_l2_1_attacked.npy')



#######

clean_purify = np.load('modelnet40_clean.npy')

si_adv_attacked = np.load('adv.npy')
si_adv_attacked_label = np.load('non_adaptive_dgcnn_truncated_diffusion_si_adv_ground.npy')






sorted_indices = np.argsort(advpc_label)
#######
sorted_advpc_label = advpc_label[sorted_indices]
sorted_advpc_attacked = advpc_attacked[sorted_indices]
sorted_advpcl2_attacked = advpcl2_attacked[sorted_indices]
#######
sorted_clean_attacked = clean_purify[sorted_indices]

#######
sorted_cwl2_attacked = cwl2_attacked[sorted_indices]
sorted_cwinf_attacked = cwinf_attacked[sorted_indices]
#######

sorted_pgdl2_attacked = pgdl2_attacked[sorted_indices]
sorted_pgd_attacked = pgd_attacked[sorted_indices]
#######
sorted_knn_attacked = knn_attacked[sorted_indices]

###
# sorted_drop_attacked = drop_attacked[sorted_indices]
# sorted_add_attacked = add_attacked[sorted_indices]
# ###


advpc = []

batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
       
    else:
        batch_b = sorted_advpc_attacked[i:i + batch_size ]
        
    j += 1
    advpc.append(batch_b)
advpc = np.concatenate(advpc, axis=0)



advpcl2 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_advpcl2_attacked[i:i + batch_size ]
    j += 1
    advpcl2.append(batch_b)
advpcl2 = np.concatenate(advpcl2, axis=0)


cwl2 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_cwl2_attacked[i:i + batch_size ]

    j += 1
    cwl2.append(batch_b)
cwl2 = np.concatenate(cwl2, axis=0)


cwinf = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_cwinf_attacked[i:i + batch_size ]
    j += 1
    cwinf.append(batch_b)
cwinf = np.concatenate(cwinf, axis=0)


pgdl2 = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_pgdl2_attacked[i:i + batch_size ]
    j += 1
    pgdl2.append(batch_b)
pgdl2 = np.concatenate(pgdl2, axis=0)


pgd = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_pgd_attacked[i:i + batch_size ]
    j += 1
    pgd.append(batch_b)
pgd = np.concatenate(pgd, axis=0)


knn = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = sorted_knn_attacked[i:i + batch_size ]
    j += 1
    knn.append(batch_b)
knn = np.concatenate(knn, axis=0)

si_adv = []
batch_size = 16
j =0
for i in range(0, 2468, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_attacked[i:i + batch_size  ]
    else:
        batch_b = si_adv_attacked[i:i + batch_size ]
    j += 1
    si_adv.append(batch_b)
si_adv = np.concatenate(si_adv, axis=0)

if not os.path.exists('IF_3new'):
    os.mkdir('IF_3new')

np.save('IF_3new/clean_knn_attacked.npy',knn)
np.save('IF_3new/clean_pgd_attacked.npy',pgd)
np.save('IF_3new/clean_pgdl2_attacked.npy',pgdl2)
np.save('IF_3new/clean_cwinf_attacked.npy',cwinf)
np.save('IF_3new/clean_cwl2_attacked.npy',cwl2)
np.save('IF_3new/clean_advpcl2_attacked.npy',advpcl2)
np.save('IF_3new/clean_advpc_attacked.npy',advpc)
np.save('IF_3new/label.npy',sorted_advpc_label)

np.save('IF_3new/clean_si_adv_attacked.npy',si_adv)
