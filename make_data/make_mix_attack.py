import os

import numpy as np

# replace dataroot to your data
advpc_attacked = np.load('advpc_dgcnn_truncated_diffusion_pgd_linf_1_attacked.npy')
advpc_label = np.load('advpc_dgcnn_truncated_diffusion_pgd_linf_1_ground.npy')
advpcl2_attacked = np.load('advpc_dgcnn_truncated_diffusion_pgd_l2_1_attacked.npy')

cwl2_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_cw_l2_1_change_dgcnn_attacked.npy')
cw_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_cw_inf_1__attacked.npy')

pgdl2_attacked = np.load(''
                       'non_adaptive_dgcnn_truncated_diffusion_pgd_l2_1_change_dgcnn_attacked.npy')
pgd_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_pgd_linf_1_attacked.npy')

si_adv_attacked = np.load('adv.npy')

knn_attacked = np.load('non_adaptive_dgcnn_truncated_diffusion_knn_l2_1_attacked.npy')

clean_purify = np.load('modelnet40_clean.npy')




sorted_indices = np.argsort(advpc_label)
sorted_advpc_label = advpc_label[sorted_indices]
sorted_advpc_attacked = advpc_attacked[sorted_indices]
sorted_advpcl2_attacked = advpcl2_attacked[sorted_indices]


sorted_clean_attacked = clean_purify[sorted_indices]

sorted_cwl2_attacked = cwl2_attacked[sorted_indices]
sorted_cw_attacked = cw_attacked[sorted_indices]


sorted_pgdl2_attacked = pgdl2_attacked[sorted_indices]
sorted_pgd_attacked = pgd_attacked[sorted_indices]



sorted_knn_attacked = knn_attacked[sorted_indices]



new_attacked = np.zeros_like(sorted_advpc_attacked)


new_attacked[0:493, :, :] = sorted_advpc_attacked[0:493, :, :]
new_attacked[493:986, :, :] = sorted_cw_attacked[493:986, :, :]
new_attacked[986:1479, :, :] = sorted_pgd_attacked[986:1479, :, :]
new_attacked[1479:1972, :, :] = si_adv_attacked[1479:1972, :, :]
new_attacked[1972:2468, :, :] = sorted_clean_attacked[1972:2468, :, :]




new_attackedl2 = np.zeros_like(sorted_advpc_attacked)


new_attackedl2[0:493, :, :] = sorted_advpcl2_attacked[0:493, :, :]
new_attackedl2[493:986, :, :] = sorted_cwl2_attacked[493:986, :, :]
new_attackedl2[986:1479, :, :] = sorted_pgdl2_attacked[986:1479, :, :]
new_attackedl2[1479:1972, :, :] = sorted_knn_attacked[1479:1972, :, :]
new_attackedl2[1972:2468, :, :] = sorted_clean_attacked[1972:2468, :, :]


if not os.path.exists('mix_attack'):
    os.mkdir('mix_attack')
np.save('mix_attack/changing_inf_attack.npy',new_attacked)
np.save('mix_attack/changing_l2_attack.npy',new_attackedl2)
np.save('mix_attack/label.npy',sorted_advpc_label)