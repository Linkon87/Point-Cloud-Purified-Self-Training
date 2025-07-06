import os

import numpy as np

advpc_recons = np.load('advpc_dgcnn_truncated_diffusion_latent_linf_1_attacked.npy')
advpc_label = np.load('advpc_dgcnn_truncated_diffusion_latent_linf_1_ground.npy')
advpcl2_recons = np.load('advpcl2_dgcnn_truncated_diffusion_latent_linf_1_attacked.npy')


pgdl2_recons = np.load(''
                       'dgcnn_truncate_diffusion_pgd_l2_1_t10_attacked.npy')
pgd_recons = np.load('dgcnn_truncate_diffusion_pgd_linf_1_t10_attacked.npy')

autoattack_recons = np.load('non_adaptive_dgcnn_truncated_diffusion_apgd_linf_1_attacked.npy')
autoattackl2_recons = np.load('non_adaptive_dgcnn_truncated_diffusion_apgd_l2_1_attacked.npy')

sde_recons = np.load('sde_dgcnn_truncated_diffusion_pgd_linf_1_attacked.npy')
sdel2_recons = np.load('sde_dgcnn_truncated_diffusion_pgd_l2_1_attacked.npy')


clean_purify = np.load('modelnet40_clean.npy')




sorted_indices = np.argsort(advpc_label)

sorted_advpc_label = advpc_label[sorted_indices]
sorted_advpc_recons = advpc_recons[sorted_indices]
sorted_advpcl2_recons = advpcl2_recons[sorted_indices]


sorted_clean_recons = clean_purify[sorted_indices]


sorted_pgdl2_recons = pgdl2_recons[sorted_indices]
sorted_pgd_recons = pgd_recons[sorted_indices]

sorted_apgd_recons = autoattack_recons[sorted_indices]
sorted_apgdl2_recons = autoattackl2_recons[sorted_indices]

sorted_sde_recons = sde_recons[sorted_indices]
sorted_sdel2_recons = sdel2_recons[sorted_indices]



new_recons = np.zeros_like(sorted_advpc_recons)


new_recons[0:493, :, :] = sorted_advpc_recons[0:493, :, :]
new_recons[493:986, :, :] = sorted_sde_recons[493:986, :, :]
new_recons[986:1479, :, :] = sorted_pgd_recons[986:1479, :, :]
new_recons[1479:1972, :, :] = sorted_apgd_recons[1479:1972, :, :]
new_recons[1972:2468, :, :] = sorted_clean_recons[1972:2468, :, :]



new_reconsl2 = np.zeros_like(sorted_advpc_recons)


new_reconsl2[0:493, :, :] = sorted_advpcl2_recons[0:493, :, :]
new_reconsl2[493:986, :, :] = sorted_sdel2_recons[493:986, :, :]
new_reconsl2[986:1479, :, :] = sorted_pgdl2_recons[986:1479, :, :]
new_reconsl2[1479:1972, :, :] = sorted_apgdl2_recons[1479:1972, :, :]
new_reconsl2[1972:2468, :, :] = sorted_clean_recons[1972:2468, :, :]






if not os.path.exists('mix_attack'):
    os.mkdir('mix_attack')
np.save('mix_attack/ada_changing_inf_attack.npy',new_recons)
np.save('mix_attack/ada_changing_l2_attack.npy',new_reconsl2)
np.save('mix_attack/label.npy',sorted_advpc_label)