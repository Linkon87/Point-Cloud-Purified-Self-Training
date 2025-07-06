import os

import numpy as np

advpc_recons = np.load('pointnet2_advpc_latent_ScanObjectNN_diffusion_recons.npy')
advpc_label = np.load('non_adaptive_pointMLP_diffusion_clean_ScanObjectNN_diffusion_ground.npy')

advpcl2_recons = np.load('pointnet2_advpcl2_latent_ScanObjectNN_diffusion_recons.npy')
#######
auto_recons = np.load('pointnet2_auto_latent_ScanObjectNN_diffusion_recons.npy')

autol2_recons = np.load('pointnet2_autol2_latent_ScanObjectNN_diffusion_recons.npy')

#######
pgd_recons = np.load('pointnet2_pgd-latent_ScanObjectNN_diffusion_recons.npy')
pgdl2_recons = np.load(''
                       'pointnet2_pgdl2-latent_ScanObjectNN_diffusion_recons.npy')


sde_recons = np.load('pointnet2_sde_latent_ScanObjectNN_diffusion_recons.npy')
sdel2_recons = np.load(''
                       'pointnet2_sdel2_latent_ScanObjectNN_diffusion_recons.npy')
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
sorted_auto_recons = auto_recons[sorted_indices]
sorted_autol2_recons = autol2_recons[sorted_indices]
#######

sorted_pgdl2_recons = pgdl2_recons[sorted_indices]
sorted_pgd_recons = pgd_recons[sorted_indices]

sorted_sdel2_recons = sdel2_recons[sorted_indices]
sorted_sde_recons = sde_recons[sorted_indices]
#######

###
new_attacked = []
batch_size = 64
batch_id = 0
for i in range(0, 2882, batch_size):
    if batch_id  == 0:
        batch_b = sorted_advpc_recons[i:i + batch_size  ]
    elif batch_id == 1:
        batch_b = sorted_auto_recons[i:i + batch_size  ]
    elif batch_id  == 2:
        batch_b = sorted_pgd_recons[i:i + batch_size  ]
    elif batch_id == 3:
        batch_b = sorted_clean_recons[i:i + batch_size]
    elif batch_id == 4:
        batch_b = sorted_sde_recons[i:i + batch_size]
  
    if batch_id == 4:
        batch_id = 0
    else:
        batch_id += 1
    new_attacked.append(batch_b)

new_attacked = np.concatenate(new_attacked, axis=0)


new_attackedl2 = []
batch_size = 64
batch_id = 0
for i in range(0, 2882, batch_size):
    if batch_id  == 0:
        batch_b = sorted_advpcl2_recons[i:i + batch_size  ]
    elif batch_id == 1:
        batch_b = sorted_autol2_recons[i:i + batch_size  ]
    elif batch_id  == 2:
        batch_b = sorted_pgdl2_recons[i:i + batch_size  ]
    elif batch_id == 3:
        batch_b = sorted_clean_recons[i:i + batch_size]
    elif batch_id == 4:
        batch_b = sorted_sdel2_recons[i:i + batch_size]
  
    if batch_id == 4:
        batch_id = 0
    else:
        batch_id += 1
    new_attackedl2.append(batch_b)

new_attackedl2 = np.concatenate(new_attackedl2, axis=0)

if not os.path.exists('mix_attack'):
    os.mkdir('mix_attack')
np.save('mix_attack/changing_l2_batch.npy',new_attackedl2)
np.save('mix_attack/changing_inf_batch.npy',new_attacked)
