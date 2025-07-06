import numpy as np
from debugpy._vendored.pydevd.pydevd_attach_to_process.winappdbg.win32.version import os

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

auto = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_auto_recons[i:i + batch_size ]
    j += 1
    auto.append(batch_b)
auto = np.concatenate(auto, axis=0)

autol2 = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_autol2_recons[i:i + batch_size ]
    j += 1
    autol2.append(batch_b)
autol2 = np.concatenate(autol2, axis=0)

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


sdel2 = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_sdel2_recons[i:i + batch_size ]
    j += 1
    sdel2.append(batch_b)
sdel2 = np.concatenate(sdel2, axis=0)

sde = []
batch_size = 16
j =0
for i in range(0, 2882, batch_size):
    if j % 2 == 0:
        batch_b = sorted_clean_recons[i:i + batch_size  ]
    else:
        batch_b = sorted_sde_recons[i:i + batch_size ]
    j += 1
    sde.append(batch_b)
sde = np.concatenate(sde, axis=0)




if not os.path.exists('single_attack'):
    os.mkdir('single_attack')
np.save('single_attack/clean_sde_recons.npy',sde)
np.save('single_attack/clean_sdel2_recons.npy',sdel2)
np.save('single_attack/clean_autol2_recons.npy',autol2)
np.save('single_attack/clean_auto_recons.npy',auto)
np.save('single_attack/clean_advpcl2_recons.npy',advpcl2)
np.save('single_attack/clean_advpc_recons.npy',advpc)
np.save('single_attack/clean_pgd_recons.npy',pgd)
np.save('single_attack/clean_pgdl2_recons.npy',pgdl2)

np.save('single_attack/label.npy',sorted_advpc_label)

