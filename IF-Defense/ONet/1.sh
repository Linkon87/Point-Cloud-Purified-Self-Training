#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointmlp/changing_inf_attack.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointmlp/changing_l2_attack.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet/changing_inf_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet/changing_l2_attacked.npy

# python opt_defense.py \
#   --sample_npoint 1024 \
#   --train False  \
#   --rep_weight 500.0  \
#   --data_root ./new_protocol/dgcnn/IF_3new/changing_inf_attack.npy

# python opt_defense.py \
#   --sample_npoint 1024 \
#   --train False  \
#   --rep_weight 500.0  \
#   --data_root ./new_protocol/dgcnn/IF_3new/changing_l2_attack.npy