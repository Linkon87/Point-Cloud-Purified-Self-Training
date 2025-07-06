#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_pgd_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_pgdl2_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_advpc_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_advpcl2_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_cwinf_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_cwl2_attacked.npy

python opt_defense.py \
  --sample_npoint 1024 \
  --train False  \
  --rep_weight 500.0  \
  --data_root ./new_protocol/pointnet2/IF_3new/clean_si_adv_attacked.npy