#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


OUT_1='./out/pointnet/onet_opt/{clean_pgd_attacked-}.txt'
OUT_2='./out/pointnet/onet_opt/{clean_pgdl2_attacked-}.txt'

OUT_3='./out/pointnet/onet_opt/{clean_advpc_attacked-}.txt'
OUT_4='./out/pointnet/onet_opt/{clean_advpcl2_attacked-}.txt'

OUT_5='./out/pointnet/onet_opt/{clean_cwinf_attacked-}.txt'
OUT_6='./out/pointnet/onet_opt/{clean_cwl2_attacked-}.txt'

OUT_7='./out/pointnet/onet_opt/{clean_si_adv_attacked-}.txt'
OUT_8='./out/pointnet/onet_opt/{clean_knn_attacked-}.txt'

OUT_9='./out/pointnet/onet_opt/{changing_inf_attacked-}.txt'
OUT_10='./out/pointnet/onet_opt/{changing_l2_attacked-}.txt'

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_pgd_attacked.npz  \
  > "$OUT_1" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_pgdl2_attacked.npz  \
  > "$OUT_2" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_advpc_attacked.npz  \
  > "$OUT_3" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_advpcl2_attacked.npz  \
  > "$OUT_4" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_cwinf_attacked.npz  \
  > "$OUT_5" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_cwl2_attacked.npz  \
  > "$OUT_6" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_si_adv_attacked.npz  \
  > "$OUT_7" 2>&1

python stream_single_attack.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-clean_knn_attacked.npz  \
  > "$OUT_8" 2>&1


python all_onepass+free_changing.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-changing_inf_attacked.npz  \
  > "$OUT_9" 2>&1

python all_onepass+free_changing.py \
  --backbone pointnet \
  --test_data ONet-Opt/pointnet/onet_opt-changing_l2_attacked.npz  \
  > "$OUT_10" 2>&1