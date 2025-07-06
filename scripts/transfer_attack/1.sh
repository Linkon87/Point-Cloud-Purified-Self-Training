#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

OUT_1='./out/transfer_attack/curvenet/adaptive-pointnet2-ada_changing_inf_transfer_attack.txt'
OUT_2='./out/transfer_attack/pointnet2/adaptive-curvenet-ada_changing_inf_transfer_attack.txt'
OUT_3='./out/transfer_attack/pointnet2/adaptive-pct-ada_changing_inf_transfer_attack.txt'





 python stream_mix_attack.py \
   --backbone curvenet \
   --test_data attack_recons/pointnet2/adaptive/4changing/ada_changing_inf.npy  \
   --test_label attack_recons/pointnet2/adaptive/4changing/label.npy \
   > "$OUT_1" 2>&1
