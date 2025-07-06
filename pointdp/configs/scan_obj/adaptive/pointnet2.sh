#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)



OUT_1='./output/ScanObjectNN-pointnet2_advpc.txt'
OUT_2='./output/ScanObjectNN-pointnet2_advpcl2.txt'






python ./ScanObjectNN/main_pointnet2.py \
  --entry attack \
  --exp-config configs/scan/adaptive/pointnet2_advpc.yaml \
  --confusion \
  > "$OUT_1" 2>&1

python ./ScanObjectNN/main_pointnet2.py \
  --entry attack \
  --exp-config configs/scan/adaptive/pointnet2_advpcl2.yaml \
  --confusion \
  > "$OUT_2" 2>&1
