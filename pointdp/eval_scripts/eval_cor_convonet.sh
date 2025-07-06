if [ ! -d "./output/c/ConvONet" ]; then
    mkdir "./output/c/ConvONet"
fi

for model in 'curvenet'; do #'pointnet' 'pointnet2' 'dgcnn' 'pointMLP' 'pct'; do

if [ ! -d "./output/c/ConvONet/${model}" ]; then
    mkdir "./output/c/ConvONet/${model}"
fi

for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path ./cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/corruption/ConvONet/${model}.yaml --corruption ${cor} --severity ${sev} > ./output/c/ConvONet/${model}/${cor}_${sev}.txt

done

done

done