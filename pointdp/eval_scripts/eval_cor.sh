for model in 'pointnet2' 'pointMLP' 'pct'; do

if [ ! -d "./output/c/${model}" ]; then
    mkdir "./output/c/${model}"
fi

for cor in 'uniform' 'gaussian' 'background' 'impulse' 'upsampling' 'distortion_rbf' 'distortion_rbf_inv' 'density' 'density_inc' 'shear' 'rotation' 'cutout' 'distortion'  'occlusion' 'lidar'; do

for sev in 1 2 3 4 5; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/corruption/${model}_truncate_diffusion.yaml --corruption ${cor} --severity ${sev} > ./output/c/${model}/${cor}_${sev}.txt

done

done

done