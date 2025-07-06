#for model in 'pct' 'curvenet' 'pointnet' 'pointnet2' 'dgcnn' 'pointMLP'; do
for model in 'dgcnn'; do


if [ ! -d "./output/${model}" ]; then
    mkdir "./output/${model}"
fi

for t in 5 10 15 20 25 30 35 40 45 50; do

python configs/modify_config.py --data_path ./configs/attack/${model}_truncate_diffusion_pgd_naive.yaml --t ${t}

CUDA_VISIBLE_DEVICES=1 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/${model}_truncate_diffusion_pgd_naive.yaml > ./output/${model}/pgd_naive_${t}.txt

# CUDA_VISIBLE_DEVICES=2 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_truncate_diffusion_pgd_add.yaml > ./output/${model}/add_diffusion_feature.txt

done

done