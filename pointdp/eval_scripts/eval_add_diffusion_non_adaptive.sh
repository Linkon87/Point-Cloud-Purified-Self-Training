for model in  'pct' 'curvenet' 'pointnet' 'pointnet2'  'pointMLP' 'dgcnn'; do #

if [ ! -d "./output/${model}" ]; then
    mkdir "./output/${model}"
fi

for t in 5 10 15 20 25 30 35 40 45 50; do

python configs/modify_config.py --data_path ./configs/attack/add/${model}_truncate_diffusion_pgd_add_naive.yaml --t ${t}

# CUDA_VISIBLE_DEVICES=0 python main.py --entry attack --model-path ./cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_pgd_add.yaml > ./output/${model}/add_baseline.txt

CUDA_VISIBLE_DEVICES=1 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_truncate_diffusion_pgd_add_naive.yaml --confusion > ./output/${model}/add_diffusion_non_adaptive_${t}.txt

done

done