for model in 'pct'; do #'curvenet' 'pointnet' 'pointnet2' 'dgcnn' 'pointMLP'; do

if [ ! -d "./output/${model}" ]; then
    mkdir "./output/${model}"
fi

CUDA_VISIBLE_DEVICES=0 python main.py --entry attack --model-path ./cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_pgd_add.yaml > ./output/${model}/add_baseline.txt

# CUDA_VISIBLE_DEVICES=2 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_truncate_diffusion_pgd_add.yaml > ./output/${model}/add_diffusion_feature.txt

done
