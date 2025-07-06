for model in 'pct' 'curvenet' 'pointnet' 'pointnet2' 'dgcnn' 'pointMLP'; do

if [ ! -d "./output/${model}" ]; then
    mkdir "./output/${model}"
fi

CUDA_VISIBLE_DEVICES=1 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/sde/${model}_truncate_diffusion_pgd_sde.yaml > ./output/${model}/sde_linf.txt

# CUDA_VISIBLE_DEVICES=2 python main.py --entry attack --model-path ./runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/add/${model}_truncate_diffusion_pgd_add.yaml > ./output/${model}/add_diffusion_feature.txt

done