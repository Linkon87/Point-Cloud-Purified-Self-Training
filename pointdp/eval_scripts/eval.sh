if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'pointMLP'; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry attack --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/attack/${model}_pgd_l2.yaml --confusion >> ./output/${model}_pgd_l2.out

CUDA_VISIBLE_DEVICES=0 python main.py --entry attack --model-path runs/${model}_cls_ae_diffusion_step_1/model_best_test.pth --exp-config ./configs/attack/${model}_truncate_diffusion_pgd_l2.yaml --confusion >> ./output/${model}_truncate_diffusion_pgd_l2.out

done