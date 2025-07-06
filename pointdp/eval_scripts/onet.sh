if [ ! -d "output" ]; then
    mkdir "output"
fi

for model in 'pointMLP'; do #'pointnet' 'pointnet2' 'dgcnn' 'curvenet'; do

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/defense/ONet/advpc_${model}_pgd_l2.yaml > ./output/advpc_${model}_pgd_l2.out

CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path cor_exp/dgcnn_${model}_run_1/model_best_test.pth --exp-config ./configs/defense/ONet/advpc_${model}_pgd.yaml > ./output/advpc_${model}_pgd.out

done