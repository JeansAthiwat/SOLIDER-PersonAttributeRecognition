# for single gpu
# CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pa100k.yaml
# CUDA_VISIBLE_DEVICES=0 python infer.py --cfg ./configs/inference/bc_GPT_infer.yaml
# CUDA_VISIBLE_DEVICES=0 python infer.py --cfg ./configs/inference/bc_Labeller_infer.yaml
# CUDA_VISIBLE_DEVICES=0 python infer.py --cfg ./configs/inference/bc_test.yaml
# CUDA_VISIBLE_DEVICES=0 python infer.py --cfg ./configs/inference/bc_ctw_store-match-bag_2024-07-01_labeled_infer.yaml
CUDA_VISIBLE_DEVICES=0 python infer.py --cfg ./configs/inference/bc_ctw_store-match-bag_2024-07-01_everything_infer.yaml


# for multi-gpu
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1233 train.py --cfg ./configs/pa100k.yaml
