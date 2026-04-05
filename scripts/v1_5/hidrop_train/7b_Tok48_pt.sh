#!/bin/bash

########### Note ###########
# global size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 256
# 40G --> 4node*4gpu*bs16*acc1=256
#     --> 2node*4gpu*bs32*acc1=256
# 80G --> 1node*4gpu*BS64*acc1=256
# 96G --> 1node*4gpu*BS64*acc1=256
########### Note ###########


##### Slurm
#srun torchrun \
#    --nnodes=$SLURM_NNODES \
#    --nproc_per_node=4 \
#    --node_rank $SLURM_PROCID \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#    llava/train/train_mem_exit_topk.py \
#    --deepspeed ./scripts/zero2.json \
#    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
#    --version plain \
#    --data_path ./playground/data/LLaVA-1.5-dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#    --image_folder ./playground/data/LLaVA-1.5-dataset/LLaVA-Pretrain/images \
#    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
#    --mm_projector_type mlp2x_gelu \
#    --tune_mm_mlp_adapter True \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --bf16 True \
#    --output_dir ./checkpoints/llava-v1.5-7b-exit-topk-Tok48-pretrain \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 1000 \
#    --save_total_limit 2 \
#    --learning_rate 1e-3 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --late_entry_layer  9 \
#    --early_exit_layer 25 \
#    --layer_list "[10,14,16,18]" \
#    --image_token_keep_list "[87,10,5,1]" \
#    --report_to wandb


##### bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export WANDB_MODE=offline

deepspeed llava/train/train_mem_exit_topk.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-1.5-dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-1.5-dataset/LLaVA-Pretrain/images \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-exit-topk-Tok48-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --late_entry_layer  9 \
    --early_exit_layer 25 \
    --layer_list "[10,14,16,18]" \
    --image_token_keep_list "[87,10,5,1]" \
    --report_to wandb