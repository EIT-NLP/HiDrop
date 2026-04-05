#!/bin/bash


########### Note ###########
# global size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
# 40G --> 4node*4gpu*bs4*acc2=128
#     --> 2node*4gpu*bs4*acc4=128
# 80G --> 1node*4gpu*bS16*acc2=128 
# 96G --> 1node*4gpu*bS32*acc1=128
########### Note ###########

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank $SLURM_PROCID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     llava/train/train_mem_exit_topk.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/ckpts-llm/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ../../datasets/Open-LLaVA-NeXT-dataset/open-llava-next/open-llava-next_instruct_mix1M.json \
    --image_folder ../../datasets/Open-LLaVA-NeXT-dataset \
    --vision_tower ./checkpoints/ckpts-lmm/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/ckpts-topk/llava-v1.5-7b-exit-topk-Tok48-T10T14T16T18-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/ckpts-topk/llava-v1.5-7b-exit-topk-Tok48-T10T14T16T18-nov-1m \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
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
