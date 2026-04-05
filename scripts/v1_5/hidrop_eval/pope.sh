#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-99}"
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-99}"
LAYER_LIST="${LAYER_LIST:-[8,16,24]}"
IMAGE_TOKEN_KEEP_LIST="${IMAGE_TOKEN_KEEP_LIST:-[288,144,72]}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_loader \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/pope/llava_pope_test.jsonl \
    --image-folder $DATADIR/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --late_entry_layer $LATE_ENTRY_LAYER \
    --early_exit_layer $EARLY_EXIT_LAYER \
    --layer_list "$LAYER_LIST" \
    --image_token_keep_list "$IMAGE_TOKEN_KEEP_LIST"

python llava/eval/eval_pope.py \
    --annotation-dir $DATADIR/eval/pope/coco \
    --question-file $DATADIR/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$OUTPUT.jsonl
