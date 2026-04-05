#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

MODEL="${MODEL:-llava-v1.5-7b-exit-topk-Tok48}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-exit-topk-Tok48}"
#LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-99}"
#EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-99}"
#LAYER_LIST="${LAYER_LIST:-[8,16,24]}"
#IMAGE_TOKEN_KEEP_LIST="${IMAGE_TOKEN_KEEP_LIST:-[288,144,72]}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-9}"
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-25}"
LAYER_LIST="${LAYER_LIST:-[10,14,16,18]}"
IMAGE_TOKEN_KEEP_LIST="${IMAGE_TOKEN_KEEP_LIST:-[87,10,5,1]}"

DATADIR="./playground/data/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$MODEL \
    --question-file $DATADIR/eval/MME/llava_mme.jsonl \
    --image-folder $DATADIR/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --late_entry_layer $LATE_ENTRY_LAYER \
    --early_exit_layer $EARLY_EXIT_LAYER \
    --layer_list "$LAYER_LIST" \
    --image_token_keep_list "$IMAGE_TOKEN_KEEP_LIST"


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $OUTPUT

cd eval_tool

python calculation.py --results_dir answers/$OUTPUT
