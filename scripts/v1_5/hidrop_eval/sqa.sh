#!/bin/bash

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

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/$MODEL \
    --question-file $DATADIR/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $DATADIR/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$OUTPUT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --late_entry_layer $LATE_ENTRY_LAYER \
    --early_exit_layer $EARLY_EXIT_LAYER \
    --layer_list "$LAYER_LIST" \
    --image_token_keep_list "$IMAGE_TOKEN_KEEP_LIST"


python llava/eval/eval_science_qa.py \
    --base-dir $DATADIR/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$OUTPUT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${OUTPUT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${OUTPUT}_result.json
