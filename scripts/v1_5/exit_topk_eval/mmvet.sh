#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-99}"
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-99}"
LAYER_LIST="${LAYER_LIST:-[8,16,24]}"
IMAGE_TOKEN_KEEP_LIST="${IMAGE_TOKEN_KEEP_LIST:-[288,144,72]}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $DATADIR/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --late_entry_layer $LATE_ENTRY_LAYER \
    --early_exit_layer $EARLY_EXIT_LAYER \
    --layer_list "$LAYER_LIST" \
    --image_token_keep_list "$IMAGE_TOKEN_KEEP_LIST"

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$OUTPUT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$OUTPUT.json

