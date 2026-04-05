#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-nov}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $DATADIR/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$OUTPUT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$OUTPUT.json

