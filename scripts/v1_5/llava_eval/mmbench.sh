#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-nov}"

SPLIT="mmbench_dev_20230712"
CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$OUTPUT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data//eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $DATADIR/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $OUTPUT
