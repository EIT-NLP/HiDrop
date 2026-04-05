#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-nov}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_loader \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/pope/llava_pope_test.jsonl \
    --image-folder $DATADIR/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python llava/eval/eval_pope.py \
    --annotation-dir $DATADIR/eval/pope/coco \
    --question-file $DATADIR/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$OUTPUT.jsonl
