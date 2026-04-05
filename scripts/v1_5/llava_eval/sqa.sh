#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-nov}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_science \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $DATADIR/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$OUTPUT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1


python llava/eval/eval_science_qa.py \
    --base-dir $DATADIR/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$OUTPUT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${OUTPUT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${OUTPUT}_result.json
