#!/bin/bash

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-nov}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_loader \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $DATADIR/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file $DATADIR/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$OUTPUT.jsonl
