#!/bin/bash


MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b-check}"

CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

python -m llava.eval.model_vqa_loader \
    --model-path $CKPTDIR/$MODEL \
    --question-file $DATADIR/eval/MME/llava_mme.jsonl \
    --image-folder $DATADIR/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$OUTPUT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $OUTPUT

cd eval_tool

python calculation.py --results_dir answers/$OUTPUT
