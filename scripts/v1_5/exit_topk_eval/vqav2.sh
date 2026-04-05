#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL="${MODEL:-llava-v1.5-7b}"
OUTPUT="${OUTPUT:-llava-v1.5-7b}"
LATE_ENTRY_LAYER="${LATE_ENTRY_LAYER:-99}"
EARLY_EXIT_LAYER="${EARLY_EXIT_LAYER:-99}"
LAYER_LIST="${LAYER_LIST:-[8,16,24]}"
IMAGE_TOKEN_KEEP_LIST="${IMAGE_TOKEN_KEEP_LIST:-[288,144,72]}"

SPLIT="llava_vqav2_mscoco_test-dev2015"
CKPTDIR="/hkfs/work/workspace/scratch/tum_yvc3016-compression/workspace/ckpts/ckpts-topk"
DATADIR="../../datasets/LLaVA-1.5-dataset"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPTDIR/$MODEL \
        --question-file $DATADIR/eval/vqav2/$SPLIT.jsonl \
        --image-folder $DATADIR/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$OUTPUT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --late_entry_layer $LATE_ENTRY_LAYER \
        --early_exit_layer $EARLY_EXIT_LAYER \
        --layer_list "$LAYER_LIST" \
        --image_token_keep_list "$IMAGE_TOKEN_KEEP_LIST" &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$OUTPUT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$OUTPUT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $OUTPUT --eval_dir ./playground/data/eval/vqav2

