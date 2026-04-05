#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

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

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$MODEL \
        --question-file $DATADIR/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
        --image-folder $DATADIR/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/$OUTPUT/${CHUNKS}_${IDX}.jsonl \
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

output_file=./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/$OUTPUT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/$OUTPUT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split llava_vqav2_mscoco_test-dev2015 --ckpt $OUTPUT --eval_dir ./playground/data/eval/vqav2

