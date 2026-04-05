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
GQADIR="./playground/data/LLaVA-1.5-dataset/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ./checkpoints/$MODEL \
        --question-file $DATADIR/eval/gqa/llava_gqa_testdev_balanced.jsonl \
        --image-folder $DATADIR/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$OUTPUT/${CHUNKS}_${IDX}.jsonl \
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

output_file=./playground/data/eval/gqa/answers/$SPLIT/$OUTPUT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$OUTPUT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
