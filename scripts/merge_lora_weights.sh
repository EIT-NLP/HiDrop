#!/bin/bash


python ./scripts/merge_lora_weights.py \
            --model-path ./checkpoints/ckpts-topk/llava-v1.5-3b-exit-E15E28-lora-weights \
            --model-base ./checkpoints/ckpts-llm/mobilellama-2.7b-chat \
            --save-model-path ./checkpoints/ckpts-topk/llava-v1.5-3b-exit-E15E28-lora
