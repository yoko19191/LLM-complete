#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_memory_monitor_refresh_ms=0

python sft.py \
    --model_name_or_path 'Llama-2-7b-hf' \
    --data_path "alpaca_data_en_52k.json" \
    --bf16 True \
    --output_dir checkpoints/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000\
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True