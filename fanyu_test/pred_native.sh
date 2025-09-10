#!/bin/bash

# 使用原生Qwen3-1.7B模型进行预测的脚本
# 移除了LoRA适配器相关参数，直接使用基础模型

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B \
    --eval_dataset alpaca_gpt4_zh,identity,adgen_local \
    --dataset_dir /root/LLaMA-Factory/data \
    --template qwen \
    --output_dir /root/autodl-tmp/Qwen3_native/predict_ori \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate
