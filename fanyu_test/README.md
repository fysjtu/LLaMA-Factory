# LLaMA-Factory 预测脚本说明

## 脚本文件说明

### 1. pred.sh
- **用途**: 使用LoRA微调后的Qwen3模型进行预测
- **特点**: 包含LoRA适配器路径，使用微调后的模型权重
- **输出目录**: `/root/autodl-tmp/Qwen3_lora/predict`

### 2. pred_ori.sh  
- **用途**: 原始预测脚本（基础版本）
- **特点**: 缺少LoRA适配器参数，但保留了finetuning_type参数
- **输出目录**: `/root/autodl-tmp/Qwen3_lora/predict_ori`

### 3. pred_native.sh (新增)
- **用途**: 使用原生Qwen3-1.7B模型进行预测
- **特点**: 完全移除LoRA相关参数，直接使用基础模型
- **输出目录**: `/root/autodl-tmp/Qwen3_native/predict`

## 主要修改内容

### 从LoRA微调模型切换到原生模型的修改：

1. **移除的参数**:
   ```bash
   --adapter_name_or_path /root/autodl-tmp/Qwen3_lora/checkpoint-393
   --finetuning_type lora
   ```

2. **保留的参数**:
   - `--model_name_or_path`: 基础模型路径保持不变
   - `--template qwen`: 保持Qwen模板
   - 其他预测相关参数保持不变

3. **输出目录调整**:
   - 从 `Qwen3_lora` 改为 `Qwen3_native` 以区分输出

## 使用说明

### 运行原生模型预测：
```bash
cd /root/LLaMA-Factory/fanyu_test
./pred_native.sh
```

### 运行LoRA微调模型预测：
```bash
cd /root/LLaMA-Factory/fanyu_test  
./pred.sh
```

## 注意事项

1. **模型路径**: 确保基础模型路径 `/root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B` 存在
2. **数据集路径**: 使用绝对路径 `/root/LLaMA-Factory/data` 确保数据集可访问
3. **输出目录**: 脚本会自动创建输出目录
4. **GPU使用**: 默认使用第0号GPU，可根据需要修改 `CUDA_VISIBLE_DEVICES`

## 性能对比

- **原生模型**: 使用原始Qwen3-1.7B权重，推理速度快，但可能在某些任务上表现不如微调模型
- **LoRA微调模型**: 使用微调后的权重，在特定任务上表现更好，但推理速度稍慢
