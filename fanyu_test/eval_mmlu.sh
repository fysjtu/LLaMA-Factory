CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval \
--model_name_or_path /root/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B \
--template qwen \
--task mmlu_test \
--lang en \
--n_shot 5 \
--batch_size 1