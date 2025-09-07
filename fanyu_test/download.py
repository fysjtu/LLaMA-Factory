#模型下载
from modelscope import snapshot_download
# 下载Qwen3-1.7B模型，并打印模型本地存储路径
# 详细注释：此处调用modelscope的snapshot_download方法，下载Qwen3-1.7B模型（基础版），
# 并将下载后的本地路径打印出来，方便后续加载和使用。
qwen3_model_dir = snapshot_download('Qwen/Qwen3-1.7B')
print(qwen3_model_dir)
