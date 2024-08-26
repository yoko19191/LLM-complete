# https://zhuanlan.zhihu.com/p/663712983


# import os
# os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

# 你可以通过环境变量或在代码中设置缓存目录
# os.environ['HF_HOME'] = '/path/to/your/cache/directory'
# from huggingface_hub import set_cache_dir
# set_cache_dir('/path/to/your/cache/directory')

# import huggingface_hub
# huggingface_hub.login("HF_TOKEN") # token 从 https://huggingface.co/settings/tokens 获取



# 使用 snapshot_download 函数可以下载整个模型仓库
from huggingface_hub import snapshot_download

repo_id = "meta-llama/Meta-Llama-3.1-8B"
local_dir = "/cache"
num_workers = 8

model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    max_retries=5,
    retry_on_error=True,
    #proxies={"https": "http://localhost:7890"},
    max_workers=num_workers,
    local_dir_use_symlinks=False
)

print(f"模型下载到: {model_path}")


# # 如果你只想下载特定文件，可以使用 hf_hub_download 函数
# from huggingface_hub import hf_hub_download

# file_path = hf_hub_download(repo_id="bert-base-uncased", filename="config.json")
# print(f"文件下载到: {file_path}")


# # 对于数据集，你可以使用 datasets 库，它与 huggingface_hub 集成得很好

# from datasets import load_dataset

# # 下载数据集
# dataset = load_dataset("glue", "mrpc")
# print(dataset)

# # 如果你想手动下载数据集文件，可以像下载模型文件一样使用 hf_hub_download
# from huggingface_hub import hf_hub_download

# # 下载数据集文件
# file_path = hf_hub_download(repo_id="dataset/glue", filename="mrpc/train.tsv", repo_type="dataset")
# print(f"数据集文件下载到: {file_path}")


# pip install -U huggingface_hub
# huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf

# wget https://hf-mirror.com/hfd/hfd.sh
# chmod a+x hfd.sh
# export HF_ENDPOINT=https://hf-mirror.com
# ./hfd.sh gpt2 --tool aria2c -x 4
# ./hfd.sh wikitext --dataset --tool aria2c -x 4


# sudo apt install git-lfs aria2
# git lfs install 