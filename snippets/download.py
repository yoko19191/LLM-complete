
import time
from datetime import timedelta

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def download_with_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"下载耗时: {format_time(elapsed_time)}")
        return result
    return wrapper

# 1. 环境设置
import os
os.environ['HF_ENDPOINT'] = 'hf-mirror.com'  # 使用镜像站点
os.environ['HF_HOME'] = '/d/Data'  # 设置缓存目录
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 设置超时时间为300秒
os.environ['HF_HUB_DOWNLOAD_MAX_RETRIES'] = '5'  # 设置最大重试次数为5

# 2. 登录（如果需要访问私有仓库）
# import huggingface_hub
# huggingface_hub.login("HF_TOKEN")  # 从 https://huggingface.co/settings/tokens 获取 token

# 3. 下载整个模型仓库
from huggingface_hub import snapshot_download

@download_with_time
def download_full_repo(repo_id, local_dir, num_workers=8):
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        # proxies={"https": "http://localhost:7890"},  # 如果需要代理
        max_workers=num_workers,
        local_dir_use_symlinks=False
    )
    print(f"模型下载到: {model_path}")

# 使用示例
local_dir = "d:/Data/meta-llama"
download_full_repo("meta-llama/Meta-Llama-3.1-8B", local_dir)

# 4. 下载特定文件
# from huggingface_hub import hf_hub_download

# @download_with_time
# def download_specific_file(repo_id, filename):
#     file_path = hf_hub_download(repo_id=repo_id, filename=filename)
#     print(f"文件下载到: {file_path}")

# # 使用示例
# # download_specific_file("bert-base-uncased", "config.json")

# # 5. 下载数据集
# from datasets import load_dataset
# @download_with_time
# def download_dataset(dataset_name, subset=None):
#     dataset = load_dataset(dataset_name, subset)
#     print(dataset)

# # 使用示例
# dataset_name = "BatsResearch/ctga-v1"
# download_dataset(dataset_name)

# # 6. 下载数据集特定文件
# @download_with_time
# def download_dataset_file(repo_id, filename):
#     file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
#     print(f"数据集文件下载到: {file_path}")

# 使用示例
# download_dataset_file("dataset/glue", "mrpc/train.tsv")

# 7. 使用命令行工具下载
# 在终端中运行以下命令：
# huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf
# huggingface-cli download --repo-type dataset username/dataset-name --local-dir ./my_dataset

# 8. 使用 hfd 脚本下载（需要先下载脚本）
# wget https://hf-mirror.com/hfd/hfd.sh
# chmod a+x hfd.sh
# export HF_ENDPOINT=https://hf-mirror.com
# ./hfd.sh gpt2 --tool aria2c -x 4
# ./hfd.sh wikitext --dataset --tool aria2c -x 4

# 9. 安装额外工具（如果需要）
# sudo apt install git-lfs aria2
# git lfs install