import os
os.environ['MODELSCOPE_CACHE'] = '/path/to/your/cache/directory'

# from modelscope.hub.api import login

# # 使用你的ModelScope账号登录
# login(your_access_token)


from modelscope import snapshot_download

# 魔塔地址：https://modelscope.cn/home, 在这上面找到模型路径，修改即可
#model_path="ZhipuAI/glm-4-9b"
#model_path = "01ai/Yi-6B"
#model_path = "qwen/Qwen2-7B"
model_path = "qwen/Qwen2-7B-Instruct"

#model_path = "LLM-Research/Meta-Llama-3-8B"

cache_path="/root/autodl-tmp/models"

model_dir = snapshot_download(model_path, 
                              cache_dir=cache_path,
                              retry_times=5)

print(f"模型下载到: {model_dir}")




# 查看模型信息
# from modelscope import hub

# model_id = 'damo/nlp_structbert_sentence-similarity_chinese-base'
# config = hub.read_configuration(model_id)
# print(config)