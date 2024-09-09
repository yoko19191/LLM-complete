#!/bin/bash

# 模型名称 (根据实际情况修改)
MODEL="gpt-3.5-turbo"
ENDPOINT="" 
API_KEY=""

# 遍历不同的请求数量和最大并发数
for num_requests in 1 2 4 8 16 32; do
    for max_concurrency in 4 8 16 32 64 128; do
        echo "Running benchmark with $num_requests requests and $max_concurrency max concurrency"
        # 
        python openai_api_benchmark.py --endpoint "$ENDPOINT" --api_key "$API_KEY" --model "$MODEL" --num_requests "$num_requests" --max_concurrency "$max_concurrency" --endpoint "$ENDPOINT" --api_key "$API_KEY"
        #python openai_api_benchmark.py --model "$MODEL" --num_requests "$num_requests" --max_concurrency "$max_concurrency"
    done
done


