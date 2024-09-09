"""
This script is designed to benchmark the performance of an OpenAI API endpoint 
by sending multiple concurrent requests and measuring the latency and throughput.

Features:
- Allows specifying the API endpoint, API key, model name, number of requests, 
  and maximum concurrency via command-line arguments or environment variables.
- Uses asyncio and aiohttp to handle asynchronous HTTP requests with concurrency control.
- Calculates and displays performance metrics such as average latency, latency per token, 
  and requests/throughput per second.

Command-line Arguments:
- --endpoint: The API endpoint URL. If not provided, it's loaded from .env.
- --api-key: The API key for authentication. If not provided, it's loaded from .env.
- --model: The name of the model to use for inference (required).
- --num_requests: The total number of requests to send (required).
- --max_concurrency: The maximum number of concurrent requests (required).

Environment Variables:
- ENDPOINT: The API endpoint URL.
- API_KEY: The API key for authentication.

Usage:
- You can run this script directly from the command line, providing the necessary arguments,
  or by setting up a .env file with the required configurations.

Example:
python openai_api_benchmark.py --model "gpt-4o" --num_requests 100 --max_concurrency 10

Requirements:
- pip install aiohttp tqdm python-dotenv
"""


import time
import asyncio
import aiohttp
import statistics
import random
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm
from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv



class LLMInferenceTest:
    def __init__(self, endpoint: str, api_key: str, prompts: List[str], model: str,
                 num_requests: int, max_concurrency: int):
        self.endpoint = endpoint
        self.api_key = api_key
        self.prompts = prompts
        self.model = model
        self.num_requests = num_requests
        self.max_concurrency = max_concurrency
        self.results: List[Dict[str, Any]] = []

    async def single_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """执行单次推理请求并返回响应结果。"""
        prompt = random.choice(self.prompts)
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.perf_counter()
        async with session.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": self.model,
                "messages": messages
            }
        ) as response:
            response_json = await response.json()
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # 转换为毫秒

        usage = response_json.get("usage", {})
        return {
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }

    async def run_concurrent_tests(self):
        """管理并发请求，并收集结果。"""
        sem = asyncio.Semaphore(self.max_concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = []

            with tqdm(total=self.num_requests) as pbar:
                for _ in range(self.num_requests):
                    task = asyncio.ensure_future(self.bound_single_request(sem, session))
                    task.add_done_callback(lambda p: pbar.update())
                    tasks.append(task)

                self.results = await asyncio.gather(*tasks)

    async def bound_single_request(self, sem: asyncio.Semaphore, session: aiohttp.ClientSession):
        """限制并发执行的请求"""
        async with sem:
            return await self.single_request(session)

    def calculate_metrics(self):
        """计算并打印测试结果的指标。"""
        latencies = [result["latency"] for result in self.results]
        prompt_tokens = [result["prompt_tokens"] for result in self.results]
        completion_tokens = [result["completion_tokens"] for result in self.results]

        total_time = sum(latencies) / 1000  # 转换为秒
        avg_latency = statistics.mean(latencies)
        avg_latency_per_token = avg_latency / sum(prompt_tokens)
        avg_latency_per_output_token = avg_latency / sum(completion_tokens)
        throughput_requests = self.num_requests / total_time
        throughput_tokens = sum(completion_tokens) / total_time

        
        print("=" * 80)
        print("OpenAI Server API Performance Benchmark Results:")
        print(f"  Total requests            : {self.num_requests}")
        print(f"  Max concurrent requests   : {self.max_concurrency}")
        print(f"  Total time                : {total_time:.2f} seconds")
        print(f"  Average latency           : {avg_latency:.2f} ms")
        print(f"  Average latency per token : {avg_latency_per_token:.2f} ms")
        print(f"  Average latency per output token : {avg_latency_per_output_token:.2f} ms")
        print(f"  Throughput (requests/s)   : {throughput_requests:.2f}")
        print(f"  Throughput (tokens/s)     : {throughput_tokens:.2f}")
        print("=" * 80)

def parse_args():
    """解析命令行参数，并从.env文件中加载默认值"""
    _ = load_dotenv(find_dotenv())

    parser = ArgumentParser(description="LLM API Performance Test Script")
    parser.add_argument("--endpoint", type=str, default=os.getenv("ENDPOINT"),
                        help="API endpoint URL (default: loaded from .env)")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY"),
                        help="API key (default: loaded from .env)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to use for inference")
    parser.add_argument("--num_requests", type=int, required=True,
                        help="Number of requests to send")
    parser.add_argument("--max_concurrency", type=int, required=True,
                        help="Maximum number of concurrent requests")

    args = parser.parse_args()

    # 处理 endpoint 完整性
    if args.endpoint and not args.endpoint.endswith("/v1/chat/completions"):
        if args.endpoint.endswith("/v1"):
            args.endpoint += "/chat/completions"
        elif not args.endpoint.endswith("/v1/"):
            args.endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"
        else:
            args.endpoint += "chat/completions"

    # 验证必要的参数是否存在
    if not args.endpoint:
        print("Error: --endpoint is required (or set ENDPOINT in .env)")
        sys.exit(1)
    if not args.api_key:
        print("Error: -- is required (or set API_KEY in .env)")

    return args

if __name__ == '__main__':
    args = parse_args()

    # 示例配置
    prompts = [
        "Write a haiku about artificial intelligence.",
        "Explain quantum computing in simple terms.",
        "What are the ethical implications of autonomous vehicles?",
        "Describe the process of photosynthesis in a creative way.",
        "How might climate change affect global food security?",
        "Write a short story about a robot learning to paint.",
        "What are the potential applications of CRISPR technology?",
        "Explain the concept of blockchain to a 10-year-old.",
        "Describe the cultural significance of tea ceremonies in Japan.",
        "How might space exploration benefit life on Earth?",
        "描述未来100年后的交通工具可能是什么样子。",
        "解释为什么蜜蜂对生态系统如此重要。",
        "写一篇关于深海探索的短文。",
        "3D打印技术将如何改变医疗行业？",
        "用比喻的方式解释大脑如何存储记忆。",
        "探讨社交媒体对现代人际关系的影响。",
        "描述一个没有互联网的世界会是什么样子。",
        "纳米技术在环保领域有哪些潜在应用？",
        "写一个关于时间旅行的科幻故事梗概。",
        "解释为什么保护生物多样性对人类至关重要。"
    ]
    
    print("=" * 80)
    print("Running OpenAI Server API Performance Test...")
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Number of requests: {args.num_requests}")
    print(f"Max concurrent requests: {args.max_concurrency}")
    print("=" * 80)

    test = LLMInferenceTest(args.endpoint, args.api_key, prompts, args.model,
                            args.num_requests, args.max_concurrency)

    start_time = time.perf_counter()
    asyncio.run(test.run_concurrent_tests())
    end_time = time.perf_counter()

    test.calculate_metrics()
