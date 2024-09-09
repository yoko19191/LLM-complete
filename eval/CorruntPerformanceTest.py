"""
ConcurrenceInferenceTest.py
test items: 
1. total time(s)
2. Throughput(requests/s)
3. Average latency(ms)
4. Average latency per token(ms)
5. Average latency per output token(ms)
6. Throughput(tokens/s)
""" 


import time
import asyncio
import aiohttp
import statistics
import random
from typing import List, Dict, Any 


class LLMInferenceTest:
    def __init__(self, endpoint: str, api_key: str, prompts: List[str], model: str,
                 num_requests: int, max_concurrency: int):
        self.endpoint = endpoint
        self.api_key = api_key
        self.prompts = prompts
        self.model = model
        self.num_requests = num_requests
        self.max_concurrency = max_concurrency
        self.results: List[Dict[str, any]] = []
    
        
    async def single_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Performance a single inference request and return the response result."""
        prompt = random.choice(self.prompts)
        messages = [{"role":"user", "content": prompt}]
        
        start_time = time.perf_counter()
        async with session.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json = {
                "model": self.model,
                "messages": messages
            }
        ) as response:
            response_json = await response.json()
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000 # convert to milliseconds
        
        print(response_json)

        usage = response_json["usage"]
        
        return {
            "latency": latency,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"]
        }
        
    async def run_concurrent_tests(self):
        """Run concurrent inference tests."""
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            tasks = [self.single_request(session) for _ in range(self.num_requests)]
            self.results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            self.total_test_time = end_time - start_time
    
    def calculate_metrics(self):
        """Calculate the metrics of the test results."""
        total_tokens = sum(result["total_tokens"] for result in self.results)
        prompt_tokens = sum(result["prompt_tokens"] for result in self.results)
        completion_tokens = sum(result["completion_tokens"] for result in self.results)
        
        latencies = [result["latency"] for result in self.results]
        
        return {
            "total_time": self.total_test_time,
            "throughput_requests": self.num_requests / self.total_test_time,
            "avg_latency": statistics.mean(latencies),
            "avg_latency_per_token": sum(latencies) / total_tokens,
            "avg_latency_per_completion_token": sum(latencies) / completion_tokens,
            "throughput_tokens": total_tokens / self.total_test_time,
            "avg_prompt_tokens": prompt_tokens / self.num_requests,
            "avg_completion_tokens": completion_tokens / self.num_requests,
            "avg_total_tokens": total_tokens / self.num_requests
        }
        
        
    def run_test(self):
        """Run the full inference test and print results."""
        asyncio.run(self.run_concurrent_tests())
        metrics = self.calculate_metrics()

        print(f"Test Results:")
        print(f"1. Total time: {metrics['total_time']:.6f} s")
        print(f"2. Throughput: {metrics['throughput_requests']:.2f} requests/s")
        print(f"3. Average latency: {metrics['avg_latency']:.6f} ms")
        print(f"4. Average latency per token: {metrics['avg_latency_per_token']:.6f} ms")
        print(f"5. Average latency per completion token: {metrics['avg_latency_per_completion_token']:.6f} ms")
        print(f"6. Throughput: {metrics['throughput_tokens']:.2f} tokens/s")
        print(f"7. Average prompt tokens: {metrics['avg_prompt_tokens']:.2f}")
        print(f"8. Average completion tokens: {metrics['avg_completion_tokens']:.2f}")
        print(f"9. Average total tokens: {metrics['avg_total_tokens']:.2f}")


if __name__ == "__main__":
    import os

    # Example prompts
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

    # Configuration
    model = "gpt-4o-mini"
    num_requests = 10
    max_concurrency = 32

    #base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    
    base_url = "https://api.xi-ai.cn"
    
    # Initialize and run the test
    test = LLMInferenceTest(
        endpoint = f"{base_url}/v1/chat/completions",
        api_key="sk-4J6sXggR7YdslyOtAe9c0aEa72654cDf92A42fDc012031A4",
        prompts=prompts,
        model=model,
        num_requests=num_requests,
        max_concurrency=max_concurrency
    )
    test.run_test()