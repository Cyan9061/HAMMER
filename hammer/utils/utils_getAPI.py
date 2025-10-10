import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 🔧 SiliconFlow API 鉴权配置
SILICONFLOW_API_KEYS = os.getenv("SILICONFLOW_API_KEYS", "").split(",") if os.getenv("SILICONFLOW_API_KEYS") else []

# 🔧 gaochao API 鉴权配置
GAOCHAO_API_KEYS = [os.getenv("OPENAI_API_KEY", "")]

# 🔧 模型特定的API key映射（优先级最高）
MODEL_SPECIFIC_API_KEYS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": os.getenv("ADDITIONAL_API_KEYS", "").split(",") if os.getenv("ADDITIONAL_API_KEYS") else [],
    "Qwen/Qwen2.5-72B-Instruct": os.getenv("ADDITIONAL_API_KEYS", "").split(",") if os.getenv("ADDITIONAL_API_KEYS") else []
}

# 🔧 模型到API提供商的映射
MODEL_PROVIDER_MAPPING = {
    "gpt-4o-mini": "gaochao",
    "gpt-4o": "gaochao", 
    "gpt-4": "gaochao",
    "gpt-3.5-turbo": "gaochao",
}

# 向后兼容
API_KEYS = SILICONFLOW_API_KEYS

def get_api_keys():
    """获取API keys列表（向后兼容）"""
    return SILICONFLOW_API_KEYS

def get_api_keys_for_model(model_name: str) -> list:
    """根据模型名称获取API keys，优先使用模型特定配置"""
    if model_name in MODEL_SPECIFIC_API_KEYS:
        return MODEL_SPECIFIC_API_KEYS[model_name]
    else:
        # 回退到原有逻辑
        provider = get_provider_for_model(model_name)
        return get_api_keys_for_provider(provider)

def get_provider_for_model(model_name: str) -> str:
    """根据模型名称获取API提供商"""
    return MODEL_PROVIDER_MAPPING.get(model_name, "siliconflow")

def get_api_keys_for_provider(provider: str) -> list:
    """根据提供商获取对应的API keys"""
    if provider == "gaochao":
        return GAOCHAO_API_KEYS
    elif provider == "siliconflow":
        return SILICONFLOW_API_KEYS
    else:
        return SILICONFLOW_API_KEYS

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from threading import Lock
import itertools 

# 🔧 API端点配置
API_ENDPOINTS = {
    "siliconflow": "https://api.siliconflow.cn/v1/chat/completions",
    "gaochao": "https://api.ai-gaochao.cn/v1/chat/completions",
}

# 向后兼容
API_ENDPOINT = API_ENDPOINTS["siliconflow"]

def predict_single(input_text: str, api_key: str, model_name: str, max_retries: int = 3) -> tuple:
    """
    使用指定的 API 密钥进行单条预测（带重试机制），自动选择API提供商。

    参数:
        input_text: 完整的问题 prompt。
        api_key: 用于请求的 API 密钥。
        model_name: 要调用的模型名称。
        max_retries: 最大重试次数。

    返回:
        一个元组 (input_text, response_content, latency, token_usage)。
        如果失败，response_content 和 latency 为 None。
    """
    # 🔧 根据模型名称自动选择API提供商和端点
    provider = get_provider_for_model(model_name)
    endpoint = API_ENDPOINTS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": input_text}],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 0.9,
    }

    start_time = time.time()

    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                end_time = time.time()
                latency = round((end_time - start_time) * 1000)
                result = response.json()
                
                response_content = result['choices'][0]['message']['content'].strip()
                token_usage = result.get('usage', {}).get('total_tokens', 0)
                
                return (input_text, response_content, latency, token_usage)
            
            elif response.status_code in [429, 500, 502, 503, 504]:
                 raise Exception(f"API Rate Limit/Server Error [{response.status_code}]: {response.text[:200]}")
            
            else:
                print(f"不可重试的错误 for '{input_text[:30]}...': [{response.status_code}] {response.text[:200]}")
                return (input_text, None, None, None)

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                if "429" in str(e) or "TPM limit" in str(e) or "rate limit" in str(e).lower():
                    sleep_time = min(10 + (5 * attempt), 60)
                    print(f"检测到限流错误 for '{input_text[:30]}...'. Error: {e}. 将在 {sleep_time:.2f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                else:
                    sleep_time = 1.5 ** attempt
                    print(f"请求失败 for '{input_text[:30]}...'. Error: {e}. 将在 {sleep_time:.2f} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                time.sleep(sleep_time)
            else:
                print(f"'{input_text[:30]}...' 的所有重试均失败。")
                return (input_text, None, None, None)
    
    return (input_text, None, None, None)

def batch_predict(input_texts: list, model_name: str, max_workers: int = None) -> list:
    """
    多线程批量预测函数，动态分配 API 密钥，并显示进度。
    """
    api_keys = get_api_keys_for_model(model_name)
    provider = get_provider_for_model(model_name)
    
    if not api_keys or not all(api_keys):
        raise ValueError(f"模型 '{model_name}' 的密钥列表为空或包含无效值。请先配置您的 API 密钥。")

    # 自动设置并发数
    if max_workers is None:
        max_workers = 4 * len(api_keys)
    
    # 检查是否使用模型特定API keys
    is_model_specific = model_name in MODEL_SPECIFIC_API_KEYS
    key_type = "模型特定" if is_model_specific else "默认"
    
    print(f"🔧 使用{key_type}API keys | 提供商: {provider} | 模型: {model_name} | 可用密钥: {len(api_keys)} | 并发数: {max_workers}")
    if is_model_specific:
        print(f"🔑 模型 {model_name} 使用专用API keys")

    results = [None] * len(input_texts)
    
    # 使用 itertools.cycle 创建一个线程安全的密钥循环器
    key_cycler = itertools.cycle(api_keys)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将 future 对象直接映射到其原始索引 i，而不是文本内容
        futures = {
            executor.submit(predict_single, text, next(key_cycler), model_name): i
            for i, text in enumerate(input_texts)
        }

        # 使用 tqdm 创建进度条
        with tqdm(total=len(input_texts), desc=f"使用模型 {model_name}", unit="req") as pbar:
            for future in as_completed(futures):
                # 直接从 futures 字典中获取原始索引
                original_index = futures[future]
                original_text = input_texts[original_index]
                
                try:
                    result = future.result()
                    results[original_index] = result
                except Exception as e:
                    print(f"处理任务(索引: {original_index})时发生未预料的错误: {e}")
                    # 即使有异常，也填充一个失败结果以保持列表长度
                    results[original_index] = (original_text, None, None, None)
                finally:
                    # 每次有任务完成（无论成功失败）都更新进度条
                    pbar.update(1)
    
    return results

# --- 主程序入口 ---

if __name__ == "__main__":
    # --- 测试配置 ---
    MODEL_TO_TEST = "Qwen/Qwen2.5-7B-Instruct"
    NUM_QUESTIONS = 10000
    CONCURRENCY_LEVEL = len(API_KEYS) * 3

    # --- 生成测试问题 ---
    print(f"正在生成 {NUM_QUESTIONS} 个测试问题...")
    test_queries = [f"Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No. "+
"Now complete the following example -"+
"input: PROPERTY: - 'has a topic of technology and business mergers'"+
"TEXT: UN Council Wants Sudan Peace Deal by End of Year (Reuters). Reuters - The U.N. Security Council, on a high-profile visit to the Kenyan capital, expects Sudan and its southern opposition on Thursday to promise to complete by Dec. 31 a comprehensive peace agreement ending a 21-year civil war." for i in range(NUM_QUESTIONS)]

    print("-" * 60)
    print(f"测试开始...")
    print(f"模型: {MODEL_TO_TEST}")
    print(f"问题总数: {NUM_QUESTIONS}")
    print(f"并发数: {CONCURRENCY_LEVEL}")
    print(f"可用API密钥数: {len(API_KEYS)}")
    if len(API_KEYS) < CONCURRENCY_LEVEL:
        print("警告: API密钥数量小于并发数，可能导致密钥复用频繁，影响速率。")
    print("-" * 60)

    # 记录总耗时
    total_start_time = time.time()

    # 执行批量预测
    predictions = batch_predict(
        input_texts=test_queries,
        model_name=MODEL_TO_TEST,
        max_workers=CONCURRENCY_LEVEL
    )

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # --- 统计与分析结果 ---
    successful_requests = 0
    failed_requests = 0
    total_tokens_processed = 0
    total_latency = 0

    for _, response, latency, tokens in predictions:
        if response is not None and latency is not None:
            successful_requests += 1
            total_tokens_processed += tokens
            total_latency += latency
        else:
            failed_requests += 1

    print("\n" + "=" * 60)
    print("测试完成！结果分析：")
    print("=" * 60)
    
    print(f"总耗时: {total_duration:.2f} 秒")
    print(f"成功请求: {successful_requests} / {NUM_QUESTIONS}")
    print(f"失败请求: {failed_requests} / {NUM_QUESTIONS}")
    
    if successful_requests > 0:
        avg_latency = total_latency / successful_requests
        requests_per_second = successful_requests / total_duration
        avg_tokens_per_request = total_tokens_processed / successful_requests
        tokens_per_second = total_tokens_processed / total_duration
        
        print(f"平均延迟 (端到端): {avg_latency:.2f} ms")
        print(f"每秒请求数 (RPS): {requests_per_second:.2f} req/s")
        print(f"处理总 Token 数: {total_tokens_processed} tokens")
        print(f"平均每秒 Token 数 (TPS): {tokens_per_second:.2f} tokens/s")
        print(f"平均每次请求 Token 数: {avg_tokens_per_request:.2f} tokens/req")

    print("-" * 60)
    
    # 打印前几个结果以供查阅
    print("前5个请求的结果示例：\n")
    for i, (query, response, latency, tokens) in enumerate(predictions[:5]):
        print(f"--- 示例 {i+1} ---")
        print(f"输入: {query[:50]}...")
        if response:
            print(f"响应: {response[:100].replace(os.linesep, ' ')}...")
            print(f"状态: 成功 | 延迟: {latency} ms | Tokens: {tokens}")
        else:
            print("响应: 请求失败")
        print()