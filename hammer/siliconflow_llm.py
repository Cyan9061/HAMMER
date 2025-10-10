
"""
SiliconFlow API集成的LLM实现
支持线程安全的API key轮询分配，实现高并发调用
"""

import time
import itertools
import requests
import threading
from typing import Any, Dict, Optional, Sequence
from threading import Lock

from llama_index.core.base.llms.types import (
    ChatMessage, 
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import CustomLLM
from llama_index.core.bridge.pydantic import Field

from hammer.logger import logger

class SiliconFlowLLM(CustomLLM):
    """
    SiliconFlow API的LLM实现，支持线程安全的API key轮询分配
    并行度为API key数量 × 3
    """
    
    # API配置
    api_keys: Sequence[str] = Field(description="SiliconFlow API keys列表")
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="模型名称")
    api_endpoint: str = Field(default="https://api.siliconflow.cn/v1/chat/completions", description="API端点")
    max_tokens: int = Field(default=8192, description="最大token数")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.9, description="top_p参数")
    max_retries: int = Field(default=3, description="最大重试次数")
    timeout: int = Field(default=60, description="请求超时时间（秒）")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化API key循环器（使用私有属性，不作为Field）
        if self.api_keys:
            self._key_cycler = itertools.cycle(self.api_keys)
            self._key_lock = Lock()  # 初始化锁
        else:
            raise ValueError("api_keys不能为空")
    
    @property 
    def metadata(self) -> LLMMetadata:
        """返回LLM元数据"""
        return LLMMetadata(
            context_window=4096,  # 根据模型调整
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model_name,
        )
    
    def _get_next_api_key(self) -> str:
        """线程安全地获取下一个API key"""
        with self._key_lock:
            return next(self._key_cycler)
    
    def _make_api_request(self, messages: list, api_key: str) -> dict:
        """发起单次API请求"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        response = requests.post(
            self.api_endpoint, 
            json=payload, 
            headers=headers, 
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            response_json = response.json()
            
            # 🔥 记录token使用到全局统计
            from hammer.utils.simple_token_tracker import record_siliconflow_response
            record_siliconflow_response(response_json, self.model_name)
            
            return response_json
        elif response.status_code in [429, 500, 502, 503, 504]:
            # 可重试错误
            raise requests.exceptions.RequestException(
                f"API Rate Limit/Server Error [{response.status_code}]: {response.text[:200]}"
            )
        else:
            # 不可重试错误
            raise ValueError(f"API请求失败 [{response.status_code}]: {response.text[:200]}")
    
    def _predict_with_retry(self, messages: list) -> str:
        """带重试机制的预测"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # 获取API key
                api_key = self._get_next_api_key()
                
                # 发起请求
                result = self._make_api_request(messages, api_key)
                
                # 提取响应
                response_content = result['choices'][0]['message']['content'].strip()
                
                logger.debug(f"API调用成功，使用key: {api_key[:20]}...，尝试次数: {attempt + 1}")
                return response_content
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # 🔥 针对429限流错误使用更长的退避时间
                    if "429" in str(e) or "TPM limit" in str(e) or "rate limit" in str(e).lower():
                        sleep_time = min(10 + (5 * attempt), 60)  # 10s → 15s → 20s，最大60s
                        logger.warning(f"检测到限流错误，{sleep_time:.2f}秒后重试 (第{attempt + 1}/{self.max_retries}次): {e}")
                    else:
                        sleep_time = 1.5 ** attempt  # 指数退避
                        logger.warning(f"API请求失败，{sleep_time:.2f}秒后重试 (第{attempt + 1}/{self.max_retries}次): {e}")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API请求重试{self.max_retries}次均失败: {e}")
                    
            except Exception as e:
                last_exception = e
                logger.error(f"API请求发生意外错误: {e}")
                break
        
        # 如果所有重试都失败，返回错误信息
        error_msg = f"API调用失败: {last_exception}"
        logger.error(error_msg)
        return f"[API调用失败] {error_msg}"
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """完成文本生成（兼容llama_index接口）"""
        # 转换为chat格式
        messages = [{"role": "user", "content": prompt}]
        
        # 调用API
        response_text = self._predict_with_retry(messages)
        
        return CompletionResponse(text=response_text)
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """聊天模式（兼容llama_index接口）"""
        # 转换消息格式
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # 调用API
        response_text = self._predict_with_retry(api_messages)
        
        # 直接返回ChatResponse，不使用callback装饰器避免pydantic错误
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text)
        )
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """流式完成（暂不支持）"""
        raise NotImplementedError("SiliconFlow LLM不支持流式输出")
    
    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """流式聊天（暂不支持）"""
        raise NotImplementedError("SiliconFlow LLM不支持流式输出")

def create_siliconflow_llm(
    api_keys: Sequence[str],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
    max_tokens: int = 2000,
    temperature: float = 0.7,
    **kwargs
) -> SiliconFlowLLM:
    """
    创建SiliconFlow LLM实例的便捷函数
    
    Args:
        api_keys: API密钥列表
        model_name: 模型名称
        max_tokens: 最大token数
        temperature: 温度参数
        **kwargs: 其他参数
    
    Returns:
        SiliconFlowLLM实例
    """
    return SiliconFlowLLM(
        api_keys=api_keys,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

# 从utils_getAPI.py导入API keys
def get_api_keys():
    """获取API keys"""
    return [
        "sk-syzfeyvsvrurqpbwpiwtwjbcmlkhtojngngopcqmasmmtfmz",
        "sk-iaomffrqmsrqpupozhfrtfqnjqllormgheprxcxjuufoquaq",
        "sk-nmdoohpnnawgpsbfutuolbqpqddtsqqhhsrdyzzsnqsfunuy",
        "sk-fkvkjzfejsfcnzolxybcnbeubodbcxbczbuqnhtnqpfgyurp",
        "sk-lavogswyitrwhyfxywaylfvhjwchqppcnhauoeouypigiaaf",
        "sk-kpacfuoklmioauxqlqrpityhhbjarjqcpiknxleuvizduyxm",
        "sk-sktrhqdgufgooboqbcvriatgpciockywxbrsqngsjzjehchh",
        "sk-wkapiwrmbvxlwnwffjvzeausdlowdiohogdubzzkzgqgrddh",
        "sk-piyqetdwcqqlqsgprbmmvmnetaqqmnorrmzdsqznmtzsznxd",
        "sk-szbhnnmhpmfkkduuvymznhhewcfpgajyggzpjngbkgsbhjcv",
        "sk-fxanyzqjnqjwevqrcqclpdekwfslpkyljjrvvelnmipgdotl",
        "sk-ziseitejknrdtjybqrmsrhohoyloprglrskjoncwkhwcoqdd",
        "sk-gegmwncxlqhgmvdoyrbytftvxmkekklqukpqdgwnynijskmz",
        "sk-svcbtepfhczvctugfiboqumahybnwgfajjygncwvnwnuadqs",
        "sk-gvgrtjtbehntvavqwwvyyswntlcmcnpjnvvazxmqpolnfazg",
        "sk-qrmkkjcwuitbnqjqwwwqsvmgettmrmtfrkidxukfdkzcshpf",
        "sk-kjrzdicvsosineegefgrpjeuqbvbmqfxhspcltbwfwpkznxy",
        "sk-xqxjkftqpeacuzdresfszqcpjhmvashguthcxqggtxvhnbie",
        "sk-sygrwnulmhdkqyykgweqshsonvhdemrytyzqtkawxbwbkhok",
        "sk-tdvqsrvotngseasqtwvvdowiaozwafuqiohenoatwyqvttyz",
    ]