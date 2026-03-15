
"""
SiliconFlow-backed LLM implementation with thread-safe API key rotation.
"""

import time
import itertools
import os
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from hammer.logger import logger

class SiliconFlowLLM(CustomLLM):
    """
    SiliconFlow API implementation with thread-safe key rotation.
    Effective concurrency scales with the number of configured API keys.
    """
    
    # API configuration
    api_keys: Sequence[str] = Field(description="List of SiliconFlow API keys")
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="Model name")
    api_endpoint: str = Field(default="https://api.siliconflow.cn/v1/chat/completions", description="API endpoint")
    max_tokens: int = Field(default=8192, description="Maximum output tokens")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    _key_cycler: Any = PrivateAttr()
    _key_lock: Lock = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize key rotation state via private attrs rather than Pydantic fields.
        if self.api_keys:
            self._key_cycler = itertools.cycle(self.api_keys)
            self._key_lock = Lock()
        else:
            raise ValueError("api_keys must not be empty")
    
    @property 
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Tune per model if needed.
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model_name,
        )
    
    def _get_next_api_key(self) -> str:
        """Return the next API key in a thread-safe way."""
        with self._key_lock:
            return next(self._key_cycler)
    
    def _make_api_request(self, messages: list, api_key: str) -> dict:
        """Execute a single API request."""
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
            
            # Record token usage in the shared tracker.
            from hammer.utils.simple_token_tracker import record_siliconflow_response
            record_siliconflow_response(response_json, self.model_name)
            
            return response_json
        elif response.status_code in [429, 500, 502, 503, 504]:
            # Retryable server-side or throttling errors.
            raise requests.exceptions.RequestException(
                f"API Rate Limit/Server Error [{response.status_code}]: {response.text[:200]}"
            )
        else:
            # Non-retryable client or protocol errors.
            raise ValueError(f"API request failed [{response.status_code}]: {response.text[:200]}")
    
    def _predict_with_retry(self, messages: list) -> str:
        """Run prediction with retry handling."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Rotate across available API keys.
                api_key = self._get_next_api_key()
                
                # Execute the request.
                result = self._make_api_request(messages, api_key)
                
                # Extract the response payload.
                response_content = result['choices'][0]['message']['content'].strip()
                
                logger.debug(
                    "SiliconFlow request succeeded with key %s... on attempt %s",
                    api_key[:20],
                    attempt + 1,
                )
                return response_content
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Use a longer backoff window for rate-limit errors.
                    if "429" in str(e) or "TPM limit" in str(e) or "rate limit" in str(e).lower():
                        sleep_time = min(10 + (5 * attempt), 60)
                        logger.warning(
                            "Rate limit detected; retrying in %.2fs (%s/%s): %s",
                            sleep_time,
                            attempt + 1,
                            self.max_retries,
                            e,
                        )
                    else:
                        sleep_time = 1.5 ** attempt
                        logger.warning(
                            "API request failed; retrying in %.2fs (%s/%s): %s",
                            sleep_time,
                            attempt + 1,
                            self.max_retries,
                            e,
                        )
                    time.sleep(sleep_time)
                else:
                    logger.error("API request failed after %s retries: %s", self.max_retries, e)
                    
            except Exception as e:
                last_exception = e
                logger.error("Unexpected API request failure: %s", e)
                break
        
        # Return a structured error string after all retries are exhausted.
        error_msg = f"API request failed: {last_exception}"
        logger.error(error_msg)
        return f"[API request failed] {error_msg}"
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Run completion generation in the llama-index compatible format."""
        # Convert to chat-style payload.
        messages = [{"role": "user", "content": prompt}]
        
        # Execute the request.
        response_text = self._predict_with_retry(messages)
        
        return CompletionResponse(text=response_text)
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Run chat generation in the llama-index compatible format."""
        # Convert message objects to API payloads.
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Execute the request.
        response_text = self._predict_with_retry(api_messages)
        
        # Return ChatResponse directly to avoid callback/Pydantic issues.
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text)
        )
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Streaming completion is not supported."""
        raise NotImplementedError("SiliconFlow LLM does not support streaming output")
    
    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """Streaming chat is not supported."""
        raise NotImplementedError("SiliconFlow LLM does not support streaming output")

def create_siliconflow_llm(
    api_keys: Sequence[str],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
    max_tokens: int = 2000,
    temperature: float = 0.7,
    **kwargs
) -> SiliconFlowLLM:
    """
    Convenience factory for SiliconFlowLLM.
    
    Args:
        api_keys: API keys
        model_name: Model name
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        **kwargs: Additional keyword arguments
    
    Returns:
        SiliconFlowLLM instance
    """
    return SiliconFlowLLM(
        api_keys=api_keys,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

def get_api_keys():
    """Read SiliconFlow API keys from environment variables first."""

    def _split_env(value: str) -> list[str]:
        return [key.strip() for key in value.split(",") if key.strip()]

    keys: list[str] = []
    for env_var in ("SILICONFLOW_API_KEYS", "SILICONFLOW_API_KEY", "ADDITIONAL_API_KEYS"):
        env_value = os.getenv(env_var, "")
        if env_value:
            keys.extend(_split_env(env_value))

    deduped: list[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)

    if deduped:
        return deduped

    logger.warning("No SiliconFlow API keys detected; using a placeholder key to keep imports working.")
    return ["EMPTY_SILICONFLOW_API_KEY"]
