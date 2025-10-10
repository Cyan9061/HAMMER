# Simple Token Tracking System
# 替代复杂的instrumentation系统，直接追踪LLM API调用的token使用情况

from dataclasses import dataclass
from typing import List, Dict, Any
import threading
from hammer.logger import logger

@dataclass
class SimpleTokenUsage:
    """简单的token使用记录"""
    llm_name: str
    total_tokens: int
    call_type: str  # "agent" 或 "rag"

# 全局token记录列表 - 线程安全
_token_lock = threading.Lock()
GLOBAL_TOKEN_USAGE: List[SimpleTokenUsage] = []

def record_token_usage(llm_name: str, total_tokens: int = 0, additional_info: Dict[str, Any] = None):
    """
    记录token使用情况
    
    Args:
        llm_name: 模型名称
        total_tokens: 总token数
        additional_info: 额外信息(可选)
    """
    # 根据模型名称判断调用类型
    call_type = "agent" if "gpt-4o-mini" in llm_name.lower() else "rag"
    
    usage = SimpleTokenUsage(
        llm_name=llm_name,
        total_tokens=total_tokens,
        call_type=call_type
    )
    
    with _token_lock:
        GLOBAL_TOKEN_USAGE.append(usage)
    
    # 🔥 强制使用INFO级别，确保token记录被显示
    logger.info(f"🔢 [TOKEN_DEBUG] 记录token使用: {llm_name} - {total_tokens} tokens - {call_type}")
    logger.info(f"🔢 [TOKEN_DEBUG] 当前全局token记录总数: {len(GLOBAL_TOKEN_USAGE)}")

def record_llm_response(response, call_type: str = "rag", llm_name: str = None):
    """
    从通用LLM响应中提取并记录token使用情况
    
    Args:
        response: LLM API响应对象
        call_type: 调用类型 ("agent" 或 "rag")  
        llm_name: 模型名称(可选)
    """
    try:
        # 尝试从响应中推断LLM名称
        if llm_name is None:
            if hasattr(response, 'model'):
                llm_name = response.model
            else:
                llm_name = "unknown_llm"
        
        # 尝试OpenAI格式
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            if total_tokens == 0:  # 如果total_tokens为0，手动计算
                total_tokens = getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
            return
        
        # 如果没有usage信息，记录错误
        logger.error(f"⚠️ 无法从LLM响应中提取token信息: {type(response)}")
        raise ValueError(f"无法从LLM响应中提取token信息: {type(response)}")
            
    except Exception as e:
        logger.error(f"❌ 记录通用LLM token失败: {e}")

def record_openai_response(response, llm_name: str):
    """
    从OpenAI格式的响应中提取并记录token使用情况
    
    Args:
        response: OpenAI API响应对象
        llm_name: 模型名称
    """
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            if total_tokens == 0:  # 如果total_tokens为0，手动计算
                total_tokens = getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
        else:
            logger.warning(f"⚠️ OpenAI响应缺少usage信息: {llm_name}")
    except Exception as e:
        logger.error(f"❌ 记录OpenAI token失败: {llm_name} - {e}")

def record_siliconflow_response(response_json: dict, llm_name: str):
    """
    从SiliconFlow API响应中提取并记录token使用情况
    
    Args:
        response_json: SiliconFlow API响应JSON
        llm_name: 模型名称
    """
    try:
        if 'usage' in response_json:
            usage = response_json['usage']
            total_tokens = usage.get('total_tokens', 0)
            if total_tokens == 0:  # 如果total_tokens为0，手动计算
                total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
        else:
            logger.warning(f"⚠️ SiliconFlow响应缺少usage信息: {llm_name}")
    except Exception as e:
        logger.error(f"❌ 记录SiliconFlow token失败: {llm_name} - {e}")

def get_token_statistics():
    """
    获取token使用统计
    
    Returns:
        Tuple[Dict, Dict]: (agent_tokens, rag_tokens)
    """
    with _token_lock:
        agent_tokens = {"total": 0, "calls": 0}
        rag_tokens = {"total": 0, "calls": 0}
        
        for usage in GLOBAL_TOKEN_USAGE:
            if usage.call_type == "agent":
                agent_tokens["total"] += usage.total_tokens
                agent_tokens["calls"] += 1
            else:
                rag_tokens["total"] += usage.total_tokens
                rag_tokens["calls"] += 1
        
        return agent_tokens, rag_tokens

def clear_token_usage():
    """清空token使用记录"""
    with _token_lock:
        GLOBAL_TOKEN_USAGE.clear()
    logger.debug("🧹 Token使用记录已清空")

def get_debug_info():
    """获取调试信息"""
    with _token_lock:
        total_calls = len(GLOBAL_TOKEN_USAGE)
        agent_calls = sum(1 for u in GLOBAL_TOKEN_USAGE if u.call_type == "agent")
        rag_calls = sum(1 for u in GLOBAL_TOKEN_USAGE if u.call_type == "rag")
        
        return {
            "total_calls": total_calls,
            "agent_calls": agent_calls,
            "rag_calls": rag_calls,
            "recent_calls": GLOBAL_TOKEN_USAGE[-5:] if GLOBAL_TOKEN_USAGE else []
        }

def print_debug_info():
    """打印调试信息"""
    info = get_debug_info()
    logger.info(f"🔢 Token追踪调试:")
    logger.info(f"  总调用数: {info['total_calls']}")
    logger.info(f"  Agent调用: {info['agent_calls']}")
    logger.info(f"  RAG调用: {info['rag_calls']}")
    if info['recent_calls']:
        logger.info(f"  最近5次调用:")
        for usage in info['recent_calls']:
            logger.info(f"    {usage.llm_name}: {usage.total_tokens} tokens ({usage.call_type})")