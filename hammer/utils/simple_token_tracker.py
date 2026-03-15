# Simple Token Tracking System
# Lightweight replacement for the heavier instrumentation stack; tracks token usage directly from LLM API calls.

from dataclasses import dataclass
from typing import List, Dict, Any
import threading
from hammer.logger import logger

@dataclass
class SimpleTokenUsage:
    """Simple token-usage record."""
    llm_name: str
    total_tokens: int
    call_type: str  # "agent" or "rag"

# Thread-safe global token usage list.
_token_lock = threading.Lock()
GLOBAL_TOKEN_USAGE: List[SimpleTokenUsage] = []

def record_token_usage(llm_name: str, total_tokens: int = 0, additional_info: Dict[str, Any] = None):
    """
    Record token usage.
    
    Args:
        llm_name: Model name
        total_tokens: Total token count
        additional_info: Optional extra metadata
    """
    # Infer call type from the model name.
    call_type = "agent" if "gpt-4o-mini" in llm_name.lower() else "rag"
    
    usage = SimpleTokenUsage(
        llm_name=llm_name,
        total_tokens=total_tokens,
        call_type=call_type
    )
    
    with _token_lock:
        GLOBAL_TOKEN_USAGE.append(usage)
    
    # Use INFO level so token tracking is always visible in logs.
    logger.info("[TOKEN_DEBUG] Recorded token usage: %s - %s tokens - %s", llm_name, total_tokens, call_type)
    logger.info("[TOKEN_DEBUG] Global token record count: %s", len(GLOBAL_TOKEN_USAGE))

def record_llm_response(response, call_type: str = "rag", llm_name: str = None):
    """
    Extract and record token usage from a generic LLM response.
    
    Args:
        response: LLM API response object
        call_type: Call type ("agent" or "rag")
        llm_name: Optional model name
    """
    try:
        # Try to infer the LLM name from the response.
        if llm_name is None:
            if hasattr(response, 'model'):
                llm_name = response.model
            else:
                llm_name = "unknown_llm"
        
        # Handle OpenAI-style responses.
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            if total_tokens == 0:
                total_tokens = getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
            return
        
        # Error if no usage metadata is available.
        logger.error("Could not extract token information from LLM response: %s", type(response))
        raise ValueError(f"Could not extract token information from LLM response: {type(response)}")
            
    except Exception as e:
        logger.error("Failed to record generic LLM tokens: %s", e)

def record_openai_response(response, llm_name: str):
    """
    Extract and record token usage from an OpenAI-style response.
    
    Args:
        response: OpenAI API response object
        llm_name: Model name
    """
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            total_tokens = getattr(usage, 'total_tokens', 0)
            if total_tokens == 0:
                total_tokens = getattr(usage, 'prompt_tokens', 0) + getattr(usage, 'completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
        else:
            logger.warning("OpenAI response is missing usage info: %s", llm_name)
    except Exception as e:
        logger.error("Failed to record OpenAI tokens for %s: %s", llm_name, e)

def record_siliconflow_response(response_json: dict, llm_name: str):
    """
    Extract and record token usage from a SiliconFlow API response.
    
    Args:
        response_json: SiliconFlow API response JSON
        llm_name: Model name
    """
    try:
        if 'usage' in response_json:
            usage = response_json['usage']
            total_tokens = usage.get('total_tokens', 0)
            if total_tokens == 0:
                total_tokens = usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
            record_token_usage(llm_name=llm_name, total_tokens=total_tokens)
        else:
            logger.warning("SiliconFlow response is missing usage info: %s", llm_name)
    except Exception as e:
        logger.error("Failed to record SiliconFlow tokens for %s: %s", llm_name, e)

def get_token_statistics():
    """
    Return token usage statistics.
    
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
    """Clear all token usage records."""
    with _token_lock:
        GLOBAL_TOKEN_USAGE.clear()
    logger.debug("Token usage records cleared")

def get_debug_info():
    """Return token-tracking debug information."""
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
    """Log token-tracking debug information."""
    info = get_debug_info()
    logger.info("Token tracking debug info:")
    logger.info("  Total calls: %s", info['total_calls'])
    logger.info("  Agent calls: %s", info['agent_calls'])
    logger.info("  RAG calls: %s", info['rag_calls'])
    if info['recent_calls']:
        logger.info("  Recent 5 calls:")
        for usage in info['recent_calls']:
            logger.info("    %s: %s tokens (%s)", usage.llm_name, usage.total_tokens, usage.call_type)
