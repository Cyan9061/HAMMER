
# 版本：2.0 - 轻量级Token统计器
# 描述：移除了所有成本计算逻辑，专注于Token统计，并与解耦的评估架构兼容。

import inspect
import time
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass

from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from openinference.instrumentation.llama_index._handler import (
    EventHandler,
    _Span,
    _SpanHandler,
    _get_token_counts,
    _init_span_kind,
    _SUPPRESS_INSTRUMENTATION_KEY,
    context_api,
    get_attributes_from_context,
    time_ns,
    INPUT_VALUE,
    OUTPUT_VALUE,
    LLM_MODEL_NAME,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_TOTAL,
)
from opentelemetry.trace import NoOpTracer
from pydantic import PrivateAttr

from hammer.logger import logger

# =============================================================================
# 全局数据容器
# =============================================================================

@dataclass
class LLMTokenUsageData:
    """一个简单的数据类，用于存储单次LLM调用的Token使用情况。"""
    llm_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency: float

# 全局列表，用于在评估期间临时存储所有LLM调用的Token使用数据
GLOBAL_LLM_USAGE_DATA: List[LLMTokenUsageData] = []

# =============================================================================
# 核心监控逻辑
# =============================================================================

class TokenTrackingSpan(_Span):
    """自定义的Span类，专注于捕获和处理Token统计。"""
    _instance: Any = PrivateAttr()
    _bound_args: inspect.BoundArguments = PrivateAttr()

    def __init__(
        self,
        *args,
        instance: Any,
        bound_args: inspect.BoundArguments,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._instance = instance
        self._bound_args = bound_args

    def process_event(self, event: BaseEvent, **kwargs: Any) -> None:
        """
        处理LLM事件的核心方法。
        新逻辑：计算Token使用情况，并直接添加到全局列表中。
        """
        super().process_event(event)

        if isinstance(event, (LLMChatStartEvent, LLMCompletionStartEvent)):
            self["call_start"] = time.time()
            return

        if not isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            return

        self["call_end"] = time.time()
        latency = self._attributes.get("call_end", 0) - self._attributes.get("call_start", 0)

        # 从事件属性中安全地获取token统计信息，如果不存在则默认为0
        input_tokens = self._attributes.get(LLM_TOKEN_COUNT_PROMPT, 0)
        output_tokens = self._attributes.get(LLM_TOKEN_COUNT_COMPLETION, 0)
        total_tokens = self._attributes.get(LLM_TOKEN_COUNT_TOTAL, 0)
        
        # 如果total_tokens为0但分项不为0，则手动计算
        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        usage_data = LLMTokenUsageData(
            llm_name=self._attributes.get(LLM_MODEL_NAME, "unknown_model"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency=latency,
        )
        
        # 无论在何处调用，都将数据安全地添加到全局列表
        GLOBAL_LLM_USAGE_DATA.append(usage_data)

    def _has_token_counts(self) -> bool:
        """检查是否已从响应中提取了token计数。"""
        return all(
            key in self._attributes
            for key in [
                LLM_TOKEN_COUNT_PROMPT,
                LLM_TOKEN_COUNT_COMPLETION,
            ]
        )

    def _extract_token_counts(
        self, response: Union[ChatResponse, CompletionResponse]
    ) -> None:
        """
        从不同格式的LLM响应中尽力提取token计数。
        已移除所有与成本（字符数、时长）相关的回退逻辑。
        """
        super()._extract_token_counts(response)
        if self._has_token_counts():
            return

        # 尝试从response.raw.usage中提取
        if (raw := getattr(response, "raw", None)) and (usage := getattr(raw, "usage", None)):
            for k, v in _get_token_counts(usage):
                self[k] = v
            if self._has_token_counts():
                return

        logger.warning(
            f"未能从模型 '{self._attributes.get(LLM_MODEL_NAME)}' 的响应中提取Token计数。 "
            f"响应类型: {type(response)}. 统计将记为0。"
        )

class TokenTrackingSpanHandler(_SpanHandler):
    """自定义的Span处理器，确保使用我们的TokenTrackingSpan。"""
    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[_Span]:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        with self.lock:
            parent = self.open_spans.get(parent_span_id) if parent_span_id else None
        otel_span = self._otel_tracer.start_span(
            name=id_.partition("-")[0],
            start_time=time_ns(),
            attributes=dict(get_attributes_from_context()),
            context=(parent.context if parent else None),
        )
        span = TokenTrackingSpan(
            otel_span=otel_span,
            span_kind=_init_span_kind(instance),
            parent=parent,
            id_=id_,
            parent_id=parent_span_id,
            instance=instance,
            bound_args=bound_args,
        )
        span.process_instance(instance)
        span.process_input(instance, bound_args)
        return span

class TokenTrackingEventHandler(EventHandler):
    """事件处理器，使用我们自定义的SpanHandler来覆盖默认行为。"""
    def __init__(self, **kwargs) -> None:
        super().__init__(tracer=NoOpTracer())
        # 关键：用我们自己的处理器替换默认的
        self._span_handler = TokenTrackingSpanHandler(tracer=NoOpTracer())