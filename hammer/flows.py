
"""

"""
import time
import typing as T
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
import llama_index.core.instrumentation as instrument
from llama_index.agent.introspective import (
    IntrospectiveAgentWorker,
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.agent.introspective.reflective.tool_interactive_reflection import (
    StoppingCallable,
)
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    Response,
    get_response_synthesizer,
)
from llama_index.core.agent import (
    AgentChatResponse,
    AgentRunner,
    FunctionCallingAgentWorker,
    ReActAgent,
)
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.llms import ChatMessage, CompletionResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import LLMRerank, PrevNextNodePostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    SubQuestionQueryEngine,
    TransformQueryEngine,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata
from numpy import ceil

from hammer.configuration import cfg
from hammer.instrumentation.arize import instrument_arize
from hammer.llm import get_llm_name, get_tokenizer
from hammer.logger import logger
# from hammer.studies import get_critique_template, get_react_template
from hammer.rerankers.enhanced_factory import build_reranker_postprocessor

if cfg.instrumentation.tracing_enabled:
    instrument_arize()

@dataclass(kw_only=True)
class Flow:
    response_synthesizer_llm: LLM | FunctionCallingLLM
    template: str | None = None
    get_examples: T.Callable | None = None
    name: str = "Generator Flow"
    params: dict | None = None
    enforce_full_evaluation: bool = False

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @property
    def verbose(self):
        log_level = logger.level
        if log_level <= 20:
            return True
        return False

    @property
    def prompt_template(self) -> PromptTemplate:
        if self.template is None:
            raise ValueError("Flow template not set. Cannot create prompt template.")
        prompt_template = None
        function_mappings = None
        if self.get_examples is not None:
            function_mappings = {"few_shot_examples": self.get_examples}

        logger.debug("Creating prompt template from '%s'", self.template)
        prompt_template = PromptTemplate(
            template=self.template,
            prompt_type=PromptType.QUESTION_ANSWER,
            function_mappings=function_mappings,
        )

        return prompt_template

    def get_prompt(self, query) -> str:
        if self.template is None:
            return query

        if self.get_examples is None:
            return self.template.format(query_str=query)

        examples = self.get_examples(query)
        assert examples, "No examples found for few-shot prompting"

        return self.template.format(
            query_str=query,
            few_shot_examples=examples,
        )

    def generate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float, T.List]:
        response, duration = self._generate(query)
        return response, duration, []

    def _generate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float]:
        assert self.response_synthesizer_llm is not None, (
            "Response synthesizer LLM is not set. Cannot generate."
        )
        start_time = time.perf_counter()
        prompt = self.get_prompt(query)
        response: CompletionResponse = self.response_synthesizer_llm.complete(prompt)
        
        # 🔥 记录token使用到全局统计
        from hammer.utils.simple_token_tracker import record_llm_response
        from hammer.llm import get_llm_name
        llm_name = get_llm_name(self.response_synthesizer_llm)
        record_llm_response(response, call_type="rag", llm_name=llm_name)
        
        duration = time.perf_counter() - start_time
        return response, duration

    async def agenerate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float, T.List]:
        response, duration = await self._agenerate(query)
        return response, duration, []

    async def _agenerate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float]:
        assert self.response_synthesizer_llm is not None, (
            "Response synthesizer LLM is not set. Cannot generate."
        )
        start_time = time.perf_counter()
        prompt = self.get_prompt(query)
        response: CompletionResponse = await self.response_synthesizer_llm.acomplete(
            prompt
        )
        
        # 🔥 记录token使用到全局统计
        from hammer.utils.simple_token_tracker import record_llm_response
        from hammer.llm import get_llm_name
        llm_name = get_llm_name(self.response_synthesizer_llm)
        record_llm_response(response, call_type="rag", llm_name=llm_name)
        
        duration = time.perf_counter() - start_time
        return response, duration

@dataclass(kw_only=True)
class RetrieverFlow(Flow):
    """Flow that only retrieves documents."""

    retriever: BaseRetriever
    docstore: BaseDocumentStore | None = None
    hyde_llm: LLM | None = None
    additional_context_num_nodes: int = 0
    name: str = "Retriever Only Flow"

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        node_postprocessors: T.List[BaseNodePostprocessor] = []
        if self.additional_context_num_nodes > 0:
            from hammer.logger import logger
            logger.info(f"🔗 [RetrieverOnlyFlow] 启用Additional Context功能: 上下文节点数={self.additional_context_num_nodes}")
            logger.info(f"🔗 [RetrieverOnlyFlow] Additional Context配置: 前后各{int(ceil(self.additional_context_num_nodes / 2))}个节点, mode=both")
            assert self.docstore is not None
            node_postprocessors.append(
                PrevNextNodePostprocessor(
                    docstore=self.docstore,
                    num_nodes=int(ceil(self.additional_context_num_nodes / 2)),
                    mode="both",
                )
            )
            logger.info(f"✅ [RetrieverOnlyFlow] Additional Context后处理器已添加到pipeline")
        response_synthesizer = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
        )
        base_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        if self.hyde_llm is not None:
            hyde = HyDEQueryTransform(llm=self.hyde_llm, include_original=True)
            return TransformQueryEngine(base_engine, query_transform=hyde)
        return base_engine

    @cached_property
    def tokenizer(self) -> T.Callable:
        return get_tokenizer(get_llm_name(self.response_synthesizer_llm))

    def generate(self, query: str, *args, **kwargs):
        raise NotImplementedError("RetrieverFlow does not support generation.")

    async def agenerate(self, query: str, *args, **kwargs):
        raise NotImplementedError("RetrieverFlow does not support generation.")

    def retrieve(self, query: str) -> T.Tuple[T.List[NodeWithScore], float]:
        start_time = time.perf_counter()
        qb = QueryBundle(query)
        if isinstance(self.query_engine, TransformQueryEngine):
            response = self.query_engine.query(qb)
            assert isinstance(response, Response), (
                f"Expected Response, got {type(response)=}"
            )
            retrieval_result = response.source_nodes
        else:
            retrieval_result = self.query_engine.retrieve(qb)
        duration = time.perf_counter() - start_time
        return retrieval_result, duration

    async def aretrieve(self, query: str) -> T.Tuple[T.List[NodeWithScore], float]:
        start_time = time.perf_counter()
        qb = QueryBundle(query)
        if isinstance(self.query_engine, TransformQueryEngine):
            response = await self.query_engine.aquery(qb)
            assert isinstance(response, Response), (
                f"Expected Response, got {type(response)=}"
            )
            retrieval_result = response.source_nodes
        else:
            assert hasattr(self.query_engine, "aretrieve"), (
                f"{self.query_engine} does not have 'aretrieve' method"
            )
            retrieval_result = await self.query_engine.aretrieve(qb)
        duration = time.perf_counter() - start_time
        return retrieval_result, duration

@dataclass(kw_only=True)
class RAGFlow(Flow):
    retriever: BaseRetriever
    docstore: BaseDocumentStore | None = None
    hyde_llm: LLM | None = None
    reranker_llm: LLM | None = None
    reranker_top_k: int | None = None
    name: str = "RAG Flow"
    additional_context_num_nodes: int = 0

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        node_postprocessors: T.List[BaseNodePostprocessor] = []

        # 2. 替换原有的 Reranker 逻辑
        # ----- 旧逻辑 START -----
        # if self.reranker_llm is not None:
        #     assert self.reranker_top_k, (
        #         "Reranker enabled, need reranker_top_k param set"
        #     )
        #     node_postprocessors.append(
        #         LLMRerank(top_n=self.reranker_top_k, llm=self.reranker_llm)
        #     )
        # ----- 旧逻辑 END -----

        # +++++ 新逻辑 START +++++
        if self.params and self.params.get("reranker_enabled"):
            reranker = build_reranker_postprocessor(self.params)
            if reranker:
                node_postprocessors.append(reranker)
        # +++++ 新逻辑 END +++++
        if self.additional_context_num_nodes > 0:
            from hammer.logger import logger
            logger.info(f"🔗 [RAGFlow] 启用Additional Context功能: 上下文节点数={self.additional_context_num_nodes}")
            logger.info(f"🔗 [RAGFlow] Additional Context配置: 前后各{int(ceil(self.additional_context_num_nodes / 2))}个节点, mode=both")
            assert self.docstore is not None
            node_postprocessors.append(
                PrevNextNodePostprocessor(
                    docstore=self.docstore,
                    num_nodes=int(ceil(self.additional_context_num_nodes / 2)),
                    mode="both",
                )
            )
            logger.info(f"✅ [RAGFlow] Additional Context后处理器已添加到pipeline")
        response_synthesizer = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=self.prompt_template,
        )
        retriever = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        if self.hyde_llm is not None:
            hyde = HyDEQueryTransform(llm=self.hyde_llm, include_original=True)
            retriever = TransformQueryEngine(retriever, query_transform=hyde)  # type: ignore
        return retriever

    def get_prompt(self, query) -> str:
        if self.template is None:
            return query

        if self.get_examples is None:
            return self.template.format(query_str=query)

        examples = self.get_examples(query)
        assert examples, "No examples found for few-shot prompting"

        return self.template.format(
            query_str=query,
            few_shot_examples=examples,
        )

    def retrieve(self, query: str) -> T.List[NodeWithScore]:
        return self.query_engine.retrieve(QueryBundle(query))

    async def aretrieve(self, query: str) -> T.List[NodeWithScore]:
        assert hasattr(self.query_engine, "aretrieve"), (
            f"{self.query_engine} does not have 'aretrieve' method"
        )
        return await self.query_engine.aretrieve(QueryBundle(query))

    def _generate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = self.query_engine.query(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),  # type: ignore
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration

    async def _agenerate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = await self.query_engine.aquery(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        assert isinstance(response.response, str), (
            f"Expected str, got {type(response.response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration

class Flows(Enum):
    GENERATOR_FLOW = Flow
    RAG_FLOW = RAGFlow
    # LLAMA_INDEX_REACT_AGENT_FLOW = ReActAgentFlow
    # LLAMA_INDEX_CRITIQUE_AGENT_FLOW = CritiqueAgentFlow
    # LLAMA_INDEX_SUB_QUESTION_FLOW = SubQuestionRAGFlow
    # LLAMA_INDEX_LATS_RAG_AGENT = LATSAgentFlow
    # RETRIEVER_FLOW = RetrieverFlow
