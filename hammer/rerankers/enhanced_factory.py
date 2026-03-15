# hammer/rerankers/enhanced_factory.py
"""
Enhanced Reranker Factory integrating user's comprehensive reranker framework
with the existing RAG optimization system.
"""

import typing as T
from pathlib import Path
import sys
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from hammer.llm import LLMs, get_llm
from hammer.logger import logger

# Import user's reranker components with fallback handling
try:
    # Try to import user's reranker framework
    # Note: This requires fixing internal imports in user's Rerank framework
    from hammer.Rerank.RerankFactory import METHOD_MAP
    from hammer.Schema.DocumentSchema import Document, Question, Context, Answer
    logger.info("Successfully imported the user Rerank framework")
    USER_RERANK_AVAILABLE = True
except ImportError as e:
    logger.info("User Rerank framework unavailable; using fallback mode: %s", e)
    logger.info("Enable enhanced rerankers by fixing internal imports in the Rerank framework")
    METHOD_MAP = {}
    USER_RERANK_AVAILABLE = False
    
    # Create dummy classes for type hints when user's framework is not available
    class Document:
        def __init__(self, question=None, contexts=None):
            self.question = question
            self.contexts = contexts or []
            self.reorder_contexts = []
    
    class Question:
        def __init__(self, question):
            self.question = question
            
    class Context:
        def __init__(self, text, id, score=0.0):
            self.text = text
            self.id = id
            self.score = score

class UserRerankAdapter(BaseNodePostprocessor):
    """
    Adapter to integrate user's reranker models with LlamaIndex's postprocessor interface.
    This bridges the gap between user's Document/Context schema and LlamaIndex's NodeWithScore.
    """
    
    _reranker_name: str = PrivateAttr()
    _model_name: str | None = PrivateAttr()
    _top_n: int = PrivateAttr()
    _kwargs: T.Dict[str, T.Any] = PrivateAttr(default_factory=dict)
    _reranker: T.Any = PrivateAttr(default=None)

    def __init__(self, reranker_name: str, model_name: str = None, top_n: int = 5, **kwargs):
        super().__init__()
        # Store configuration first and initialize lazily to avoid field access issues.
        self._reranker_name = reranker_name.lower()
        self._model_name = model_name
        self._top_n = top_n
        self._kwargs = kwargs
        self._reranker = None
        
        # Delay initialization until object state is ready.
        self._initialize_reranker()
        
    def _initialize_reranker(self):
        """Initialize the user reranker lazily."""
        try:
            if not USER_RERANK_AVAILABLE:
                raise ValueError(f"User's Rerank framework not available. Please check dependencies.")
                
            if self._reranker_name not in METHOD_MAP:
                raise ValueError(f"Unknown reranker: {self._reranker_name}. Available: {list(METHOD_MAP.keys())}")
            
            # Use the framework-level Reranking entrypoint so predefined model mapping works.
            from hammer.Rerank.Reranking import Reranking
            
            # Create reranker through the user's Reranking class.
            init_params = {
                'method': self._reranker_name,
            }
            
            if self._model_name and self._model_name != "default":
                init_params['model_name'] = self._model_name
            
            # Only forward supported parameters to avoid unexpected keyword errors.
            supported_params = ['device', 'batch_size', 'api_key']
            for param_name in supported_params:
                if param_name in self._kwargs:
                    init_params[param_name] = self._kwargs[param_name]
                
            # Reranking handles the HF_PRE_DEFIND_MODELS mapping internally.
            self._reranker = Reranking(**init_params)
            logger.info(
                "Initialized reranker %s with model %s",
                self._reranker_name,
                self._model_name,
            )
            
        except Exception as e:
            logger.error("Failed to initialize reranker %s: %s", self._reranker_name, e)
            self._reranker = None
            raise

    def _convert_to_user_format(self, nodes: T.List[NodeWithScore], query: str) -> Document:
        """Convert LlamaIndex nodes to user's Document format"""
        question = Question(query)
        # Create an empty Answer object because retrieval does not produce answers.
        answers = Answer([])  
        contexts = []
        
        for i, node in enumerate(nodes):
            context = Context(
                text=node.get_content(),
                id=i,
                score=getattr(node, 'score', 0.0)
            )
            contexts.append(context)
            
        return Document(question=question, answers=answers, contexts=contexts)
    
    def _convert_from_user_format(self, document: Document, original_nodes: T.List[NodeWithScore]) -> T.List[NodeWithScore]:
        """Convert user's reranked results back to LlamaIndex format"""
        # Map content back to the original retrieved nodes.
        content_to_node = {}
        for node in original_nodes:
            content_to_node[node.get_content()] = node
            
        reranked_nodes = []
        for context in document.reorder_contexts[:self._top_n]:
            original_node = content_to_node.get(context.text)
            if original_node:
                # Preserve reranker scores when available.
                if hasattr(context, 'score'):
                    original_node.score = context.score
                reranked_nodes.append(original_node)
                
        return reranked_nodes

    def _postprocess_nodes(
        self,
        nodes: T.List[NodeWithScore],
        query_bundle: T.Optional[QueryBundle] = None,
    ) -> T.List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []
        
        if self._reranker is None:
            logger.error("Reranker %s is not initialized", self._reranker_name)
            return nodes[:self._top_n]  # Fall back to the original order.

        try:
            # Convert to the user's schema.
            document = self._convert_to_user_format(nodes, query_bundle.query_str)
            
            # Apply reranking.
            reranked_documents = self._reranker.rank([document])
            
            # Convert back to LlamaIndex nodes.
            return self._convert_from_user_format(reranked_documents[0], nodes)
            
        except Exception as e:
            logger.error("Reranking failed for %s: %s", self._reranker_name, e)
            return nodes[:self._top_n]  # Fall back to the original order.

def build_reranker_postprocessor(params: T.Dict[str, T.Any]) -> T.Optional[BaseNodePostprocessor]:
    """
    Build reranker postprocessor based on configuration parameters.
    This function bridges the original factory.py with user's enhanced rerankers.
    """
    if not params.get("reranker_enabled"):
        return None
        
    # Read reranker configuration from either legacy or current parameter names.
    reranker_name = (
        params.get("reranker_llm_name")
        or params.get("reranker_llm")
        or params.get("reranker_model", "flashrank")
    )
    reranker_top_k = params.get("reranker_top_k", 5)
    
    logger.info("Building reranker %s with top_k=%s", reranker_name, reranker_top_k)
    
    # LLM-based rerankers are handled by the prompt-builder path.
    if reranker_name in LLMs:
        logger.info(
            "ℹ️ LLM reranker '%s' will be handled by the prompt builder path; skipping serial LLMRerank.",
            reranker_name,
        )
        return None
    
    # Handle enhanced rerankers backed by the user's framework.
    reranker_mapping = {
        # Map config-level names to the framework's reranker ids.
        'ColbertRanker': 'colbert_ranker',
        'colbert_ranker': 'colbert_ranker',
        'FlashRank': 'flashrank',
        'flashrank': 'flashrank',
        'Flashrank': 'flashrank',
        'EchoRank': 'echorank',
        'echorank': 'echorank',
        'TransformerRanker': 'transformer_ranker',
        'transformer_ranker': 'transformer_ranker',
        'MonoT5': 'monot5',
        'monot5': 'monot5',
        'RankT5': 'rankt5',
        'rankt5': 'rankt5',
        'ListT5': 'listt5',
        'listt5': 'listt5',
        'TWOLAR': 'twolar',
        'twolar': 'twolar',
        'MonoBERT': 'monobert_ranker',
        'monobert_ranker': 'monobert_ranker',
        'InRanker': 'inranker',
        'inranker': 'inranker',
        'UPR': 'upr',
        'upr': 'upr',
    }
    
    user_reranker_name = reranker_mapping.get(reranker_name)
    if user_reranker_name and user_reranker_name in METHOD_MAP and USER_RERANK_AVAILABLE:
        try:
            # Resolve model name from config or a reranker-specific default.
            model_name = params.get("reranker_model_name")
            if not model_name or model_name == "default":
                # Use framework-compatible defaults for each reranker family.
                model_defaults = {
                    'flashrank': 'ms-marco-MiniLM-L-12-v2',
                    'transformer_ranker': 'mxbai-rerank-xsmall', 
                    'colbert_ranker': 'Colbert',
                    'monot5': 'monot5-base-msmarco',
                    'rankt5': 'rankt5-base',
                    'listt5': 'listt5-base', 
                    'twolar': 'twolar-xl',
                    'monobert_ranker': 'monobert-large',
                    'inranker': 'inranker-base',
                    'echorank': 'flan-t5-large',
                    'upr': 't5-base'
                }
                model_name = model_defaults.get(user_reranker_name)
            
            # Reuse the project GPU-selection logic when available.
            try:
                from hammer.tuner.main_tuner_mcts import DEVICE_ID
                optimal_device = f"cuda:{DEVICE_ID}"
                logger.info("Specialized reranker %s will use %s", user_reranker_name, optimal_device)
                device = params.get("reranker_device", optimal_device)
            except ImportError:
                device = params.get("reranker_device", "cuda")
                logger.warning("Could not import GPU selection logic; using %s", device)
            
            return UserRerankAdapter(
                reranker_name=user_reranker_name,
                model_name=model_name or "default",
                top_n=reranker_top_k,
                device=device,
                batch_size=params.get("reranker_batch_size", 16)
            )
        except Exception as e:
            logger.error("Failed to create enhanced reranker %s: %s", user_reranker_name, e)
    elif user_reranker_name and not USER_RERANK_AVAILABLE:
        logger.warning(
            "User Rerank framework is unavailable for %s; falling back to the original factory.",
            reranker_name,
        )
    
    # Fall back to the original factory for unknown rerankers.
    try:
        from hammer.rerankers.factory import get_reranker
        return get_reranker(reranker_name, reranker_top_k)
    except Exception as e:
        logger.error("All reranker creation attempts failed: %s", e)
        return None

# For backward compatibility
def get_enhanced_reranker(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """Compatibility wrapper for the enhanced reranker factory."""
    params = {
        "reranker_enabled": True,
        "reranker_llm_name": name,
        "reranker_top_k": top_k
    }
    return build_reranker_postprocessor(params)
