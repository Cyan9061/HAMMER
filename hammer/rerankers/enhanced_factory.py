# hammer/rerankers/enhanced_factory.py
"""
Enhanced Reranker Factory integrating user's comprehensive reranker framework
with the existing RAG optimization system.
"""

import typing as T
from pathlib import Path
import sys
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from hammer.llm import LLMs, get_llm
from hammer.logger import logger

# Import user's reranker components with fallback handling
try:
    # Try to import user's reranker framework
    # Note: This requires fixing internal imports in user's Rerank framework
    from hammer.Rerank.RerankFactory import METHOD_MAP
    from hammer.Schema.DocumentSchema import Document, Question, Context, Answer
    logger.info("✅ Successfully imported user's Rerank framework")
    USER_RERANK_AVAILABLE = True
except ImportError as e:
    logger.info(f"ℹ️ User's Rerank framework not available, using fallback mode: {e}")
    logger.info("💡 To enable enhanced rerankers, please fix internal imports in Rerank framework")
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
    
    def __init__(self, reranker_name: str, model_name: str = None, top_n: int = 5, **kwargs):
        super().__init__()
        # 保存参数但不初始化reranker，避免字段访问错误
        self._reranker_name = reranker_name.lower()
        self._model_name = model_name
        self._top_n = top_n
        self._kwargs = kwargs
        self._reranker = None
        
        # 延迟初始化reranker
        self._initialize_reranker()
        
    def _initialize_reranker(self):
        """延迟初始化用户的reranker实例"""
        try:
            if not USER_RERANK_AVAILABLE:
                raise ValueError(f"User's Rerank framework not available. Please check dependencies.")
                
            if self._reranker_name not in METHOD_MAP:
                raise ValueError(f"Unknown reranker: {self._reranker_name}. Available: {list(METHOD_MAP.keys())}")
            
            # 🔑 修复：使用用户的Reranking类而不是直接实例化具体的reranker类
            # 这确保了HF_PRE_DEFIND_MODELS映射系统被正确使用
            from hammer.Rerank.Reranking import Reranking
            
            # Create reranker through user's Reranking class
            init_params = {
                'method': self._reranker_name,
            }
            
            if self._model_name and self._model_name != "default":
                init_params['model_name'] = self._model_name
            
            # 只添加支持的参数，避免unexpected keyword argument错误
            supported_params = ['device', 'batch_size', 'api_key']
            for param_name in supported_params:
                if param_name in self._kwargs:
                    init_params[param_name] = self._kwargs[param_name]
                
            # 使用用户的Reranking类，这会自动处理HF_PRE_DEFIND_MODELS映射
            self._reranker = Reranking(**init_params)
            logger.info(f"✅ Successfully initialized {self._reranker_name} reranker with model {self._model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self._reranker_name} reranker: {e}")
            self._reranker = None
            raise

    def _convert_to_user_format(self, nodes: T.List[NodeWithScore], query: str) -> Document:
        """Convert LlamaIndex nodes to user's Document format"""
        question = Question(query)
        # Create empty Answer object since we don't have answers from retrieval
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
        # Create mapping from content to original nodes
        content_to_node = {}
        for node in original_nodes:
            content_to_node[node.get_content()] = node
            
        reranked_nodes = []
        for context in document.reorder_contexts[:self._top_n]:
            original_node = content_to_node.get(context.text)
            if original_node:
                # Update score if available
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
            logger.error(f"❌ Reranker not initialized for {self._reranker_name}")
            return nodes[:self._top_n]  # Fallback to original order

        try:
            # Convert to user's format
            document = self._convert_to_user_format(nodes, query_bundle.query_str)
            
            # Apply user's reranker
            reranked_documents = self._reranker.rank([document])
            
            # Convert back to LlamaIndex format
            return self._convert_from_user_format(reranked_documents[0], nodes)
            
        except Exception as e:
            logger.error(f"❌ Reranking failed with {self._reranker_name}: {e}")
            return nodes[:self._top_n]  # Fallback to original order

def build_reranker_postprocessor(params: T.Dict[str, T.Any]) -> T.Optional[BaseNodePostprocessor]:
    """
    Build reranker postprocessor based on configuration parameters.
    This function bridges the original factory.py with user's enhanced rerankers.
    """
    if not params.get("reranker_enabled"):
        return None
        
    # Get reranker configuration
    reranker_name = params.get("reranker_llm_name") or params.get("reranker_model", "flashrank")
    reranker_top_k = params.get("reranker_top_k", 5)
    
    logger.info(f"🔧 Building reranker: {reranker_name}, top_k={reranker_top_k}")
    
    # Handle LLM-based rerankers (backward compatibility)
    if reranker_name in LLMs:
        try:
            from llama_index.core.postprocessor import LLMRerank
            llm = get_llm(reranker_name)
            return LLMRerank(top_n=reranker_top_k, llm=llm)
        except Exception as e:
            logger.error(f"❌ Failed to create LLMRerank with {reranker_name}: {e}")
            return None
    
    # Handle user's enhanced rerankers
    reranker_mapping = {
        # Map config names to user's reranker names
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
            # Get model name from predefined models or params
            model_name = params.get("reranker_model_name")
            if not model_name or model_name == "default":
                # Set reasonable defaults based on reranker type using user's predefined models
                model_defaults = {
                    'flashrank': 'ms-marco-MiniLM-L-12-v2',
                    'transformer_ranker': 'mxbai-rerank-xsmall', 
                    'colbert_ranker': 'Colbert',
                    'monot5': 'monot5-base-msmarco',
                    'rankt5': 'rankt5-base',
                    'listt5': 'listt5-base', 
                    'twolar': 'twolar-xl',
                    'monobert_ranker': 'monobert-large',
                    'inranker': 'inranker-base',  # 🔑 修复：使用用户预定义的简化名称
                    'echorank': 'flan-t5-large',
                    'upr': 't5-base'
                }
                model_name = model_defaults.get(user_reranker_name)
            
            # 🔧 智能GPU分配：使用huggingface_helper的GPU选择逻辑
            try:
                from hammer.tuner.main_tuner_mcts import DEVICE_ID
                optimal_device = f"cuda:{DEVICE_ID}"
                logger.info(f"🎯 专用reranker '{user_reranker_name}' 使用智能分配的GPU: {optimal_device}")
                device = params.get("reranker_device", optimal_device)
            except ImportError:
                device = params.get("reranker_device", "cuda")
                logger.warning(f"⚠️ 无法导入GPU选择逻辑，使用默认设备: {device}")
            
            return UserRerankAdapter(
                reranker_name=user_reranker_name,
                model_name=model_name or "default",  # 提供默认值避免None
                top_n=reranker_top_k,
                device=device,
                batch_size=params.get("reranker_batch_size", 16)
            )
        except Exception as e:
            logger.error(f"❌ Failed to create {user_reranker_name}: {e}")
    elif user_reranker_name and not USER_RERANK_AVAILABLE:
        logger.warning(f"⚠️ User's Rerank framework not available for {reranker_name}. Falling back to original factory.")
    
    # Fallback to original factory for unknown rerankers
    try:
        from hammer.rerankers.factory import get_reranker
        return get_reranker(reranker_name, reranker_top_k)
    except Exception as e:
        logger.error(f"❌ All reranker creation attempts failed: {e}")
        return None

# For backward compatibility
def get_enhanced_reranker(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """Enhanced version of get_reranker that supports user's framework"""
    params = {
        "reranker_enabled": True,
        "reranker_llm_name": name,
        "reranker_top_k": top_k
    }
    return build_reranker_postprocessor(params)