import typing as T
import torch
import Stemmer
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, TransformComponent
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import (
    CodeSplitter,
    LangchainNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.node_parser.interface import NodeParser

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - langchain package split
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from hammer.hf_endpoint_embeddings import HFEndpointEmbeddings
from hammer.huggingface_helper import get_embedding_model
from hammer.llm import get_llm, get_tokenizer
from hammer.logger import logger
# from hammer.retrievers.storage import get_cached, index_cache_lock, put_cache
# from hammer.tuner.core import build_splitter

def build_splitter(study_config: T.Any, params: T.Dict[str, T.Any]) -> NodeParser:
    chunk_size = 2 ** int(params["splitter_chunk_exp"])
    overlap = int(params["splitter_chunk_overlap_frac"] * chunk_size)
    llm_name = params["response_synthesizer_llm"]
    match params["splitter_method"]:
        case "html":
            return CodeSplitter(
                language="html",
                max_chars=4 * chunk_size,
            )
        case "sentence":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                tokenizer=get_tokenizer(llm_name),
            )
        case "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                tokenizer=get_tokenizer(llm_name),
            )
        case "recursive":
            return LangchainNodeParser(
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer=AutoTokenizer.from_pretrained(
                        params.get("rag_embedding_model", "/mnt/data/wangshu/llm_lm/bge-m3")
                    ),
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )
        case _:
            raise ValueError("Invalid splitter")
#dense_retriever
def get_or_build_dense_index(
    study_config: T.Any,
    params: T.Dict[str, T.Any],
    documents: T.List[Document],
    transforms: T.List[TransformComponent],
    embedding_model: BaseEmbedding,
    max_chunks: int = 0,
    use_cache: bool = False,
) -> T.Tuple[VectorStoreIndex, BaseDocumentStore]:
    if False:
        pass
        # logger.info("Using cache. Acquiring dense index build lock for trial")
        # with index_cache_lock(study_config, params) as cache_key:
        #     logger.info(f"Acquired dense index build lock for trial: {cache_key=}")

        #     index = get_cached(cache_key)
        #     if index is not None:
        #         logger.info(
        #             f"Returning index {index} and docstore {index.docstore} from cache"
        #         )
        #         return index, index.docstore
        #     elif not index:
        #         logger.info(
        #             "Cache miss, building dense index for embedding model: %s",
        #             embedding_model.model_name,
        #         )
        #         index, docstore = _build_dense_index(
        #             documents, transforms, embedding_model, max_chunks=max_chunks
        #         )

        #         put_cache(cache_key, index)
        #         return index, docstore
    else:
        logger.info("Not looking in cache, building dense index")
        index, docstore = _build_dense_index(
            documents, transforms, embedding_model, max_chunks=max_chunks
        )
        return index, docstore

    raise RuntimeError(f"Failed to get dense index for trial with params {params}")

def _build_dense_index(
    documents: T.List[Document],
    transforms: T.List[TransformComponent],
    embedding_model: BaseEmbedding,
    max_chunks: int = 0,
) -> T.Tuple[VectorStoreIndex, BaseDocumentStore]:
    from hammer.configuration import cfg

    logger.info("Building dense index")
    logger.debug(f"Embedding model type is {type(embedding_model)})")

    pipeline = IngestionPipeline(transformations=transforms)
    nodes = pipeline.run(
        documents=documents,
        show_progress=cfg.optuna.show_progress,
    )
    if max_chunks:
        nodes = nodes[:max_chunks]
    if not isinstance(embedding_model, HFEndpointEmbeddings):
        embedding_model.reset_timeouts(total_chunks=len(nodes))  # type: ignore
    # import pdb
    # pdb.set_trace()
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embedding_model,
        insert_batch_size=2048,
        show_progress=cfg.optuna.show_progress,
    )
    del embedding_model  # Drop the Python reference explicitly.
    torch.cuda.empty_cache()  # Release cached GPU memory aggressively.
    return index, index.docstore

def _build_sparse_index(
    documents: T.List[Document], transforms: T.List[TransformComponent], top_k: int
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    logger.info("Building sparse index")
    from hammer.configuration import cfg

    pipeline = IngestionPipeline(
        transformations=transforms,
    )
    nodes = pipeline.run(
        documents=documents,
        show_progress=cfg.optuna.show_progress,
    )
    docstore = StorageContext.from_defaults().docstore
    docstore.add_documents(nodes)
    bm25_kwargs = {
        "nodes": list(docstore.docs.values()),
        "similarity_top_k": top_k,
    }
    return (
        _create_bm25_retriever(bm25_kwargs),
        docstore,
    )


def _create_bm25_retriever(bm25_kwargs: T.Dict[str, T.Any]) -> BaseRetriever:
    """Handle signature drift across llama-index BM25 retriever versions."""
    try:
        return BM25Retriever.from_defaults(
            **bm25_kwargs,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
    except TypeError:
        logger.warning("BM25Retriever.from_defaults no longer accepts stemmer/language; falling back to minimal args.")
        return BM25Retriever.from_defaults(**bm25_kwargs)

def build_rag_retriever(
    study_config: T.Any, params: T.Dict[str, T.Any]
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    logger.info(f"Building retriever for {params=}")
    logger.debug(f"Study config: {study_config.model_config=}")
    rag_method = params["rag_method"]
    top_k = int(params["rag_top_k"])
    query_decomp_enabled = params["rag_query_decomposition_enabled"]

    assert rag_method in [
        "dense",
        "sparse",
        "hybrid",
    ], f"RAG method `{rag_method}` not supported"

    logger.info("Loading grounding data documents")
    documents = list(study_config.dataset.iter_grounding_data())
    splitter = build_splitter(study_config, params)
    transforms: T.List[TransformComponent] = [splitter]

    # Build indexes
    sparse_retriever = sparse_docstore = None
    if rag_method in ["sparse", "hybrid"]:
        sparse_retriever, sparse_docstore = _build_sparse_index(
            documents, transforms, top_k
        )
        if not query_decomp_enabled and rag_method == "sparse":
            return sparse_retriever, sparse_docstore

    dense_retriever = dense_docstore = None
    if rag_method in ["dense", "hybrid"]:
        embedding_model_name = str(params["rag_embedding_model"])
        
        # Use the logical CUDA device id and initialize the CUDA context safely.
        import os
        import torch
        
        # Safe CUDA memory probe used only for debugging.
        def safe_cuda_memory_check():
            try:
                if not torch.cuda.is_available():
                    print("WARNING: CUDA is unavailable; skipping GPU memory check")
                    return
                    
                if os.environ.get('CUDA_VISIBLE_DEVICES'):
                    # When CUDA_VISIBLE_DEVICES is set, PyTorch renumbers visible devices.
                    device_id = 0
                else:
                    # Otherwise, keep using the default logical device.
                    device_id = 0
                
                # Clamp to a valid device id.
                if device_id >= torch.cuda.device_count():
                    device_id = 0
                
                # Make sure the CUDA context exists before querying memory.
                torch.cuda.set_device(device_id)
                torch.cuda.init()
                
                allocated = torch.cuda.memory_allocated(device_id) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**2    # MB
                print(f"GPU {device_id}: Allocated = {allocated:.2f} MB | Reserved = {reserved:.2f} MB")
                
            except Exception as e:
                print(f"WARNING: Failed to read GPU memory info: {e}")
        
        # Run the CUDA memory probe before loading the embedding model.
        safe_cuda_memory_check()
   
        embedding_model, _ = get_embedding_model(
            embedding_model_name,
            timeout_config=study_config.timeouts,
            total_chunks=0,
            device=study_config.optimization.embedding_device,
            use_hf_endpoint_models=study_config.optimization.use_hf_embedding_models,
        )
        
        # Run it again after the model is loaded.
        safe_cuda_memory_check()
            
        assert embedding_model is not None

        max_chunks = 0
        if study_config.toy_mode:
            max_chunks = 2048

        dense_index, dense_docstore = get_or_build_dense_index(
            study_config,
            params,
            documents,
            transforms,
            embedding_model,
            max_chunks=max_chunks,
        )
      
        dense_retriever = dense_index.as_retriever(
            embed_model=embedding_model, similarity_top_k=top_k
        )
        if not query_decomp_enabled and rag_method == "dense":
            return dense_retriever, dense_docstore

    # Not dense or sparse - build fusion retriever
    if rag_method == "hybrid":
        # Hybrid mode, use both retrievers with weights
        retrievers = [sparse_retriever, dense_retriever]
        hybrid_bm25_weight = float(
            params.get("rag_hybrid_bm25_weight", params.get("hybrid_bm25_weight"))
        )
        retriever_weights = [hybrid_bm25_weight, 1 - hybrid_bm25_weight]

    else:
        # Otherwise, pick the active retriever
        retriever = dense_retriever or sparse_retriever
        retrievers = [retriever]
        retriever_weights = [1]

    fusion_retriever_params = {
        "llm": get_llm("gpt-4o-mini"),  # Not used without query decomposition enabled
        "mode": FUSION_MODES(params["rag_fusion_mode"]),
        "use_async": False,
        "verbose": False,  # Disable verbose query-decomposition output.
        "similarity_top_k": top_k,
        "num_queries": 1,
        "retriever_weights": retriever_weights,
        "retrievers": retrievers,
    }

    docstore = dense_docstore or sparse_docstore

    if rag_method == "hybrid":
        assert dense_docstore is not None and sparse_docstore is not None
        docstore = sparse_docstore
        docstore.add_documents(list(dense_docstore.docs.values()))

    if query_decomp_enabled:
        # Get query decomp params
        query_decomposition_num_queries = params["rag_query_decomposition_num_queries"]
        query_decomposition_llm_name = str(params["rag_query_decomposition_llm_name"])
        # Use the configured LLM instance so QueryFusionRetriever does not fall back to OpenAI.
        # Actual query decomposition is handled by BatchLLMCaller in optimized_rag_prompt_builder.py.
        # The LLM instance is passed here only to satisfy QueryFusionRetriever initialization.
        query_decomposition_llm = get_llm(query_decomposition_llm_name)
        # Add query decomp params and retriever
        fusion_retriever_params.update(
            **{
                "llm": query_decomposition_llm,
                "num_queries": query_decomposition_num_queries,
            }
        )

    fusion_retriever = QueryFusionRetriever(**fusion_retriever_params)
    return fusion_retriever, docstore

def build_dummy_retriever(
    study_config: T.Any,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    """Builds a dummy retriever that returns all documents in the corpus."""
    logger.info("Building dummy retriever that returns all documents")
    documents = list(study_config.dataset.iter_grounding_data())

    from hammer.configuration import cfg

    pipeline = IngestionPipeline(transformations=[])
    nodes = pipeline.run(
        documents=documents,
        show_progress=cfg.optuna.show_progress,
    )

    docstore = StorageContext.from_defaults().docstore
    docstore.add_documents(nodes)

    class DummyRetriever(BaseRetriever):
        async def aretrieve(self, query: str) -> T.List:
            return [NodeWithScore(node=n) for n in docstore.docs.values()]

        def _retrieve(self, query: str) -> T.List[NodeWithScore]:
            raise NotImplementedError("DummyRetriever only supports async retrieval.")

    return DummyRetriever(), docstore
