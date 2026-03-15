
import os
import typing as T

from filelock import FileLock, Timeout
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

try:
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding
except ImportError:  # pragma: no cover - optional provider
    OpenAILikeEmbedding = None  # type: ignore[assignment]
from optimum.onnxruntime import ORTModelForFeatureExtraction
from slugify import slugify
from transformers import AutoTokenizer

from hammer.configuration import cfg
from hammer.embeddings.timeouts import EmbeddingTimeoutMixin, TimeoutConfig
# from hammer.hf_endpoint_models import get_hf_endpoint_embed_model
from hammer.logger import logger

EmbeddingDeviceType = T.Any

def get_hf_token():
    hf_token = str(cfg.hf_embeddings.api_key.get_secret_value())
    if not hf_token or hf_token == "NOT SET":
        return {}
    return {"HF_TOKEN": hf_token}

def load_hf_token_into_env():
    hf_token = get_hf_token()
    # only update the environment if set
    if hf_token:
        os.environ.update(hf_token)

class HuggingFaceEmbeddingWithTimeout(EmbeddingTimeoutMixin, HuggingFaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if OpenAILikeEmbedding is not None:
    class OpenAILikeEmbeddingWithTimeout(EmbeddingTimeoutMixin, OpenAILikeEmbedding):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
else:
    OpenAILikeEmbeddingWithTimeout = None  # type: ignore[assignment]

LOCAL_EMBEDDING_MODELS = (
    {
        model.model_name: OpenAILikeEmbeddingWithTimeout(
            model_name=model.model_name,
            api_base=str(model.api_base),
            api_key=model.api_key.get_secret_value()
            if model.api_key is not None
            else cfg.local_models.default_api_key.get_secret_value(),
            timeout=model.timeout,
            dimensions=model.dimensions,
            additional_kwargs=model.additional_kwargs,
        )
        for model in cfg.local_models.embedding
    }
    if cfg.local_models.embedding and OpenAILikeEmbedding is not None
    else {}
)

class OptimumEmbeddingWithTimeout(EmbeddingTimeoutMixin, OptimumEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_optimal_gpu_device() -> str:
    """Return the preferred GPU device for embedding workloads.

    The current project path uses the logical MCTS device id when available.
    """
    try:
        import torch
        import os
        import inspect
        
        # Optional memory-fragmentation tuning can be added here if needed.
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Reuse the logical device selection from the MCTS entrypoint.
        # import pdb
        # pdb.set_trace()
        from hammer.tuner.main_tuner_mcts import DEVICE_ID as mcts_gpu_id
        # from hammer.tuner.main_tuner_tpe import DEVICE_ID as tpe_gpu_id
        # mcts_gpu = os.environ.get('HAMMER_MCTS_GPU', f'{mcts_gpu_id}')
        # tpe_gpu = os.environ.get('HAMMER_TPE_GPU', f'{tpe_gpu_id}')
        
        # # Detect MCTS/TPE via the call stack if we want route-specific assignment.
        # stack = inspect.stack()
        # is_mcts_experiment = any('mcts' in frame.filename.lower() for frame in stack)
        # is_tpe_experiment = any('tpe' in frame.filename.lower() for frame in stack)
        
        # # Assign GPUs based on the experiment type.
        # if is_mcts_experiment:
        #     device_str = f"cuda:{mcts_gpu}"
        #     logger.info(f"MCTS experiment using GPU {device_str}")
        #     return device_str
        # elif is_tpe_experiment:
        #     device_str = f"cuda:{tpe_gpu}"
        #     logger.info(f"TPE experiment using GPU {device_str}")
        #     return device_str
        
        # if not torch.cuda.is_available():
        #     logger.info("CUDA is unavailable; falling back to CPU")
        #     return "cpu"
        
        # # If the experiment type is unknown, fall back to the MCTS GPU.
        # logger.info(f"Experiment type unknown; defaulting to MCTS GPU cuda:{mcts_gpu}")
        return f"cuda:{mcts_gpu_id}"
        
        # gpu_count = torch.cuda.device_count()
        # if gpu_count == 0:
        #     logger.info("No GPU detected; using CPU")
        #     return "cpu"
        # elif gpu_count == 1:
        #     logger.info("Single-GPU environment; using cuda:0")
        #     torch.cuda.empty_cache()
        #     return "cuda:0"
        
        # # Multi-GPU environment: choose from a preferred GPU range.
        # preferred_gpus = [4, 5, 6, 7]
        # available_preferred = [gpu_id for gpu_id in preferred_gpus if gpu_id < gpu_count]
        
        # if available_preferred:
        #     logger.info(f"Detected {gpu_count} GPUs; choosing from preferred set {preferred_gpus}...")
        #     gpu_candidates = available_preferred
        # else:
        #     logger.info(f"Detected {gpu_count} GPUs; preferred GPUs unavailable, selecting automatically...")
        #     gpu_candidates = list(range(gpu_count))
        
        # gpu_info = []
        # current_device = torch.cuda.current_device()  # Save the current device.
        
        # for i in gpu_candidates:
        #     try:
        #         # Read GPU properties.
        #         props = torch.cuda.get_device_properties(i)
        #         total_memory = props.total_memory
                
        #         # Temporarily switch to the candidate device to inspect memory.
        #         torch.cuda.set_device(i)
                
        #         # Clear cached memory before sampling.
        #         torch.cuda.empty_cache()
                
        #         allocated_memory = torch.cuda.memory_allocated(i)
        #         cached_memory = torch.cuda.memory_reserved(i)
        #         free_memory = total_memory - cached_memory
                
        #         gpu_info.append({
        #             'device_id': i,
        #             'name': props.name,
        #             'total_memory': total_memory,
        #             'allocated_memory': allocated_memory,
        #             'cached_memory': cached_memory,
        #             'free_memory': free_memory,
        #             'free_ratio': free_memory / total_memory
        #         })
                
        #         logger.debug(f"GPU {i} ({props.name}): "
        #                     f"total={total_memory//1024**3}GB, "
        #                     f"free={free_memory//1024**3}GB ({free_memory/total_memory*100:.1f}%)")
                            
        #     except Exception as e:
        #         logger.warning(f"GPU {i} probe failed: {e}")
        #         continue
        
        # # Restore the original device.
        # torch.cuda.set_device(current_device)
        
        # if not gpu_info:
        #     logger.warning("All GPU probes failed; falling back to the default CUDA device")
        #     return "cuda"
        
        # # Selection strategy: prefer GPUs with more free memory and preferred ids.
        # optimal_gpu = max(gpu_info, key=lambda x: (
        #     x['device_id'] in preferred_gpus,  # Prefer the configured GPU range.
        #     x['free_ratio'],                   # Higher free-memory ratio is better.
        #     x['device_id'] != 0                # Avoid GPU0 when possible.
        # ))
        
        # device_str = f"cuda:{optimal_gpu['device_id']}"
        # logger.info(f"Selected GPU {optimal_gpu['device_id']} ({optimal_gpu['name']}): "
        #            f"free={optimal_gpu['free_memory']//1024**3}GB "
        #            f"({optimal_gpu['free_ratio']*100:.1f}%)")
        
        # # Warn when GPU0 is chosen even though other devices exist.
        # if optimal_gpu['device_id'] == 0 and len(gpu_info) > 1:
        #     logger.warning("GPU0 was selected and may compete with LLM workloads")
        
        # return device_str
        
    except ImportError:
        logger.warning("PyTorch is not installed; using the default CUDA device")
        return "cuda"
    except Exception as e:
        logger.error("GPU detection failed: %s; using the default CUDA device", e)
        return "cuda"

def get_hf_embedding_model(
    name,
    timeout_config: TimeoutConfig = TimeoutConfig(),
    total_chunks: int = 0,
    device: EmbeddingDeviceType = None,
) -> BaseEmbedding | None:
    """Use generic LLamaIndex HuggingFaceEmbedding model.

    name: name of the embedding model
    device: Device type to pass in. LlamaIndex uses Torch to autodiscover if None
    """
    if not name:
        return None
    if device == "onnx-cpu":
        device = None
    
    # Resolve the actual device when generic "cuda" is requested.
    if device == "cuda":
        actual_device = get_optimal_gpu_device()
        logger.info("Embedding model %s will use device %s", name, actual_device)
    else:
        actual_device = device

    logger.debug(f"Getting HuggingFace model '{name}'")
    logger.info("Loading HuggingFace embedding model")
    logger.info("Model path: %s", name)
    logger.info("Device: %s", actual_device)
    
    # import pdb
    # pdb.set_trace()
    # print("\n\n\n\n\n\n\n\n\n\n-----------------cfg.paths.huggingface_cache.as_posix():----------------------\n",cfg.paths.huggingface_cache.as_posix())
    # print("-------------------------------------------------------------------")
    model = HuggingFaceEmbeddingWithTimeout(
        model_name=name,
        device=actual_device,
        trust_remote_code=True,
        # cache_folder=cfg.paths.huggingface_cache.as_posix(),
        timeout_config=timeout_config,
        total_chunks=total_chunks,
        use_auth_token=get_hf_token(),
    )
    
    logger.info("HuggingFace embedding model loaded: %s", type(model).__name__)
    return model

def get_onnx_embedding_model(
    name: str, timeout_config: TimeoutConfig = TimeoutConfig(), total_chunks: int = 0
) -> BaseEmbedding | None:
    if not name:
        return None
    logger.debug("Getting ONNX version of '%s'", name)
    model_folder = cfg.paths.onnx_dir / slugify(name)
    model_folder.mkdir(parents=True, exist_ok=True)
    try:
        if (model_folder / "model.onnx").exists():
            logger.debug("ONNX model for %s already exists. Skipping creation.", name)
        else:
            logger.info("Creating ONNX version of '%s'", name)
            write_lock = model_folder / "write.lock"
            with FileLock(write_lock, timeout=timeout_config.onnx_timeout):
                model = ORTModelForFeatureExtraction.from_pretrained(
                    name,
                    export=True,
                    trust_remote_code=True,
                    use_auth_token=get_hf_token(),
                )
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                model.save_pretrained(model_folder)
                tokenizer.save_pretrained(model_folder)
            logger.info("Done creating ONNX version of '%s'", name)

        return OptimumEmbeddingWithTimeout(
            folder_name=model_folder.as_posix(),
            timeout_config=timeout_config,
            total_chunks=total_chunks,
        )
    except (ValueError, Timeout):
        logger.warning("Cannot get ONNX version of '%s'", name)
        write_lock.unlink(missing_ok=True)
        raise

def get_embedding_model(
    name: str,
    timeout_config: TimeoutConfig = TimeoutConfig(),
    total_chunks: int = 0,
    device: EmbeddingDeviceType = "cpu",
    use_hf_endpoint_models: bool = False,
) -> T.Tuple[BaseEmbedding | None, bool | None]:
    """
    Returns an embedding model based on the name and device type.
    4-stage fallback:
    1. check if the model can be served by a local vLLM Endpoint
    2. check if the model can be served by a dedicated HF endpoint
    3. if not try to get an onnx model with cpu backend
    4. if that fails, get a torch model
    """
    if not name:
        logger.warning("No embedding model name provided.")
        return None, None

    if name in LOCAL_EMBEDDING_MODELS:
        return LOCAL_EMBEDDING_MODELS[name], False

    if use_hf_endpoint_models and cfg.hf_embeddings.models_config_map.get(name, False):
        logger.info("Getting HF endpoint model: %s", name)
        logger.error("HF endpoint models are not supported any more.")
        # return get_hf_endpoint_embed_model(name), False
    elif device == "onnx-cpu":
        try:
            logger.info("Getting ONNX model: %s", name)
            return (
                get_onnx_embedding_model(
                    name, timeout_config=timeout_config, total_chunks=total_chunks
                ),
                True,
            )
        except ValueError:
            logger.info("Getting local HF model for CPU: %s", name)
            return (
                get_hf_embedding_model(
                    name,
                    timeout_config=timeout_config,
                    total_chunks=total_chunks,
                    device="cpu",
                ),
                False,
            )
    else:
        logger.info("Getting local HF model for device '%s': %s", device, name)
        return (
            get_hf_embedding_model(
                name,
                timeout_config=timeout_config,
                total_chunks=total_chunks,
                device=device,
            ),
            False,
        )
