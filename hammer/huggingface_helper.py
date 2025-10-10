
import os
import typing as T

from filelock import FileLock, Timeout
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from optimum.onnxruntime import ORTModelForFeatureExtraction
from slugify import slugify
from transformers import AutoTokenizer

from hammer.configuration import cfg
from hammer.embeddings.timeouts import EmbeddingTimeoutMixin
# from hammer.hf_endpoint_models import get_hf_endpoint_embed_model
from hammer.logger import logger
from hammer.studies import EmbeddingDeviceType, TimeoutConfig

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

class OpenAILikeEmbeddingWithTimeout(EmbeddingTimeoutMixin, OpenAILikeEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    if cfg.local_models.embedding
    else {}
)

class OptimumEmbeddingWithTimeout(EmbeddingTimeoutMixin, OptimumEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_optimal_gpu_device() -> str:
    """智能检测GPU使用情况，选择显存最多的GPU设备。
    
    支持通过环境变量指定特定GPU：
    - HAMMER_MCTS_GPU: MCTS实验专用GPU (默认: 5)
    - HAMMER_TPE_GPU: TPE实验专用GPU (默认: 7)
    """
    try:
        import torch
        import os
        import inspect
        
        # 🔧 内存碎片化优化设置
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # 🎯 检查是否有强制指定的GPU设备
        # import pdb
        # pdb.set_trace()
        from hammer.tuner.main_tuner_mcts import DEVICE_ID as mcts_gpu_id
        # from hammer.tuner.main_tuner_tpe import DEVICE_ID as tpe_gpu_id
        # mcts_gpu = os.environ.get('HAMMER_MCTS_GPU', f'{mcts_gpu_id}')
        # tpe_gpu = os.environ.get('HAMMER_TPE_GPU', f'{tpe_gpu_id}')
        
        # # 通过调用栈判断当前是MCTS还是TPE实验
        # stack = inspect.stack()
        # is_mcts_experiment = any('mcts' in frame.filename.lower() for frame in stack)
        # is_tpe_experiment = any('tpe' in frame.filename.lower() for frame in stack)
        
        # # 根据实验类型分配GPU
        # if is_mcts_experiment:
        #     device_str = f"cuda:{mcts_gpu}"
        #     logger.info(f"🎯 MCTS实验：使用GPU {device_str}")
        #     return device_str
        # elif is_tpe_experiment:
        #     device_str = f"cuda:{tpe_gpu}"
        #     logger.info(f"🎯 TPE实验：使用GPU {device_str}")
        #     return device_str
        
        # if not torch.cuda.is_available():
        #     logger.info("💻 CUDA不可用，使用CPU设备")
        #     return "cpu"
        
        # # 🔧 如果既不是MCTS也不是TPE实验，使用默认的MCTS GPU设备
        # logger.info(f"💻 未明确识别实验类型，默认使用MCTS GPU: cuda:{mcts_gpu}")
        return f"cuda:{mcts_gpu_id}"
        
        # gpu_count = torch.cuda.device_count()
        # if gpu_count == 0:
        #     logger.info("💻 未检测到GPU设备，使用CPU")
        #     return "cpu"
        # elif gpu_count == 1:
        #     logger.info("💻 单GPU环境，使用cuda:0")
        #     torch.cuda.empty_cache()
        #     return "cuda:0"
        
        # # 多GPU环境：智能选择最优GPU（仅从指定范围选择）
        # preferred_gpus = [4, 5, 6, 7]  # 用户指定的首选GPU范围
        # available_preferred = [gpu_id for gpu_id in preferred_gpus if gpu_id < gpu_count]
        
        # if available_preferred:
        #     logger.info(f"💻 多GPU环境：检测到{gpu_count}个GPU，从首选GPU {preferred_gpus} 中智能选择...")
        #     gpu_candidates = available_preferred
        # else:
        #     logger.info(f"💻 多GPU环境：检测到{gpu_count}个GPU，首选GPU不可用，智能选择...")
        #     gpu_candidates = list(range(gpu_count))
        
        # gpu_info = []
        # current_device = torch.cuda.current_device()  # 保存当前设备
        
        # for i in gpu_candidates:
        #     try:
        #         # 获取GPU属性
        #         props = torch.cuda.get_device_properties(i)
        #         total_memory = props.total_memory
                
        #         # 临时设置到该GPU获取内存信息
        #         torch.cuda.set_device(i)
                
        #         # 🔧 清理内存碎片
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
        #                     f"总显存={total_memory//1024**3}GB, "
        #                     f"空闲={free_memory//1024**3}GB ({free_memory/total_memory*100:.1f}%)")
                            
        #     except Exception as e:
        #         logger.warning(f"GPU {i} 检测失败: {e}")
        #         continue
        
        # # 恢复原来的设备
        # torch.cuda.set_device(current_device)
        
        # if not gpu_info:
        #     logger.warning("所有GPU检测失败，使用默认CUDA设备")
        #     return "cuda"
        
        # # 选择策略：优先选择空闲显存最多的GPU
        # # 优先选择首选GPU范围内的GPU
        # optimal_gpu = max(gpu_info, key=lambda x: (
        #     x['device_id'] in preferred_gpus,  # 首选GPU优先级
        #     x['free_ratio'],                   # 空闲显存比例
        #     x['device_id'] != 0                # 避开GPU0
        # ))
        
        # device_str = f"cuda:{optimal_gpu['device_id']}"
        # logger.info(f"🏆 智能选择GPU {optimal_gpu['device_id']} ({optimal_gpu['name']}): "
        #            f"空闲显存={optimal_gpu['free_memory']//1024**3}GB "
        #            f"({optimal_gpu['free_ratio']*100:.1f}%)")
        
        # # 特殊情况警告：如果选中GPU0且存在其他GPU
        # if optimal_gpu['device_id'] == 0 and len(gpu_info) > 1:
        #     logger.warning("⚠️  选中了GPU0，可能与LLM竞争资源，建议检查GPU负载")
        
        # return device_str
        
    except ImportError:
        logger.warning("💻 PyTorch未安装，使用默认CUDA设备")
        return "cuda"
    except Exception as e:
        logger.error(f"💻 GPU检测失败: {e}，使用默认CUDA设备")
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
    
    # 🔧 智能GPU资源分配：使用最优GPU设备
    if device == "cuda":
        actual_device = get_optimal_gpu_device()
        logger.info(f"🔧 智能GPU分配: embedding模型 '{name}' 使用 {actual_device}")
    else:
        actual_device = device

    logger.debug(f"Getting HuggingFace model '{name}'")
    # 🔥 增强日志：显示完整的模型加载信息
    logger.info(f"🤗 正在加载HuggingFace Embedding模型")
    logger.info(f"📍 模型路径: {name}")
    logger.info(f"🖥️  设备: {actual_device}")
    
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
    
    # 🔥 新增：确认模型加载完成
    logger.info(f"✅ HuggingFace Embedding模型加载完成: {type(model).__name__}")
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
