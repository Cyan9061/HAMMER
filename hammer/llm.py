
import json
import os
import typing as T
from json import JSONDecodeError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import tiktoken
from google.cloud.aiplatform_v1beta1.types import content
from google.oauth2 import service_account
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai_like import OpenAILike

try:
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
except ImportError:  # pragma: no cover - optional provider
    AnthropicVertex = AsyncAnthropicVertex = None  # type: ignore[assignment]

try:
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
except ImportError:  # pragma: no cover - optional provider
    AzureOpenAIEmbedding = None  # type: ignore[assignment]

try:
    from llama_index.llms.anthropic import Anthropic
except ImportError:  # pragma: no cover - optional provider
    Anthropic = None  # type: ignore[assignment]

try:
    from llama_index.llms.azure_inference import AzureAICompletionsModel
except ImportError:  # pragma: no cover - optional provider
    AzureAICompletionsModel = None  # type: ignore[assignment]

try:
    from llama_index.llms.azure_openai import AzureOpenAI
except ImportError:  # pragma: no cover - optional provider
    AzureOpenAI = None  # type: ignore[assignment]

try:
    from llama_index.llms.cerebras import Cerebras
except ImportError:  # pragma: no cover - optional provider
    Cerebras = None  # type: ignore[assignment]

try:
    from llama_index.llms.vertex import Vertex
except ImportError:  # pragma: no cover - optional provider
    Vertex = None  # type: ignore[assignment]

from transformers import AutoTokenizer
# Cache the tokenizer lazily to avoid remote downloads at import time.
_qwen_tokenizer = None

# Import the custom SiliconFlow LLM wrapper.
from hammer.siliconflow_llm import create_siliconflow_llm, get_api_keys
from mypy_extensions import DefaultNamedArg

from hammer.configuration import (
    NON_OPENAI_CONTEXT_WINDOW_FACTOR,
    AnthropicVertexLLM,
    AzureAICompletionsLLM,
    AzureOpenAILLM,
    CerebrasLLM,
    OpenAILikeLLM,
    Settings,
    VertexAILLM,
    cfg,
)
from hammer.logger import logger
# from hammer.patches import _get_all_kwargs

# Anthropic._get_all_kwargs = _get_all_kwargs  # type: ignore


def _get_qwen_tokenizer():
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        try:
            _qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        except Exception as exc:
            logger.warning("Failed to load Qwen tokenizer, falling back to cl100k_base: %s", exc)
            return None
    return _qwen_tokenizer

def _scale(
    context_window_length: int, factor: float = NON_OPENAI_CONTEXT_WINDOW_FACTOR
) -> int:
    return int(context_window_length * factor)


def _safe_construct_optional(
    dependency: T.Any,
    label: str,
    factory: T.Callable[[], T.Any],
):
    if dependency is None:
        logger.debug("Optional dependency for %s is unavailable; skipping.", label)
        return None
    try:
        return factory()
    except Exception as exc:
        logger.debug("Skipping optional component %s: %s", label, exc)
        return None

if (hf_token := cfg.hf_embeddings.api_key.get_secret_value()) != "NOT SET":
    os.environ["HF_TOKEN"] = hf_token

LOCAL_MODELS = (
    {
        model.model_name: OpenAILike(  # type: ignore
            api_base=str(model.api_base),
            api_key=model.api_key.get_secret_value()
            if model.api_key is not None
            else cfg.local_models.default_api_key.get_secret_value(),
            model=model.model_name,
            max_tokens=model.max_tokens,
            context_window=_scale(model.context_window),
            is_chat_model=model.is_chat_model,
            is_function_calling_model=model.is_function_calling_model,
            timeout=model.timeout,
            max_retries=model.max_retries,
            additional_kwargs=model.additional_kwargs,
        )
        for model in cfg.local_models.generative
    }
    if cfg.local_models.generative
    else {}
)

AZURE_GPT35_TURBO = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_GPT35_TURBO",
    lambda: AzureOpenAI(
        model="gpt-3.5-turbo",
        deployment_name="gpt-35",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-06-01",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_GPT4O_MINI = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_GPT4O_MINI",
    lambda: AzureOpenAI(
        model="gpt-4o-mini",
        deployment_name="gpt-4o-mini",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-06-01",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_GPT4O_STD = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_GPT4O_STD",
    lambda: AzureOpenAI(
        model="gpt-4o",
        deployment_name="gpt-4o",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-06-01",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_o1 = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_o1",
    lambda: AzureOpenAI(
        model="o1",
        deployment_name="o1",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-12-01-preview",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_o3_MINI = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_o3_MINI",
    lambda: AzureOpenAI(
        model="o3-mini",
        deployment_name="o3-mini",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-12-01-preview",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_GPT35_TURBO_1106 = _safe_construct_optional(
    AzureOpenAI,
    "AZURE_GPT35_TURBO_1106",
    lambda: AzureOpenAI(
        model="gpt-35-turbo",
        deployment_name="gpt-35-turbo",
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-06-01",
        temperature=0,
        max_retries=0,
        additional_kwargs={"user": "hammer"},
    ),
)

AZURE_TEXT_EMBEDDING_ADA_002 = _safe_construct_optional(
    AzureOpenAIEmbedding,
    "AZURE_TEXT_EMBEDDING_ADA_002",
    lambda: AzureOpenAIEmbedding(
        model=os.getenv("DEFAULT_EMBEDDING_MODEL_ALT", ""),
        deployment_name=os.getenv("DEFAULT_EMBEDDING_MODEL_ALT", ""),
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2023-03-15-preview",
    ),
)

AZURE_TEXT_EMBEDDING_3_LARGE = _safe_construct_optional(
    AzureOpenAIEmbedding,
    "AZURE_TEXT_EMBEDDING_3_LARGE",
    lambda: AzureOpenAIEmbedding(
        model=os.getenv("DEFAULT_EMBEDDING_MODEL", ""),
        deployment_name=os.getenv("DEFAULT_EMBEDDING_MODEL", ""),
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=str(cfg.azure_oai.api_url),
        api_version="2024-06-01",
    ),
)

GCP_SAFETY_SETTINGS = {
    content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

try:
    GCP_CREDS = json.loads(cfg.gcp_vertex.credentials.get_secret_value())
except JSONDecodeError:
    GCP_CREDS = {}

GCP_GEMINI_PRO = _safe_construct_optional(
    Vertex,
    "GCP_GEMINI_PRO",
    lambda: Vertex(
        model="gemini-1.5-pro-002",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        max_tokens=8000,
        context_window=_scale(2000000),
        max_retries=0,
        additional_kwargs={},
    ),
)

GCP_GEMINI_FLASH = _safe_construct_optional(
    Vertex,
    "GCP_GEMINI_FLASH",
    lambda: Vertex(
        model="gemini-1.5-flash-002",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        context_window=_scale(1048000),
        max_tokens=8000,
        max_retries=0,
        additional_kwargs={},
    ),
)

GCP_GEMINI_FLASH_EXP = _safe_construct_optional(
    Vertex,
    "GCP_GEMINI_FLASH_EXP",
    lambda: Vertex(
        model="gemini-2.0-flash-lite-preview-02-05",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        max_tokens=8000,
        context_window=_scale(1048000),
        max_retries=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        additional_kwargs={},
    ),
)

if Vertex is not None:
    class VertexFlashThink(Vertex):
        def __init__(
            self,
            model: str = "text-bison",
            project: T.Optional[str] = None,
            location: T.Optional[str] = None,
            credentials: T.Optional[T.Any] = None,
            **kwargs,
        ):
            super().__init__(
                model=model,
                project=project,
                location=location,
                credentials=credentials,
                **kwargs,
            )

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                num_output=self.max_tokens,
                context_window=self.context_window,
                is_chat_model=self._is_chat_model,
                is_function_calling_model=False,
                model_name=self.model,
                system_role=(
                    MessageRole.USER if self._is_gemini else MessageRole.SYSTEM
                ),
            )
else:
    VertexFlashThink = None  # type: ignore[assignment]

GCP_GEMINI_FLASH_THINK_EXP = _safe_construct_optional(
    VertexFlashThink,
    "GCP_GEMINI_FLASH_THINK_EXP",
    lambda: VertexFlashThink(
        model="gemini-2.0-flash-thinking-exp-01-21",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        max_tokens=8000,
        context_window=_scale(32000),
        max_retries=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        additional_kwargs={},
    ),
)

GCP_GEMINI_PRO_EXP = _safe_construct_optional(
    Vertex,
    "GCP_GEMINI_PRO_EXP",
    lambda: Vertex(
        model="gemini-2.0-pro-exp-02-05",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        max_tokens=8000,
        context_window=_scale(1048000),
        max_retries=0,
        additional_kwargs={},
    ),
)

GCP_GEMINI_FLASH2 = _safe_construct_optional(
    Vertex,
    "GCP_GEMINI_FLASH2",
    lambda: Vertex(
        model="gemini-2.0-flash-001",
        project=cfg.gcp_vertex.project_id,
        credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {},
        temperature=0,
        max_tokens=8000,
        context_window=_scale(1048000),
        max_retries=0,
        safety_settings=GCP_SAFETY_SETTINGS,
        additional_kwargs={},
    ),
)

# TogetherLlamaSmall = OpenAILike(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#     api_base="https://api.together.xyz/v1",
#     api_key=cfg.togetherai.api_key.get_secret_value(),
#     api_version=None,  # type: ignore
#     max_tokens=2000,
#     context_window=_scale(131072),
#     is_chat_model=True,
#     is_function_calling_model=True,
#     timeout=3600,
#     max_retries=0,
# )

# Use SiliconFlow-backed hosted models instead of a local Ollama runtime.
Qwen_2_5_7b = create_siliconflow_llm(
    api_keys=get_api_keys(),
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_tokens=2000,
    temperature=0,
    timeout=60,
    max_retries=3
)

# Qwen2-7B legacy variant via SiliconFlow.
Qwen_2_7b = create_siliconflow_llm(
    api_keys=get_api_keys(),
    model_name="Qwen/Qwen2-7B-Instruct",
    max_tokens=2000,
    temperature=0,
    timeout=60,
    max_retries=3
)

# DeepSeek-R1-32B via SiliconFlow.
Deepseek_R1_32b = create_siliconflow_llm(
    api_keys=get_api_keys(),
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    max_tokens=4000,
    temperature=0,
    timeout=120,
    max_retries=3
)

# Qwen2.5-72B via SiliconFlow.
Qwen_2_5_72b = create_siliconflow_llm(
    api_keys=get_api_keys(),
    model_name="Qwen/Qwen2.5-72B-Instruct",
    max_tokens=4000,
    temperature=0,
    timeout=120,
    max_retries=3
)

# GPT-4o-mini via the configured Gaochao-compatible OpenAI-like endpoint.
GPT4o_mini = OpenAILike(
    model="gpt-4o-mini",
    api_base="https://api.ai-gaochao.cn/v1",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    max_tokens=4000,
    context_window=16384,
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=120,
    max_retries=3,
    temperature=0
)

TogetherDeepseekR1 = OpenAILike(
    model="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=5000,
    context_window=_scale(16384),
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)

TogetherDeepseekV3 = OpenAILike(
    model="deepseek-ai/DeepSeek-V3",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2048,
    context_window=_scale(16384),
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)

class DROpenAILike(OpenAILike):
    pass

DataRobotDeployedLLM = DROpenAILike(
    model="datarobot/model-name",
    api_base=str(cfg.datarobot.endpoint),
    api_key=cfg.datarobot.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2000,
    is_chat_model=True,
    context_window=_scale(14000),
    is_function_calling_model=True,
    timeout=3600,
    max_retries=0,
)

DRDeepseekLlama70BReasoning = DROpenAILike(
    model="datarobot/DeepSeek-Llama",
    api_base=str(cfg.datarobot.endpoint),
    api_key=cfg.datarobot.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=3000,
    is_chat_model=True,
    context_window=_scale(14000),
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)

def add_scoped_credentials_anthropic(anthropic_llm: Anthropic) -> Anthropic:
    """Add Google service account credentials to an Anthropic LLM"""
    if anthropic_llm is None or AnthropicVertex is None or AsyncAnthropicVertex is None:
        return anthropic_llm
    credentials = (
        service_account.Credentials.from_service_account_info(GCP_CREDS).with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        if GCP_CREDS
        else None
    )
    sync_client = anthropic_llm._client
    assert isinstance(sync_client, AnthropicVertex)
    sync_client.credentials = credentials
    anthropic_llm._client = sync_client
    async_client = anthropic_llm._aclient
    assert isinstance(async_client, AsyncAnthropicVertex)
    async_client.credentials = credentials
    anthropic_llm._aclient = async_client
    return anthropic_llm

ANTHROPIC_CLAUDE_SONNET_35 = _safe_construct_optional(
    Anthropic,
    "ANTHROPIC_CLAUDE_SONNET_35",
    lambda: add_scoped_credentials_anthropic(
        Anthropic(
            model="claude-3-5-sonnet-v2@20241022",
            project_id=str(cfg.gcp_vertex.project_id),
            region="us-east5",
            temperature=0,
        )
    ),
)

ANTHROPIC_CLAUDE_HAIKU_35 = _safe_construct_optional(
    Anthropic,
    "ANTHROPIC_CLAUDE_HAIKU_35",
    lambda: add_scoped_credentials_anthropic(
        Anthropic(
            model="claude-3-5-haiku@20241022",
            project_id=str(cfg.gcp_vertex.project_id),
            region="us-east5",
            temperature=0,
        )
    ),
)

if AzureAICompletionsModel is not None:
    class AzureAICompletionsModelLlama(AzureAICompletionsModel):
        def __init__(self, credential, model_name, endpoint, temperature=0):
            super().__init__(
                credential=credential,
                model_name=model_name,
                endpoint=endpoint,
                temperature=temperature,
            )

        @property
        def metadata(self):
            return LLMMetadata(
                context_window=120000,
                num_output=1000,
                is_chat_model=True,
                is_function_calling_model=False,
                model_name="Llama-3.3-70B-Instruct",
            )

    class AzureAICompletionsModelPhi4(AzureAICompletionsModel):
        def __init__(self, credential, model_name, endpoint, temperature=0):
            super().__init__(
                credential=credential,
                model_name=model_name,
                endpoint=endpoint,
                temperature=temperature,
            )

        @property
        def metadata(self):
            return LLMMetadata(
                context_window=14000,
                num_output=1000,
                is_chat_model=True,
                is_function_calling_model=False,
                model_name="Phi-4",
            )

    class AzureAICompletionsModelR1(AzureAICompletionsModel):
        def __init__(self, credential, model_name, endpoint, temperature=0):
            super().__init__(
                credential=credential,
                model_name=model_name,
                endpoint=endpoint,
                temperature=temperature,
            )

        @property
        def metadata(self):
            return LLMMetadata(
                context_window=120000,
                num_output=8000,
                is_chat_model=True,
                is_function_calling_model=False,
                model_name="Deepseek-R1",
            )

    class AzureAICompletionsModelMistral(AzureAICompletionsModel):
        def __init__(self, credential, model_name, endpoint, temperature=0):
            super().__init__(
                credential=credential,
                model_name=model_name,
                endpoint=endpoint,
                temperature=temperature,
            )

        @property
        def metadata(self):
            return LLMMetadata(
                context_window=120000,
                num_output=2056,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name="mistral-large-2411",
            )
else:
    AzureAICompletionsModelLlama = None  # type: ignore[assignment]
    AzureAICompletionsModelPhi4 = None  # type: ignore[assignment]
    AzureAICompletionsModelR1 = None  # type: ignore[assignment]
    AzureAICompletionsModelMistral = None  # type: ignore[assignment]

AZURE_LLAMA33_70B = _safe_construct_optional(
    AzureAICompletionsModelLlama,
    "AZURE_LLAMA33_70B",
    lambda: AzureAICompletionsModelLlama(
        credential=cfg.azure_inference_llama33.api_key.get_secret_value(),  # type: ignore[arg-type]
        model_name=str(cfg.azure_inference_llama33.model_name),  # type: ignore[arg-type]
        endpoint=(
            "https://"
            + str(cfg.azure_inference_llama33.default_deployment)
            + "."
            + str(cfg.azure_inference_llama33.region_name)
            + ".models.ai.azure.com"
        ),
        temperature=0,  # type: ignore[arg-type]
    ),
)

AZURE_PHI4 = _safe_construct_optional(
    AzureAICompletionsModelPhi4,
    "AZURE_PHI4",
    lambda: AzureAICompletionsModelPhi4(
        credential=cfg.azure_inference_phi4.api_key.get_secret_value(),  # type: ignore[arg-type]
        model_name=str(cfg.azure_inference_phi4.model_name),  # type: ignore[arg-type]
        endpoint=(
            "https://"
            + str(cfg.azure_inference_phi4.default_deployment)
            + "."
            + str(cfg.azure_inference_phi4.region_name)
            + ".models.ai.azure.com"
        ),
        temperature=0,  # type: ignore[arg-type]
    ),
)

AZURE_R1 = _safe_construct_optional(
    AzureAICompletionsModelR1,
    "AZURE_R1",
    lambda: AzureAICompletionsModelR1(
        credential=cfg.azure_inference_r1.api_key.get_secret_value(),  # type: ignore[arg-type]
        model_name=str(cfg.azure_inference_r1.model_name),  # type: ignore[arg-type]
        endpoint=(
            "https://"
            + str(cfg.azure_inference_r1.default_deployment)
            + "."
            + str(cfg.azure_inference_r1.region_name)
            + ".models.ai.azure.com"
        ),
        temperature=0,  # type: ignore[arg-type]
    ),
)

MISTRAL_LARGE = _safe_construct_optional(
    AzureAICompletionsModelMistral,
    "MISTRAL_LARGE",
    lambda: AzureAICompletionsModelMistral(
        credential=cfg.azure_inference_mistral.api_key.get_secret_value(),  # type: ignore[arg-type]
        model_name=str(cfg.azure_inference_mistral.model_name),  # type: ignore[arg-type]
        endpoint=(
            "https://"
            + str(cfg.azure_inference_mistral.default_deployment)
            + "."
            + str(cfg.azure_inference_mistral.region_name)
            + ".models.ai.azure.com"
        ),
        temperature=0,  # type: ignore[arg-type]
    ),
)

CEREBRAS_LLAMA_31_8B = _safe_construct_optional(
    Cerebras,
    "CEREBRAS_LLAMA_31_8B",
    lambda: Cerebras(
        model="llama3.1-8b",
        api_key=cfg.cerebras.api_key.get_secret_value(),
    ),
)

CEREBRAS_LLAMA_33_70B = _safe_construct_optional(
    Cerebras,
    "CEREBRAS_LLAMA_33_70B",
    lambda: Cerebras(
        model="llama-3.3-70b",
        api_key=cfg.cerebras.api_key.get_secret_value(),
        is_function_calling_model=False,
        context_window=8000,
    ),
)

def _construct_azure_openai_llm(name: str, llm_config: AzureOpenAILLM) -> AzureOpenAI:
    return AzureOpenAI(
        model=llm_config.metadata.model_name,
        deployment_name=llm_config.deployment_name or llm_config.metadata.model_name,
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=cfg.azure_oai.api_url.unicode_string(),
        api_version=llm_config.api_version or cfg.azure_oai.api_version,
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )

def _construct_vertex_ai_llm(name: str, llm_config: VertexAILLM) -> Vertex:
    credentials = (
        service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {}
    )
    return Vertex(
        model=llm_config.model or llm_config.metadata.model_name,
        project=cfg.gcp_vertex.project_id,
        credentials=credentials,
        temperature=llm_config.temperature,
        safety_settings=llm_config.safety_settings or GCP_SAFETY_SETTINGS,
        max_tokens=llm_config.metadata.num_output,
        context_window=_scale(llm_config.metadata.context_window),
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
        location=cfg.gcp_vertex.region,
    )

def _construct_anthropic_vertex_llm(
    name: str, llm_config: AnthropicVertexLLM
) -> Anthropic:
    anthropic_llm = Anthropic(
        model=llm_config.model,
        project_id=llm_config.project_id or cfg.gcp_vertex.project_id,
        region=llm_config.region or cfg.gcp_vertex.region,
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )
    return add_scoped_credentials_anthropic(anthropic_llm)

def _construct_azure_ai_completions_llm(
    name: str, llm_config: AzureAICompletionsLLM
) -> AzureAICompletionsModel:
    return AzureAICompletionsModel(
        credential=llm_config.api_key.get_secret_value(),
        endpoint=llm_config.endpoint.unicode_string(),
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        metadata=llm_config.metadata.model_dump(),
    )

def _construct_cerebras_llm(name: str, llm_config: CerebrasLLM) -> Cerebras:
    return Cerebras(
        model=llm_config.model,
        api_key=cfg.cerebras.api_key.get_secret_value(),
        api_base=cfg.cerebras.api_url.unicode_string(),
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        context_window=llm_config.metadata.context_window,  # Use raw value as per existing Cerebras configs
        is_function_calling_model=llm_config.metadata.is_function_calling_model,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )

def _construct_openai_like_llm(name: str, llm_config: OpenAILikeLLM) -> OpenAILike:
    return OpenAILike(
        model=llm_config.model,
        api_base=str(llm_config.api_base),
        api_key=llm_config.api_key.get_secret_value(),
        api_version=llm_config.api_version,  # type: ignore
        max_tokens=llm_config.metadata.num_output,
        context_window=_scale(llm_config.metadata.context_window),
        is_chat_model=llm_config.metadata.is_chat_model,
        is_function_calling_model=llm_config.metadata.is_function_calling_model,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )

def load_configured_llms(config: Settings) -> T.Dict[str, FunctionCallingLLM]:
    _dynamically_loaded_llms: T.Dict[str, FunctionCallingLLM] = {}
    if not config.generative_models:
        return {}
    logger.debug(
        f"Loading LLMs from 'generative_models' configuration: {list(config.generative_models.keys())}"
    )
    for name, llm_config_instance in config.generative_models.items():
        llm_instance: T.Optional[FunctionCallingLLM] = None
        try:
            provider = getattr(llm_config_instance, "provider", None)

            if provider == "azure_openai" and isinstance(
                llm_config_instance, AzureOpenAILLM
            ):
                llm_instance = _construct_azure_openai_llm(name, llm_config_instance)
            elif provider == "vertex_ai" and isinstance(
                llm_config_instance, VertexAILLM
            ):
                llm_instance = _construct_vertex_ai_llm(name, llm_config_instance)
            elif provider == "anthropic_vertex" and isinstance(
                llm_config_instance, AnthropicVertexLLM
            ):
                llm_instance = _construct_anthropic_vertex_llm(
                    name, llm_config_instance
                )
            elif provider == "azure_ai" and isinstance(
                llm_config_instance, AzureAICompletionsLLM
            ):
                llm_instance = _construct_azure_ai_completions_llm(
                    name, llm_config_instance
                )
            elif provider == "cerebras" and isinstance(
                llm_config_instance, CerebrasLLM
            ):
                llm_instance = _construct_cerebras_llm(name, llm_config_instance)
            elif provider == "openai_like" and isinstance(
                llm_config_instance, OpenAILikeLLM
            ):
                llm_instance = _construct_openai_like_llm(name, llm_config_instance)
            else:
                raise ValueError(
                    f"Unsupported provider type '{provider}' or "
                    f"mismatched Pydantic config model type for model '{name}'."
                )
                continue

            if llm_instance:
                _dynamically_loaded_llms[name] = llm_instance
                logger.debug(f"Successfully loaded LLM '{name}' from configuration.")
        except Exception as e:
            # Log with traceback for easier debugging
            logger.error(
                f"Failed to load configured LLM '{name}' due to: {e}", exc_info=True
            )
            raise
    return _dynamically_loaded_llms

# When adding a model, keep the registry and tests in sync.
LLMs = {
    # SiliconFlow-hosted models.
    "Qwen2_5-7b": Qwen_2_5_7b,
    "Qwen2-7b": Qwen_2_7b,
    "DeepSeek-R1-32b": Deepseek_R1_32b,
    "Qwen2.5-72b": Qwen_2_5_72b,
    
    # Gaochao-backed OpenAI-compatible model.
    "gpt-4o-mini": GPT4o_mini,
    
    **LOCAL_MODELS,
}

LLMs = {name: llm for name, llm in LLMs.items() if llm is not None}

LLMs.update(load_configured_llms(cfg))

def get_llm(name: str | None = None):
    if not name:
        logger.warning("No LLM name specified.")
        return None
    assert name in LLMs, (
        f"Invalid LLM name specified: {name}. Valid options are: {list(LLMs.keys())}"
    )
    return LLMs[name]

def get_llm_name(llm: LLM | FunctionCallingLLM | None = None):
    for llm_name, llm_instance in LLMs.items():
        if llm == llm_instance:
            return llm_name
    raise ValueError("Invalid LLM specified")

def is_function_calling(llm: LLM):
    try:
        if getattr(llm.metadata, "is_function_calling_model", False):
            if "flash" in llm.metadata.model_name:
                return False
            return True
    except ValueError:
        return False

def get_tokenizer(
    name: str,
) -> T.Callable[
    [
        str,
        DefaultNamedArg(T.Literal["all"] | T.AbstractSet[str], "allowed_special"),
        DefaultNamedArg(T.Literal["all"] | T.Collection[str], "disallowed_special"),
    ],
    list[int],
]:
    # GPT-4o-mini uses the OpenAI tokenizer.
    if name == "gpt-4o-mini":
        return tiktoken.encoding_for_model("gpt-4o-mini").encode
    
    # Local OpenAI-like models default to the GPT-4o-mini tokenizer.
    if name in LOCAL_MODELS:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode
    
    # Qwen and DeepSeek SiliconFlow models use the Qwen tokenizer when available.
    if name in ["Qwen2_5-7b", "Qwen2-7b", "Qwen2.5-72b", "DeepSeek-R1-32b"]:
        tokenizer = _get_qwen_tokenizer()
        if tokenizer is not None:
            return tokenizer.encode
        return tiktoken.get_encoding("cl100k_base").encode
        
    raise ValueError(f"Invalid tokenizer specified: {name}. Available models: {list(LLMs.keys())}")

if __name__ == "__main__":
    # Smoke-test the model registry.
    print(f"Available models: {list(LLMs.keys())}")
    print(f"Qwen 7B model: {get_llm_name(Qwen_2_5_7b)}")
