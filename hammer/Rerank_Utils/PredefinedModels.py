# https://github.com/DataScienceUIBK/Rankify/blob/main/rankify/utils/pre_defind_models.py
HF_PRE_DEFIND_MODELS ={
    'upr':{
        't5-small':'google/t5-small-lm-adapt',
        't5-base':'google/t5-base-lm-adapt',
        't5-large':'google/t5-large-lm-adapt',
        't0-3b':'bigscience/T0_3B',
        't0-11b':'bigscience/T0',
        'gpt-neo-2.7b':'EleutherAI/gpt-neo-2.7B',
        'gpt-j-6b':'EleutherAI/gpt-j-6b',
        'gpt2':'openai-community/gpt2',
        'gpt2-medium':'openai-community/gpt2-medium',
        'gpt2-large':'openai-community/gpt2-large',
        'gpt2-xl':'openai-community/gpt2-xl',
        
    },

    'flashrank':{
        "ms-marco-TinyBERT-L-2-v2": "ms-marco-TinyBERT-L-2-v2",
        "ms-marco-MiniLM-L-12-v2": "ms-marco-MiniLM-L-12-v2",
        "ms-marco-MultiBERT-L-12": "ms-marco-MultiBERT-L-12",
        "rank-T5-flan": "rank-T5-flan",
        "ce-esci-MiniLM-L12-v2": "ce-esci-MiniLM-L12-v2",
        "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full",
        "miniReranker_arabic_v1": "miniReranker_arabic_v1",
        
    },
    'flashrank-model-file':{
        "ms-marco-TinyBERT-L-2-v2": "flashrank-TinyBERT-L-2-v2.onnx",
        "ms-marco-MiniLM-L-12-v2": "flashrank-MiniLM-L-12-v2_Q.onnx",
        "ms-marco-MultiBERT-L-12": "flashrank-MultiBERT-L12_Q.onnx",
        "rank-T5-flan": "flashrank-rankt5_Q.onnx",
        "ce-esci-MiniLM-L12-v2": "flashrank-ce-esci-MiniLM-L12-v2_Q.onnx",
        "rank_zephyr_7b_v1_full": "rank_zephyr_7b_v1_full.Q4_K_M.gguf",
        "miniReranker_arabic_v1": "miniReranker_arabic_v1.onnx",
        
    },
    'monot5':{
        "monot5-base-msmarco": "castorini/monot5-base-msmarco",
        "monot5-base-msmarco-10k":"castorini/monot5-base-msmarco-10k",
        "monot5-large-msmarco": "castorini/monot5-large-msmarco",
        "monot5-large-msmarco-10k": "castorini/monot5-large-msmarco-10k",
        "monot5-base-med-msmarco": "castorini/monot5-base-med-msmarco",
        "monot5-3b-med-msmarco": "castorini/monot5-3b-med-msmarco",
        "monot5-3b-msmarco-10k": "castorini/monot5-3b-msmarco-10k",
        "mt5-base-en-msmarco": "unicamp-dl/mt5-base-en-msmarco",
        "ptt5-base-pt-msmarco-10k-v2": "unicamp-dl/ptt5-base-pt-msmarco-10k-v2",
        "ptt5-base-pt-msmarco-100k-v2": "unicamp-dl/ptt5-base-pt-msmarco-100k-v2",
        "ptt5-base-en-pt-msmarco-100k-v2": "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2",
        "mt5-base-en-pt-msmarco-v2": "unicamp-dl/mt5-base-en-pt-msmarco-v2",
        "mt5-base-mmarco-v2": "unicamp-dl/mt5-base-mmarco-v2",
        "mt5-base-en-pt-msmarco-v1": "unicamp-dl/mt5-base-en-pt-msmarco-v1",
        "mt5-base-mmarco-v1": "unicamp-dl/mt5-base-mmarco-v1",
        "ptt5-base-pt-msmarco-10k-v1": "unicamp-dl/ptt5-base-pt-msmarco-10k-v1",
        "ptt5-base-pt-msmarco-100k-v1": "unicamp-dl/ptt5-base-pt-msmarco-100k-v1",
        "ptt5-base-en-pt-msmarco-10k-v1": "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1",
        "mt5-3B-mmarco-en-pt": "unicamp-dl/mt5-3B-mmarco-en-pt",
        "mt5-13b-mmarco-100k": "unicamp-dl/mt5-13b-mmarco-100k",
        "monoptt5-small": "unicamp-dl/monoptt5-small",
        "monoptt5-base": "unicamp-dl/monoptt5-base",
        "monoptt5-large": "unicamp-dl/monoptt5-large",
        "monoptt5-3b": "unicamp-dl/monoptt5-3b",
        
    },
    'rankt5':{
        'rankt5-base': 'Soyoung97/RankT5-base',
        'rankt5-large': 'Soyoung97/RankT5-large',
        'rankt5-3b': 'Soyoung97/RankT5-3b',
        
    },
    'listt5':{
        'listt5-base':'Soyoung97/ListT5-base',
        'listt5-3b': 'Soyoung97/ListT5-3b',
       
    },
    'inranker':{
        'inranker-small': 'unicamp-dl/InRanker-small',
        'inranker-base' :'unicamp-dl/InRanker-base',
        'inranker-3b':'unicamp-dl/InRanker-3B',
        
    },
    'apiranker':{
        "cohere":"cohere",
        "jina":"jina",
        "voyage":"voyage",
        "mixedbread.ai":"mixedbread.ai",
    },
    'transformer_ranker': {
        "mxbai-rerank-xsmall":"mixedbread-ai/mxbai-rerank-xsmall-v1",
        "mxbai-rerank-base": "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-rerank-large": "mixedbread-ai/mxbai-rerank-large-v1",
        "bge-reranker-base":"BAAI/bge-reranker-base",
        "bge-reranker-large":"BAAI/bge-reranker-large",
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        "bce-reranker-base" :"maidalun1020/bce-reranker-base_v1",
        "jina-reranker-tiny":'jinaai/jina-reranker-v1-tiny-en',
        "jina-reranker-turbo":"jinaai/jina-reranker-v1-turbo-en",
        "jina-reranker-base-multilingual":"jinaai/jina-reranker-v2-base-multilingual",
        "gte-multilingual-reranker-base":"Alibaba-NLP/gte-multilingual-reranker-base",
        "camembert-base-mmarcoFR":"antoinelouis/crossencoder-camembert-base-mmarcoFR",
        "camembert-large-mmarcoFR":"antoinelouis/crossencoder-camembert-large-mmarcoFR",
        "camemberta-base-mmarcoFR":"antoinelouis/crossencoder-camemberta-base-mmarcoFR",
        "distilcamembert-mmarcoFR":"antoinelouis/crossencoder-distilcamembert-mmarcoFR",
        "cross-encoder-mmarco-mMiniLMv2-L12-H384-v1":"corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1",
        "nli-deberta-v3-large":"cross-encoder/nli-deberta-v3-large",
        "ms-marco-MiniLM-L-12-v2":"cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-MiniLM-L-6-v2":"cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-4-v2":"cross-encoder/ms-marco-MiniLM-L-4-v2",
        "ms-marco-MiniLM-L-2-v2":"cross-encoder/ms-marco-MiniLM-L-2-v2",
        "ms-marco-TinyBERT-L-2-v2":"cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "ms-marco-electra-base":"cross-encoder/ms-marco-electra-base",
        "ms-marco-TinyBERT-L-6":"cross-encoder/ms-marco-TinyBERT-L-6",
        "ms-marco-TinyBERT-L-4":"cross-encoder/ms-marco-TinyBERT-L-4",
        "ms-marco-TinyBERT-L-2":"cross-encoder/ms-marco-TinyBERT-L-2",
        "msmarco-MiniLM-L12-en-de-v1":"cross-encoder/msmarco-MiniLM-L12-en-de-v1",
        "msmarco-MiniLM-L6-en-de-v1":"cross-encoder/msmarco-MiniLM-L6-en-de-v1",
        
    },
    'llm_layerwise_ranker':{
        "bge-multilingual-gemma2":"BAAI/bge-multilingual-gemma2",
        "bge-reranker-v2-gemma":"BAAI/bge-reranker-v2-gemma",
        "bge-reranker-v2-minicpm-layerwise":"BAAI/bge-reranker-v2-minicpm-layerwise",
        "bge-reranker-v2.5-gemma2-lightweight":"BAAI/bge-reranker-v2.5-gemma2-lightweight",
        
    },
    'first_ranker':{
        "First-Model":"rryisthebest/First_Model",
        "Llama-3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
        
    },
    'lit5dist':{
        "LiT5-Distill-base": "castorini/LiT5-Distill-base",
        "LiT5-Distill-large":	"castorini/LiT5-Distill-large",
        "LiT5-Distill-xl":	"castorini/LiT5-Distill-xl",
        "LiT5-Distill-base-v2":	"castorini/LiT5-Distill-base-v2",
        "LiT5-Distill-large-v2":	"castorini/LiT5-Distill-large-v2",
        "LiT5-Distill-xl-v2":	"castorini/LiT5-Distill-xl-v2",
        
    },
    'lit5score':{
        "LiT5-Score-base":	"castorini/LiT5-Score-base",
        "LiT5-Score-large":	"castorini/LiT5-Score-large",
        "LiT5-Score-xl":	"castorini/LiT5-Score-xl",
        
    },
    'vicuna_reranker':{
        "rank_vicuna_7b_v1": "castorini/rank_vicuna_7b_v1",
        "rank_vicuna_7b_v1_noda":	"castorini/rank_vicuna_7b_v1_noda",
        "rank_vicuna_7b_v1_fp16":	"castorini/rank_vicuna_7b_v1_fp16",
        "rank_vicuna_7b_v1_noda_fp16":	"castorini/rank_vicuna_7b_v1_noda_fp16",
        
    },

    'splade_reranker':{
        "splade-cocondenser":"naver/splade-cocondenser-ensembledistil",
       
    },
    'sentence_transformer_reranker': {
        "all-MiniLM-L6-v2":"all-MiniLM-L6-v2",
        "gtr-t5-base":"sentence-transformers/gtr-t5-base",
        "gtr-t5-large":"sentence-transformers/gtr-t5-large",
        "gtr-t5-xl":"sentence-transformers/gtr-t5-xl",
        "gtr-t5-xxl":"sentence-transformers/gtr-t5-xxl",
        "sentence-t5-base":"sentence-transformers/sentence-t5-base",
        "sentence-t5-xl":"sentence-transformers/sentence-t5-xl",
        "sentence-t5-xxl":"sentence-transformers/sentence-t5-xxl",
        "sentence-t5-large":"sentence-transformers/sentence-t5-large",
        "distilbert-multilingual-nli-stsb-quora-ranking":"sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",
        "msmarco-bert-co-condensor":"sentence-transformers/msmarco-bert-co-condensor",
        "msmarco-roberta-base-v2":"sentence-transformers/msmarco-roberta-base-v2",
        
    },
    'colbert_ranker':{
        "Colbert": "colbert-ir/colbertv2.0",
        "FranchColBERT": "bclavie/FraColBERTv2",
        "JapanColBERT": "bclavie/JaColBERTv2",
        "SpanishColBERT": "AdrienB134/ColBERTv2.0-spanish-mmarcoES",
        'jina-colbert-v1-en': 'jinaai/jina-colbert-v1-en',
        'ArabicColBERT-250k':'akhooli/arabic-colbertv2-250k-norm',
        'ArabicColBERT-711k':'akhooli/arabic-colbertv2-711k-norm',
        'BengaliColBERT':'turjo4nis/colbertv2.0-bn',
        'mxbai-colbert-large-v1':'mixedbread-ai/mxbai-colbert-large-v1',
        
    },
    'monobert':{
        "monobert-large": "castorini/monobert-large-msmarco"
    },
    'monobert_ranker':{  # 🔑 修复命名不一致：添加monobert_ranker条目以匹配METHOD_MAP
        "monobert-large": "castorini/monobert-large-msmarco"
    },
    'twolar':{
        'twolar-xl':"Dundalia/TWOLAR-xl"
    },
    'echorank':{
        'flan-t5-large' : 'google/flan-t5-large',
        'flan-t5-xl' : 'google/flan-t5-xl'
    },
}

PREDICTION_TOKENS = {
    "default": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "unicamp-dl/InRanker-small": ["▁false", "▁true"],
    "unicamp-dl/InRanker-base": ["▁false", "▁true"],
    "unicamp-dl/InRanker-3B": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
    "unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
    "unicamp-dl/monoptt5-small": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-base": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-large": ["▁Não", "▁Sim"],
    "unicamp-dl/monoptt5-3b": ["▁Não", "▁Sim"],
}