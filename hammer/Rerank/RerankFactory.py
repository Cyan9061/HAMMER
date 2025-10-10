from hammer.Rerank.Upr import UPR
from hammer.Rerank.ColbertRanker import ColBERTReranker
from hammer.Rerank.Flashrank import FlashRanker
from hammer.Rerank.Monot5 import MonoT5
from hammer.Rerank.Rankt5 import RankT5
from hammer.Rerank.Echorank import EchoRankReranker
from hammer.Rerank.Listt5 import ListT5
from hammer.Rerank.Twolar import TWOLAR
from hammer.Rerank.TransformerRanker import TransformerRanker
from hammer.Rerank.Monobert import MonoBERT
from hammer.Rerank.InRanker import InRanker
METHOD_MAP ={
    # Existing reranking methods
    'upr': UPR,
    'flashrank' : FlashRanker,
    'monot5': MonoT5,
    'rankt5': RankT5,
    'listt5': ListT5,
    'transformer_ranker': TransformerRanker,
    #'first_ranker':FirstModelReranker,
    #'lit5dist': LiT5DistillReranker,
    #'lit5score': LiT5ScoreReranker,
    #'vicuna_reranker': VicunaReranker,
    #'zephyr_reranker': ZephyrReranker,
    'colbert_ranker': ColBERTReranker,
    'twolar':TWOLAR,
    'echorank':EchoRankReranker,
    'monobert_ranker':MonoBERT,
    "inranker": InRanker
}
