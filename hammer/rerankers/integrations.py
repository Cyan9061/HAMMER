# hammer/rerankers/integrations.py

import threading
from typing import List, Optional, Tuple
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from hammer.logger import logger

try:
    from flashrank.rerank import RerankRequest, Reranker
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

class FlashRankPostprocessor(BaseNodePostprocessor):
    """一个适配器，用于将 FlashRank 集成到 LlamaIndex 的后处理流程中。"""
    def __init__(self, model_name: str = "flashrank/ms-marco-MiniLM-L-12-v2", top_n: int = 5):
        super().__init__()
        if not FLASHRANK_AVAILABLE:
            raise ImportError("FlashRank not available. Please install: pip install flashrank")
        self._reranker = Reranker(model_name=model_name)
        self._top_n = top_n

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []

        passages_to_rerank = [
            {"id": node.node_id, "text": node.get_content()} for node in nodes
        ]
        request = RerankRequest(query=query_bundle.query_str, passages=passages_to_rerank)
        reranked_results = self._reranker.rerank(request)

        node_map = {node.node_id: node for node in nodes}
        final_nodes = []
        for result in reranked_results:
            node_id = result["id"]
            if node_id in node_map:
                node = node_map[node_id]
                node.score = result["score"]
                final_nodes.append(node)
        
        return final_nodes[:self._top_n]

class SentenceTransformerReranker(BaseNodePostprocessor):
    """使用SentenceTransformers的CrossEncoder进行重排序"""
    
    def __init__(self, model, top_n: int = 5):
        super().__init__()
        self.model = model
        self._top_n = top_n
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []
        
        try:
            # 准备输入对
            pairs = [(query_bundle.query_str, node.get_content()) for node in nodes]
            
            # 获取重排序分数 - 禁用内部进度条
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # 更新节点分数
            for i, node in enumerate(nodes):
                node.score = float(scores[i])
            
            # 按分数排序并返回top_n
            sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
            return sorted_nodes[:self._top_n]
            
        except Exception as e:
            logger.error(f"SentenceTransformer reranking failed: {e}")
            return nodes[:self._top_n]  # 回退到原始顺序

class SingletonCrossEncoder:
    """单例CrossEncoder模型管理器 - 解决重复加载和GPU竞争问题"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: str = None):
        if device is None:
            # 自动检测设备 - 当使用CUDA_VISIBLE_DEVICES时，总是使用逻辑设备0
            try:
                import os
                if os.environ.get('CUDA_VISIBLE_DEVICES'):
                    # 有CUDA_VISIBLE_DEVICES环境变量时，使用逻辑设备0
                    device = 'cuda:0'
                else:
                    # 没有环境变量时，尝试从main_tuner_mcts获取
                    from hammer.tuner.main_tuner_mcts import DEVICE_ID
                    device = f'cuda:{DEVICE_ID}'
            except:
                device = 'cuda:0'
        
        logger.info(f"🚀 初始化单例CrossEncoder模型: {model_name}, 设备: {device}")
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device)
            self.device = device
            self.model_name = model_name
            logger.info(f"✅ CrossEncoder模型加载成功")
        except Exception as e:
            logger.error(f"❌ CrossEncoder模型加载失败: {e}")
            raise
    
    @classmethod 
    def get_instance(cls, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: str = None):
        """线程安全的单例获取方法"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = cls(model_name, device)
        return cls._instance
    
    def batch_predict(self, query_doc_pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        """批量预测query-document相关性分数"""
        if not query_doc_pairs:
            return []
        
        import time
        start_time = time.time()
        
        logger.info(f"🔄 GPU批量预测: {len(query_doc_pairs)} 个query-document对, batch_size={batch_size}")
        
        # 计算批次信息
        num_batches = (len(query_doc_pairs) + batch_size - 1) // batch_size
        logger.info(f"📊 GPU推理配置: 设备={self.device}, 批次数={num_batches}")
        
        try:
            # 使用sentence_transformers的批处理能力
            scores = self.model.predict(
                query_doc_pairs, 
                batch_size=batch_size,
                show_progress_bar=False  # 禁用内部进度条
            )
            
            total_time = time.time() - start_time
            throughput = len(query_doc_pairs) / total_time
            
            logger.info(f"✅ GPU批量预测完成: {len(scores)}个分数")
            logger.info(f"📊 GPU性能: 耗时{total_time:.2f}s, 吞吐量{throughput:.1f} pairs/s")
            
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"❌ GPU批量预测失败: {e}")
            logger.error(f"🔍 失败详情: 输入对数量={len(query_doc_pairs)}, batch_size={batch_size}")
            raise

class BatchSentenceTransformerReranker(BaseNodePostprocessor):
    """批处理版本的SentenceTransformer Reranker - 真正的批处理优化"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', 
                 top_n: int = 5, batch_size: int = 32):
        super().__init__()
        
        # 使用object.__setattr__绕过Pydantic验证
        object.__setattr__(self, 'model_name', model_name)
        object.__setattr__(self, '_top_n', top_n)
        object.__setattr__(self, 'batch_size', batch_size)
        
        # 获取单例模型
        object.__setattr__(self, 'cross_encoder', SingletonCrossEncoder.get_instance(model_name))
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], 
                          query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        """单个查询的后处理 - 保持兼容性"""
        if query_bundle is None:
            raise ValueError("Query bundle is required for reranking.")
        if not nodes:
            return []
        
        try:
            # 准备输入对
            pairs = [(query_bundle.query_str, node.get_content()) for node in nodes]
            
            # 批量预测
            scores = self.cross_encoder.batch_predict(pairs, self.batch_size)
            
            # 更新节点分数
            for i, node in enumerate(nodes):
                node.score = float(scores[i])
            
            # 按分数排序并返回top_n
            sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
            return sorted_nodes[:self._top_n]
            
        except Exception as e:
            logger.error(f"BatchSentenceTransformer reranking failed: {e}")
            raise  # 不提供fallback，直接报错
    
    def batch_rerank_all_queries(self, all_queries: List[str], 
                                all_nodes_list: List[List[NodeWithScore]]) -> List[List[NodeWithScore]]:
        """真正的批处理reranking - 扁平化处理所有query-document对"""
        
        if len(all_queries) != len(all_nodes_list):
            raise ValueError("queries和nodes_list长度不匹配")
        
        import time
        start_time = time.time()
        logger.info(f"🚀 开始真正批处理reranking: {len(all_queries)}个查询")
        
        # 1. 统计总的文档数量
        total_docs = sum(len(nodes) for nodes in all_nodes_list)
        logger.info(f"📊 总文档数量: {total_docs}个，平均每查询: {total_docs/len(all_queries):.1f}个")
        
        # 1. 扁平化所有query-document对并记录映射
        flatten_start = time.time()
        logger.info("🔄 阶段1: 开始扁平化所有query-document对...")
        
        flat_pairs = []
        pair_to_query_idx = []  # 记录每个pair属于哪个查询
        pair_to_node_idx = []   # 记录每个pair属于该查询的哪个node
        
        for query_idx, (query, nodes) in enumerate(zip(all_queries, all_nodes_list)):
            if query_idx % 100 == 0:
                logger.info(f"   处理查询 {query_idx+1}/{len(all_queries)} ({(query_idx+1)/len(all_queries)*100:.1f}%)")
            
            for node_idx, node in enumerate(nodes):
                flat_pairs.append((query, node.get_content()))
                pair_to_query_idx.append(query_idx)
                pair_to_node_idx.append(node_idx)
        
        flatten_time = time.time() - flatten_start
        logger.info(f"✅ 扁平化完成: {len(flat_pairs)}个query-document对，耗时: {flatten_time:.2f}s")
        
        if not flat_pairs:
            logger.warning("⚠️ 没有query-document对需要处理")
            return [[] for _ in all_queries]
        
        # 2. 一次性批量预测所有分数
        predict_start = time.time()
        logger.info(f"🔄 阶段2: 开始批量GPU推理...")
        logger.info(f"📊 批处理参数: batch_size={self.batch_size}, 预计批次数: {len(flat_pairs)//self.batch_size + 1}")
        
        try:
            flat_scores = self.cross_encoder.batch_predict(flat_pairs, self.batch_size)
            predict_time = time.time() - predict_start
            logger.info(f"✅ 批量预测完成: {len(flat_scores)}个分数，耗时: {predict_time:.2f}s")
            logger.info(f"📊 GPU推理速度: {len(flat_scores)/predict_time:.1f} pairs/s")
        except Exception as e:
            logger.error(f"❌ 批量预测失败: {e}")
            raise
        
        # 3. 重新组织结果
        reorg_start = time.time()
        logger.info("🔄 阶段3: 开始重新组织结果...")
        
        results = [[] for _ in all_queries]
        
        for pair_idx, score in enumerate(flat_scores):
            if pair_idx % 5000 == 0 and pair_idx > 0:
                logger.info(f"   重组进度: {pair_idx}/{len(flat_scores)} ({pair_idx/len(flat_scores)*100:.1f}%)")
            
            query_idx = pair_to_query_idx[pair_idx]
            node_idx = pair_to_node_idx[pair_idx]
            
            # 更新对应node的分数
            node = all_nodes_list[query_idx][node_idx]
            node.score = float(score)
            
            # 将node添加到对应查询的结果中
            results[query_idx].append(node)
        
        reorg_time = time.time() - reorg_start
        logger.info(f"✅ 结果重组完成，耗时: {reorg_time:.2f}s")
        
        # 4. 对每个查询的结果按分数排序并截取top_k
        sort_start = time.time()
        logger.info(f"🔄 阶段4: 排序并截取top-{self._top_n}结果...")
        
        final_results = []
        for query_idx, nodes in enumerate(results):
            if nodes:
                sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
                final_results.append(sorted_nodes[:self._top_n])
            else:
                final_results.append([])
        
        sort_time = time.time() - sort_start
        total_time = time.time() - start_time
        
        logger.info(f"✅ 排序完成，耗时: {sort_time:.2f}s")
        logger.info(f"🎉 批处理reranking完成: {len([r for r in final_results if r])}个查询有结果")
        logger.info(f"⏱️ 总耗时: {total_time:.2f}s (扁平化: {flatten_time:.2f}s, GPU推理: {predict_time:.2f}s, 重组: {reorg_time:.2f}s, 排序: {sort_time:.2f}s)")
        logger.info(f"📊 整体处理速度: {len(all_queries)/total_time:.1f} queries/s, {total_docs/total_time:.1f} docs/s")
        
        return final_results