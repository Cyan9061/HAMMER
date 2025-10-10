#!/usr/bin/env python3
"""
批量处理query selection的脚本，基于run_selection_graph.py
适配多个数据集的处理需求
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import faiss
import random
import networkx as nx
from tqdm import tqdm
from typing import List, Any
from types import SimpleNamespace
from pathlib import Path

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class BatchL2FaissSelector:
    """
    批量处理的L2FaissSelector，适配多数据集处理
    """
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        print(f"Loading embedding model: {model_name}...")
        self.model = HuggingFaceEmbedding(model_name=model_name)
        Settings.embed_model = self.model
        self.embeddings = None
        self.index = None

    def _build_index(self, dataset: List[SimpleNamespace]):
        """构建索引"""
        if not dataset:
            print("Error: Dataset is empty, cannot build index.")
            return

        print("Step 1: Explicitly generating embeddings for all texts...")
        texts = [data.query for data in dataset]  # 注意这里用query字段
        embeddings_list = self.model.get_text_embedding_batch(texts, show_progress=True)
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        
        d = self.embeddings.shape[1]
        print(f"Embedding generation complete. Dimension: {d}")

        print("Step 2: Building Faiss index and LlamaIndex components...")
        faiss_index = faiss.IndexHNSWFlat(d, 32)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(
            [],
            storage_context=storage_context,
        )

        print("Step 3: Creating and inserting nodes with pre-computed embeddings...")
        nodes = []
        for i, data in enumerate(dataset):
            node = TextNode(
                id_=str(i),
                text=data.query,  # 注意这里用query字段
                embedding=self.embeddings[i].tolist()
            )
            nodes.append(node)
        
        self.index.insert_nodes(nodes, show_progress=True)
        print("L2 distance index built successfully using a robust workflow.")

    def _build_dissimilarity_graph(self, dataset: List[SimpleNamespace], similarity_threshold: float, num_samples_per_node: int, exclude_top_k: int):
        """构建不相似性图"""
        if self.index is None or self.embeddings is None:
            raise ValueError("Index and embeddings must be built first.")

        num_nodes = len(self.embeddings)
        print(f"Building dissimilarity graph (cosine_sim < {similarity_threshold}, samples_per_node={num_samples_per_node})...")

        graph = {i: {} for i in range(num_nodes)}

        retriever = self.index.as_retriever(
            similarity_top_k=exclude_top_k,
            embed_model=self.model
        )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normed_embeddings = self.embeddings / (norms + 1e-10)

        edges_added = 0
        for i in tqdm(range(num_nodes), desc="Building dissimilarity graph (Cosine Sim)"):
            query_embedding = normed_embeddings[i]
            query_text = dataset[i].query

            query_bundle = QueryBundle(query_str=query_text, embedding=query_embedding)
            similar_results = retriever.retrieve(query_bundle)
            similar_indices = {int(res.node.node_id) for res in similar_results}
            similar_indices.add(i)

            all_indices = set(range(num_nodes))
            pool_indices = list(all_indices - similar_indices)
            if not pool_indices:
                continue

            sampled_indices = random.sample(pool_indices, min(num_samples_per_node, len(pool_indices)))
            if not sampled_indices:
                continue

            sampled_embeddings = normed_embeddings[sampled_indices]
            cosine_similarities = query_embedding @ sampled_embeddings.T

            for j_local, j_global in enumerate(sampled_indices):
                sim = cosine_similarities[j_local]
                if sim < similarity_threshold:
                    weight = 1.0 - sim
                    graph[i][j_global] = weight
                    graph[j_global][i] = weight
                    edges_added += 1

        print(f"Dissimilarity graph built successfully. Added {edges_added} edges.")
        return graph

    def _greedy_peeling(self, graph, nodeset):
        """贪心剥离算法"""
        if not nodeset: 
            return set(), 0.0
        subgraph_nodes = set(nodeset)
        degrees = {u: sum(weight for v, weight in graph.get(u, {}).items() if v in subgraph_nodes) for u in subgraph_nodes}
        if not degrees: 
            return set(), 0.0
        total_weight_in_subgraph = sum(degrees.values()) / 2.0
        nodes_in_subgraph = len(degrees)
        best_density = total_weight_in_subgraph / nodes_in_subgraph if nodes_in_subgraph > 0 else 0
        best_subgraph_at_peel = set(degrees.keys())
        while nodes_in_subgraph > 1:
            min_degree_node = min(degrees, key=degrees.get)
            node_degree = degrees[min_degree_node]
            total_weight_in_subgraph -= node_degree
            del degrees[min_degree_node]
            nodes_in_subgraph -= 1
            for neighbor, weight in graph.get(min_degree_node, {}).items():
                if neighbor in degrees:
                    degrees[neighbor] -= weight
            current_density = total_weight_in_subgraph / nodes_in_subgraph if nodes_in_subgraph > 0 else 0
            if current_density > best_density:
                best_density = current_density
                best_subgraph_at_peel = set(degrees.keys())
        return best_subgraph_at_peel, best_density

    def _calculate_density(self, graph, nodeset):
        """计算节点集的密度"""
        if not nodeset or len(nodeset) == 0:
            return 0.0
        
        total_weight = 0.0
        nodeset_set = set(nodeset)
        for u in nodeset_set:
            for v, weight in graph[u].items():
                if v in nodeset_set:
                    total_weight += weight
        
        num_edges_or_total_weight = total_weight / 2
        num_nodes = len(nodeset_set)
        
        return num_edges_or_total_weight / num_nodes if num_nodes > 0 else 0.0
    
    def _combinatorial_dalkss(self, graph, k_min):
        """DalkSS算法实现"""
        print(f"目标数量 k_min={k_min}，开始DalkSS算法...")
        
        D_current = set()
        nodes_to_consider = set(graph.keys())
        candidate_sets = []
        
        while len(D_current) < k_min and nodes_to_consider:
            H_j, _ = self._greedy_peeling(graph, nodes_to_consider)
            
            if not H_j:
                print("  警告: 在剩余节点中无法找到更多密集区域，循环终止。")
                break
            
            D_current.update(H_j)
            nodes_to_consider.difference_update(H_j)
            candidate_sets.append(D_current.copy())

        if not candidate_sets:
             print("未能生成任何候选集，算法终止。")
             return []

        print(f"共生成 {len(candidate_sets)} 个候选集。现在进行任意添加并评估...")

        best_final_set = set()
        max_final_density = -1.0
        all_nodes = set(graph.keys())

        for d_j in tqdm(candidate_sets, desc="评估候选集"):
            d_prime_j = d_j.copy()
            num_to_add = k_min - len(d_prime_j)

            if num_to_add > 0:
                nodes_available_to_add = list(all_nodes - d_prime_j)
                
                if len(nodes_available_to_add) < num_to_add:
                    print(f"警告: 可供添加的节点不足({len(nodes_available_to_add)})，无法满足 k_min={k_min}。跳过此候选集。")
                    continue
                
                nodes_to_add = random.sample(nodes_available_to_add, num_to_add)
                d_prime_j.update(nodes_to_add)
            
            current_density = self._calculate_density(graph, d_prime_j)

            if current_density > max_final_density:
                max_final_density = current_density
                best_final_set = d_prime_j

        if not best_final_set:
            print("警告: 未能找到任何有效的最终集合。")
            return []

        print(f"DalkSS算法完成。选出 {len(best_final_set)} 个节点，密度为 {max_final_density:.4f}。")
        return list(best_final_set)

    def select_queries(self, dataset, num_queries_to_select, similarity_threshold=0.3, num_samples_per_node=100, exclude_top_k=30):
        """主选择函数"""
        if not dataset or len(dataset) < num_queries_to_select:
            print("Error: Dataset is empty or smaller than the number of queries to select.")
            return [], None, None
        
        self._build_index(dataset)
        
        graph = self._build_dissimilarity_graph(
            dataset=dataset,
            similarity_threshold=similarity_threshold,
            num_samples_per_node=num_samples_per_node,
            exclude_top_k=exclude_top_k
        )

        selected_indices = self._combinatorial_dalkss(graph, k_min=num_queries_to_select)
        full_graph = graph
        if not selected_indices: 
            return [], None, None
        
        return [dataset[i] for i in selected_indices], selected_indices, full_graph

def load_jsonl(file_path):
    """加载JSONL格式文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """保存为JSONL格式文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_dataset(dataset_name, model_name=os.getenv("DEFAULT_EMBEDDING_MODEL", "")):
    """处理单个数据集的query selection"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    temp_dir = Path('../../docs/dataset/unified_query_selection/temp')
    
    # 加载选择输入数据和测试数据
    selection_input_file = temp_dir / f'{dataset_name}_selection_input.jsonl'
    test_file = temp_dir / f'{dataset_name}_test.jsonl'
    remaining_train_file = temp_dir / f'{dataset_name}_remaining_train.jsonl'
    
    if not selection_input_file.exists():
        print(f"文件不存在: {selection_input_file}")
        return None
        
    selection_input_data = load_jsonl(selection_input_file)
    test_data = load_jsonl(test_file)
    remaining_train_data = load_jsonl(remaining_train_file)
    
    print(f"选择输入数据: {len(selection_input_data)} 条")
    print(f"测试数据: {len(test_data)} 条") 
    print(f"剩余训练数据: {len(remaining_train_data)} 条")
    
    # 转换为SimpleNamespace对象
    data_objects = [SimpleNamespace(**d) for d in selection_input_data]
    
    # 计算目标选择数量（30%的训练集）
    total_original_size = len(selection_input_data) + len(test_data) + len(remaining_train_data)
    target_selection_size = int(len(selection_input_data) * 1.0)  # 从30%的训练集中选择所有
    
    print(f"原始数据集总计: {total_original_size} 条")
    print(f"目标选择数量: {target_selection_size} 条")
    
    # 调整参数确保结果不超过50%
    max_allowed_size = int(total_original_size * 0.5)
    if target_selection_size > max_allowed_size:
        target_selection_size = max_allowed_size
        print(f"调整目标选择数量为: {target_selection_size} 条 (50%限制)")
    
    # 根据数据集大小调整参数
    if len(selection_input_data) < 500:
        # 小数据集参数
        similarity_threshold = 0.4
        num_samples_per_node = 50
        exclude_top_k = 10
    elif len(selection_input_data) < 1500:
        # 中等数据集参数
        similarity_threshold = 0.35
        num_samples_per_node = 80
        exclude_top_k = 20
    else:
        # 大数据集参数
        similarity_threshold = 0.3
        num_samples_per_node = 100
        exclude_top_k = 30
    
    print(f"参数设置: similarity_threshold={similarity_threshold}, num_samples_per_node={num_samples_per_node}, exclude_top_k={exclude_top_k}")
    
    # 进行query selection
    selector = BatchL2FaissSelector(model_name=model_name)
    
    selected_data, selected_indices, full_graph = selector.select_queries(
        dataset=data_objects,
        num_queries_to_select=target_selection_size,
        similarity_threshold=similarity_threshold,
        num_samples_per_node=num_samples_per_node,
        exclude_top_k=exclude_top_k
    )
    
    if not selected_data:
        print(f"错误: 数据集 {dataset_name} 的选择过程失败")
        return None
    
    print(f"实际选择了 {len(selected_data)} 条数据")
    
    # 转换回字典格式
    selected_dicts = [vars(item) for item in selected_data]
    
    # 合并选择结果和测试数据（不包括剩余训练数据）
    final_data = selected_dicts + test_data
    
    print(f"最终数据集大小: {len(final_data)} 条")
    print(f"训练数据（选择后）: {len(selected_dicts)} 条")
    print(f"测试数据: {len(test_data)} 条")
    print(f"选择比例（相对于原数据集）: {len(selected_dicts) / total_original_size:.2%}")
    print(f"最终数据集比例（相对于原数据集）: {len(final_data) / total_original_size:.2%}")
    
    # 保存结果
    output_dir = Path('../../docs/dataset/unified_query_selection')
    output_file = output_dir / f'{dataset_name}_qa_unified.json'
    
    save_jsonl(final_data, output_file)
    
    # 保存选择的具体数据（用于分析）
    temp_selected_file = temp_dir / f'{dataset_name}_selected.jsonl'
    save_jsonl(selected_dicts, temp_selected_file)
    
    print(f"结果已保存到: {output_file}")
    
    return {
        'dataset': dataset_name,
        'original_size': total_original_size,
        'selected_size': len(selected_dicts),
        'test_size': len(test_data),
        'final_size': len(final_data),
        'selection_ratio': len(selected_dicts) / total_original_size,
        'final_ratio': len(final_data) / total_original_size,
        'parameters': {
            'similarity_threshold': similarity_threshold,
            'num_samples_per_node': num_samples_per_node,
            'exclude_top_k': exclude_top_k
        }
    }

def main():
    """主函数"""
    print("="*60)
    print("批量Query Selection处理")
    print("="*60)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 数据集列表
    datasets = [
        '2wikimultihopqa',
        'FinQA',
        'MedQA', 
        'bioasq',
        'hotpotqa',
        'musique'
    ]
    
    # 本地模型路径
    local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
    
    results = []
    
    for dataset_name in datasets:
        try:
            result = process_dataset(dataset_name, model_name=local_model_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {e}")
    
    # 保存处理结果统计
    stats_file = '../../docs/dataset/unified_query_selection/temp/selection_results.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印最终统计
    print("\n" + "="*70)
    print("Query Selection处理结果汇总:")
    print("="*70)
    print(f"{'数据集':15} | {'原始':5} | {'选择':5} | {'测试':4} | {'最终':5} | {'选择比例':8} | {'最终比例':8}")
    print("-"*70)
    
    total_original = 0
    total_selected = 0
    total_test = 0
    total_final = 0
    
    for result in results:
        print(f"{result['dataset']:15} | {result['original_size']:5} | {result['selected_size']:5} | "
              f"{result['test_size']:4} | {result['final_size']:5} | {result['selection_ratio']:8.2%} | {result['final_ratio']:8.2%}")
        total_original += result['original_size']
        total_selected += result['selected_size'] 
        total_test += result['test_size']
        total_final += result['final_size']
    
    print("-"*70)
    print(f"{'总计':15} | {total_original:5} | {total_selected:5} | {total_test:4} | {total_final:5} | "
          f"{total_selected/total_original:8.2%} | {total_final/total_original:8.2%}")
    
    print(f"\n处理结果统计已保存到: {stats_file}")

if __name__ == '__main__':
    main()